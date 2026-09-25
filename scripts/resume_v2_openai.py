"""Continue only missing OpenAI predictions; retain each attempt as separate evidence.

Rebuild and verify the exact frozen labeled pool and prompt hashes. The original
failed attempts remain untouched. Pacing is deliberately inside predict and is
therefore included in continuation latency, labeled explicitly in metadata.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

from run_banking77_scaling_benchmark_v2 import build_condition, StaticDataset
from llm_classifier_bench.budgets import MatchedBudgetConfig, select_matched_pools
from llm_classifier_bench.class_definitions import load_class_definition_profile
from llm_classifier_bench.classifiers.openai import OpenAIClassifier
from llm_classifier_bench.datasets import get_dataset
from llm_classifier_bench.datasets.huggingface import HuggingFaceClassificationDataset
from llm_classifier_bench.measurement import MeasurementConfig
from llm_classifier_bench.recovery import pending_examples
from llm_classifier_bench.runner import BenchmarkRunConfig, run_benchmark, split_train_validation


class PacedOpenAI(OpenAIClassifier):
    def __init__(self, *, interval_s, **kwargs):
        super().__init__(**kwargs)
        self.interval_s = interval_s
        self.last_start = None

    def predict(self, examples):
        if self.last_start is not None:
            time.sleep(max(0., self.last_start + self.interval_s - time.monotonic()))
        self.last_start = time.monotonic()
        return super().predict(examples)

    def inference_metadata(self):
        return {**super().inference_metadata(), 'minimum_start_interval_s':self.interval_s,
                'pacing_boundary':'inside predict; continuation latency includes pacing; do not pool with original latency'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('artifacts/v2_release'))
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--budgets',type=int,nargs='+',default=[0,5,100])
    args=parser.parse_args(); root=args.root
    manifest=json.loads((root/'matrix.json').read_text())
    for path, digest in manifest['input_sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:
            raise ValueError(f'Frozen input changed: {path}')
    ds=get_dataset('banking77')
    full=HuggingFaceClassificationDataset(replace(ds.spec,data_files={s:str(Path('artifacts/v2_source',s+'.csv').resolve()) for s in ('train','test')})).load()
    profile=load_class_definition_profile('class_definitions_data/banking77/canonical_reviewed_v2.json')
    for cell in sorted(manifest['cells'],key=lambda c:(c['budget'] if c['budget'] is not None else -1,c['seed'],c['class_count'])):
        if cell['method']!='openai' or cell['budget'] not in args.budgets:
            continue
        original=list((root/'cells'/cell['id']).glob('*/runs/*/status.json'))
        if len(original)!=1: continue
        run=original[0].parent; state=json.loads(original[0].read_text())
        if state['status']!='failed' or state.get('error_type')!='RateLimitError': continue
        config=json.loads((run/'config.json').read_text())
        destination=root/'recovery'/cell['id']; definitions=destination/'definitions'; definitions.mkdir(parents=True,exist_ok=True)
        condition=build_condition(full_bundle=full,loaded_profile=profile,
            master_order=[c['name'] for c in config['dataset']['classes']],class_count=cell['class_count'],seed=cell['seed'],
            train_per_class=125,test_per_class=20,validation_fraction=.2,definitions_dir=definitions)
        budget=MatchedBudgetConfig(**config['matched_budget'])
        fit,val=split_train_validation(condition.bundle.train,validation_fraction=.2,seed=cell['seed'])
        fit,val,pool=select_matched_pools(fit,val,condition.bundle.classes,budget)
        oldpool=json.loads((run/'labeled_budget.json').read_text())
        if pool['pool_sha256']!=oldpool['pool_sha256']:
            raise ValueError('Recovery labeled pool mismatch')
        if [e.sample_id for e in condition.bundle.test]!=config['dataset']['test_sample_ids']:
            raise ValueError('Recovery test IDs/order mismatch')
        artifacts=[run/'predictions.jsonl',*sorted(destination.glob('attempt-*/predictions.jsonl'))]
        pending, completed=pending_examples(condition.bundle.test,artifacts)
        if not pending: print('COMPLETE',cell['id'],flush=True);continue
        model=config['classifier']; plan=json.loads((run/'context_plan.json').read_text())
        # Estimate rate-limit reservation from prior observed provider usage when available.
        # A conservative byte proxy divided by four is used when no usage was returned.
        interval=max(1.,max(r['estimated_total_tokens'] for r in plan['requests'])/4/160000*60)
        classifier=PacedOpenAI(interval_s=interval,model=model['model'],reasoning_effort=model['reasoning_effort'],
            classifier_name=model['name'],in_context=model['in_context'],
            context_window_tokens=model['context_window_tokens'],completion_reserve_tokens=model['completion_reserve_tokens'],
            framing_allowance_tokens=model['framing_allowance_tokens'])
        newplan=classifier.plan_context(condition.bundle.classes,fit,[e.as_input() for e in pending])
        oldrequests={r['sample_id']:r for r in plan['requests']}
        if any(r!=oldrequests[r['sample_id']] for r in newplan['requests']):
            raise ValueError('Recovery prompt hash mismatch')
        print('PENDING',cell['id'],len(pending),'retained',len(completed),'interval_s',interval,flush=True)
        if not args.execute: continue
        number=len(list(destination.glob('attempt-*')))+1
        try:
            run_benchmark(StaticDataset(replace(condition.bundle,test=pending)),classifier,
                BenchmarkRunConfig(output_root=destination,run_id=f'attempt-{number:03d}',
                    validation_fraction=.2,split_seed=cell['seed'],matched_budget=budget,
                    class_definitions_path=condition.definitions_path,pricing_path=Path(config['pricing_path']),
                    measurement=MeasurementConfig(**config['measurement']),
                    metadata={**config['run_metadata'],'recovery_of':str(run),'retained_prediction_count':len(completed),
                              'minimum_start_interval_s':interval,'protocol_amendment':'recovery-1; pacing included in latency'}))
        except Exception as exc:
            print('FAILED',cell['id'],type(exc).__name__,flush=True)


if __name__=='__main__': main()
