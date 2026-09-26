"""Execute one small condition, requesting only outputs absent from verified reuse."""
import argparse
from dataclasses import replace
import json
import traceback
from pathlib import Path
from _small_experiment import (read_json,write_json,condition_for,classifier_for,reused_rows,consolidate)
from llm_classifier_bench.budgets import MatchedBudgetConfig
from llm_classifier_bench.measurement import measurement_from_args
from llm_classifier_bench.runner import BenchmarkRunConfig,run_benchmark
from run_banking77_scaling_benchmark_v2 import StaticDataset


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--cell',required=True)
    parser.add_argument('--output',type=Path,required=True)
    opts=parser.parse_args();manifest=read_json(opts.manifest)
    cell=next(c for c in manifest['cells'] if c['id']==opts.cell)
    if cell['availability']!='ready':raise ValueError('Blocked cell cannot run')
    opts.output.mkdir(parents=True,exist_ok=False)
    args,condition=condition_for(manifest,cell,opts.output/'definitions')
    old,model=reused_rows(manifest,cell,args,condition)
    done={r['sample_id'] for r in old};pending=tuple(e for e in condition.bundle.test if e.sample_id not in done)
    (opts.output/'reused_predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in old))
    write_json(opts.output/'condition.json',{'cell':cell,'classes':[{'name':c.name,'description':c.description} for c in condition.bundle.classes],
        'test_sample_ids':[e.sample_id for e in condition.bundle.test],
        'reused_run':manifest.get('reuse',{}).get(cell['id'],{}).get('run_dir'),
        'reused_examples':len(old),'pending_examples':len(pending)})
    print(f"CONDITION {cell['id']} | total={len(condition.bundle.test)} reused={len(old)} new={len(pending)}",flush=True)
    if pending:
        classifier=classifier_for(args,cell,condition,emissary_model=model if cell['method']=='emissary' else None,request_cap=len(pending))
        try:
            run_benchmark(StaticDataset(replace(condition.bundle,test=pending)),classifier,
                BenchmarkRunConfig(output_root=opts.output/'runs',run_id='new',
                    validation_fraction=args.validation_fraction,split_seed=cell['seed'],
                    class_definitions_path=condition.definitions_path,pricing_path=args.pricing,
                    measurement=measurement_from_args(args),
                    matched_budget=MatchedBudgetConfig(cell['budget'],'per_class',cell['seed'],0),
                    metadata={'small_campaign':True,'cell_id':cell['id'],'seed':cell['seed'],
                              'class_count':cell['class_count'],'full_test_examples':len(condition.bundle.test),
                              'reused_examples':len(old),'reused_run':manifest.get('reuse',{}).get(cell['id'],{}).get('run_dir')}))
        except KeyboardInterrupt:
            consolidate(opts.output)
            raise
        except Exception as exc:
            print(f'ERROR {type(exc).__name__}: {exc}',flush=True)
            traceback.print_exc()
    result=consolidate(opts.output)
    print(f"RESULT {result['status']} | valid={result['valid_examples']}/{result['planned_examples']} | reused={result['reused_examples']}",flush=True)

    if result['status']!='completed':
        raise SystemExit(2)


if __name__=='__main__':main()
