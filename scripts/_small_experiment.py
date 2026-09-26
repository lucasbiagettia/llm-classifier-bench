"""Frozen small-condition construction and offline reuse/consolidation helpers."""
from dataclasses import asdict, replace
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys

import run_banking77_scaling_benchmark_v2 as campaign
from llm_classifier_bench.class_definitions import load_class_definition_profile
from llm_classifier_bench.classifiers.emissary import EmissaryClassifier
from llm_classifier_bench.config import EmissaryTrainingConfig
from llm_classifier_bench.datasets import get_dataset
from llm_classifier_bench.datasets.huggingface import HuggingFaceClassificationDataset
from llm_classifier_bench.metrics.evaluator import DEFAULT_METRICS, evaluate_records, evaluation_record_from_mapping
from _jev_campaign import jev_options


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arguments(manifest, cell):
    original = sys.argv
    try:
        sys.argv = ['small', *manifest['common_arguments'], '--classifiers', cell['method'],
                    '--class-counts', str(cell['class_count']), '--seeds', str(cell['seed'])]
        return campaign.parse_args()
    finally:
        sys.argv = original


@lru_cache(maxsize=2)
def source_bundle(directory):
    dataset = get_dataset('banking77')
    files = {s: str((Path(directory)/f'{s}.csv').resolve()) for s in ('train','test')}
    bundle = HuggingFaceClassificationDataset(replace(dataset.spec, data_files=files)).load()
    return replace(bundle, metadata={**bundle.metadata, 'source_sha256':{s:sha256(p) for s,p in files.items()}})


def condition_for(manifest, cell, definitions_dir):
    args = arguments(manifest, cell)
    full = source_bundle(str(args.source_dir))
    profile = load_class_definition_profile(args.definitions)
    support = campaign.resolve_eligibility_plan(canonical_names=full.class_names,
        train_examples=full.train, test_examples=full.test, requested_train_per_class=args.train_per_class,
        requested_test_per_class=args.test_per_class, required_class_count=max(manifest['class_counts']),
        min_train_per_class=args.min_train_per_class, min_test_per_class=args.min_test_per_class,
        strict_support=True)
    definitions_dir.mkdir(parents=True,exist_ok=True)
    condition = campaign.build_condition(full_bundle=full, loaded_profile=profile,
        master_order=campaign.choose_master_label_order(support.eligible_names, seed=cell['seed']),
        class_count=cell['class_count'], seed=cell['seed'], train_per_class=args.train_per_class,
        test_per_class=args.test_per_class, validation_fraction=args.validation_fraction,
        definitions_dir=definitions_dir)
    return args, condition


def classifier_for(args, cell, condition, *, emissary_model=None, request_cap=None):
    training = EmissaryTrainingConfig(shots=0,shot_unit='per_class',selection_seed=cell['seed'])
    if cell['method']=='emissary' and emissary_model:
        return EmissaryClassifier(training=training,model_id=emissary_model)
    options=jev_options(args);options['max_requests']=request_cap or cell['test_examples']
    return campaign.build_classifier(cell['method'], condition=condition, campaign_id='small-v2',
        openai_model=args.openai_model,openai_reasoning_effort=args.openai_reasoning_effort,
        bert_model=args.bert_model,bert_epochs=args.bert_epochs,bert_batch_size=args.bert_batch_size,
        bert_learning_rate=args.bert_learning_rate,bert_weight_decay=args.bert_weight_decay,
        bert_max_length=args.bert_max_length,sentence_transformer_model=args.sentence_transformer_model,
        st_embedding_batch_size=args.st_embedding_batch_size,st_c_values=tuple(args.st_c_values),
        st_max_iter=args.st_max_iter,tfidf_training=campaign.tfidf_training_config(args,cell['seed']),
        emissary_training=training,jev_config=options,
        openai_options={'in_context':True,'classifier_name':'openai-zero-shot',
                        'context_window_tokens':args.openai_context_window_tokens,
                        'completion_reserve_tokens':args.openai_completion_reserve_tokens,
                        'framing_allowance_tokens':args.openai_framing_allowance_tokens})


def validate_reuse(run, args, cell, condition):
    """Only complete zero-shot artifacts under identical inputs/model settings qualify."""
    run=Path(run); config=read_json(run/'config.json'); fit=read_json(run/'fit_metadata.json')
    if cell['method'] not in ('openai','emissary','jev') or cell['budget']!=0:
        raise ValueError('Reuse is restricted to zero-shot providers')
    if read_json(run/'status.json')['status']!='completed':
        raise ValueError('Incomplete historical runs are not reusable')
    if config['dataset']['classes'] != [asdict(c) for c in condition.bundle.classes]:
        raise ValueError('Class order/definitions differ')
    if config['dataset']['metadata'].get('source_sha256') != condition.bundle.metadata['source_sha256']:
        raise ValueError('Source identity differs')
    if config['run_metadata']['source_definition_profile_sha256'] != condition.bundle.metadata['source_definition_profile_sha256']:
        raise ValueError('Definition profile differs')
    if config.get('matched_budget',{}).get('examples')!=0 or any(fit.get(k,0) for k in
            ('training_examples_used','validation_examples_used','context_examples_used')):
        raise ValueError('Historical run consumed labels')
    if config['measurement']['batch_size']!=1 or config['measurement']['warmup_examples']!=0:
        raise ValueError('Historical measurement settings differ')
    classifier=classifier_for(args,cell,condition)
    if cell['method']=='openai':
        for key in ('model','reasoning_effort','in_context','context_window_tokens','completion_reserve_tokens','framing_allowance_tokens'):
            if config['classifier'].get(key)!=getattr(classifier,key):
                raise ValueError(f'OpenAI setting differs: {key}')
    if cell['method']=='jev':
        if config['classifier']['model']!=args.jev_model:
            raise ValueError('Jev model differs')
        if read_json(run/'inference_plan.json')['endpoint']!=classifier.endpoint:
            raise ValueError('Jev endpoint differs')
    expected={e.sample_id:e for e in condition.bundle.test}
    rows=[json.loads(line) for line in (run/'predictions.jsonl').read_text().splitlines()]
    if [r['sample_id'] for r in rows]!=config['dataset']['test_sample_ids']:
        raise ValueError('Historical prediction coverage differs')
    if cell['method']=='openai':
        old={r['sample_id']:r for r in read_json(run/'context_plan.json')['requests']}
        new=classifier.plan_context(condition.bundle.classes,(),condition.bundle.inputs())
        new={r['sample_id']:r for r in new['requests']}
        if any(new.get(key)!=request for key,request in old.items()):
            raise ValueError('OpenAI prompt hash differs')
    model=fit.get('model_id') if cell['method']=='emissary' else config['classifier']['model']
    for row in rows:
        example=expected.get(row['sample_id'])
        if example is None or (row['input'],row['gold_label'])!=(example.text,example.label):
            raise ValueError('Historical test identity differs')
        if row['model']!=model:
            raise ValueError('Historical resolved model differs')
        if cell['method']=='jev':
            encoded=classifier._encoded(classifier._payload(condition.bundle.classes,example.text))
            if hashlib.sha256(encoded).hexdigest()!=row['raw_response']['request_sha256']:
                raise ValueError('Jev request hash differs')
    return rows, model


def reused_rows(manifest,cell,args,condition):
    entry=manifest.get('reuse',{}).get(cell['id'])
    if not entry:return [],None
    run=Path(entry['run_dir'])
    for name,digest in entry['files_sha256'].items():
        if sha256(run/name)!=digest:raise ValueError(f'Reuse evidence changed: {name}')
    return validate_reuse(run,args,cell,condition)


def consolidate(cell_dir):
    """Merge quality only; keep old/new usage and latency populations separate."""
    cell_dir=Path(cell_dir); definition=read_json(cell_dir/'condition.json')
    expected=definition['test_sample_ids']; rows=[]; evidence=[]
    reuse=cell_dir/'reused_predictions.jsonl'
    if reuse.exists():
        old=[json.loads(line) for line in reuse.read_text().splitlines()]
        rows.extend(old);evidence.append({'phase':'reused','examples':len(old),'run_dir':definition['reused_run']})
    sources=sorted(cell_dir.glob('runs/*/predictions.jsonl'))
    for path in sources:
        new=[json.loads(line) for line in path.read_text().splitlines()]
        rows.extend(new);evidence.append({'phase':'new','examples':len(new),'run_dir':str(path.parent)})
    by_id={r['sample_id']:r for r in rows}
    if len(by_id)!=len(rows) or not set(by_id)<=set(expected):
        raise ValueError('Duplicate or unexpected combined prediction IDs')
    statuses=sorted(cell_dir.glob('runs/*/status.json'))
    new_status=read_json(statuses[-1]) if statuses else None
    attempt_complete=(new_status is not None and new_status['status']=='completed') if definition['pending_examples'] else True
    status='completed' if set(by_id)==set(expected) and attempt_complete else 'failed'
    ordered=[by_id[key] for key in expected if key in by_id]
    (cell_dir/'predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in ordered))
    result={'status':status,'planned_examples':len(expected),'valid_examples':len(ordered),
            'reused_examples':len(rows)-sum(e['examples'] for e in evidence if e['phase']=='new'),
            'evidence':evidence,'quality_only':True,
            'measurement_note':'Do not combine historical/new latency or charge historical usage as new spending.'}
    if status=='completed':
        quality=[m for m in DEFAULT_METRICS if m.name in ('accuracy','macro_f1','top_label_ece','adaptive_ece','multiclass_log_loss','multiclass_brier_score')]
        result['metrics']={m.name:m.as_dict() for m in evaluate_records(
            tuple(evaluation_record_from_mapping(r) for r in ordered),metrics=quality)}
        write_json(cell_dir/'metrics.json',result['metrics'])
    if new_status:result['new_attempt_status']=new_status
    write_json(cell_dir/'result.json',result)
    return result
