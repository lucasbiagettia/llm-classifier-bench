"""Rebuild English extended/brief reports using saved predictions only."""
from pathlib import Path
import csv
import json
import numpy as np
import hashlib
from llm_classifier_bench.metrics.evaluator import DEFAULT_METRICS,evaluate_records,evaluation_record_from_mapping
from llm_classifier_bench.metrics.uncertainty import stratified_quality_bootstrap

REPORT=Path('reports/v2')
CACHE=Path('.local/report-cache/v2')


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

METHODS={'tfidf':'TF-IDF + LR (50/class)','sentence-transformer':'MiniLM + LR (50/class)',
         'bert':'BERT (50/class)','openai':'OpenAI zero-shot','emissary':'Emissary routing zero-shot',
         'jev':'Jev zero-shot','emissary-qwen':'Emissary Qwen SFT (100/class)'}
QUALITY=[m for m in DEFAULT_METRICS if m.name in ('accuracy','macro_f1','top_label_ece','adaptive_ece','multiclass_log_loss','multiclass_brier_score')]


def records(path):return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
def num(value):return '—' if value is None else f'{value:.3f}'
def pct(value):return '—' if value is None else f'{100*value:.1f}%'


def collect(completion_root):
    expected={}
    for k in (5,10,15,20):
        p=Path(f'artifacts/v2_small/cells/openai__s42__k{k}__b0/predictions.jsonl')
        expected[k]={r['sample_id']:(r['input'],r['gold_label']) for r in records(p)}
        assert len(expected[k])==40*k
    results=[];complete_records={};measurements=[];evidence_hashes={}
    for method,title in METHODS.items():
        for k in (5,10,15,20):
            budget=100 if method=='emissary-qwen' else 50 if method in ('tfidf','sentence-transformer','bert') else 0
            key=f'{method}__s42__k{k}__b{budget}'
            root=Path('artifacts/v2_small/cells')/key
            candidate=completion_root/'cells'/key
            if method=='emissary-qwen' or (candidate/'result.json').exists():root=candidate
            path=root/'predictions.jsonl';rows=records(path) if path.exists() else []
            saved=read_json(root/'result.json') if (root/'result.json').exists() else {}
            ids=[r['sample_id'] for r in rows]
            if len(ids)!=len(set(ids)) or any(expected[k].get(r['sample_id'])!=(r['input'],r['gold_label']) for r in rows):
                raise ValueError(f'Invalid prediction identities: {root}')
            valid=saved.get('status')=='completed' and set(ids)==set(expected[k])
            item={'method':method,'title':title,'k':k,'fit_per_class':budget,'n':len(rows),'expected':40*k,
                  'status':'completed' if valid else 'incomplete' if rows else 'pending','evidence':str(root),
                  'probability_vectors':sum(bool(r.get('probabilities')) for r in rows)}
            if path.exists():evidence_hashes[str(path)]=sha256(path)
            if valid:
                rec=tuple(evaluation_record_from_mapping(r) for r in rows)
                metrics={m.name:m.as_dict() for m in evaluate_records(rec,metrics=QUALITY)}
                # Never report calibration of a success-selected subset as full-cohort calibration.
                if any(not r.get('probabilities') for r in rows):
                    for name in ('top_label_ece','adaptive_ece','multiclass_log_loss','multiclass_brier_score'):
                        metrics[name]={'name':name,'value':None,'available':False,'metadata':{'reason':'Full-cohort probabilities unavailable'}}
                item['metrics']=metrics
                cache=CACHE/f'{key}.json'
                if cache.exists() and read_json(cache).get('prediction_sha256')==sha256(path):ci=read_json(cache)
                else:
                    ci={**stratified_quality_bootstrap(rec),'prediction_sha256':sha256(path)};write_json(cache,ci)
                item['uncertainty']=ci;complete_records[method,k]=rec
                # Each phase retains its own latency/cost population; no historical/new pooling.
                for evidence in saved.get('evidence',[]):
                    if not evidence.get('run_dir') or not evidence.get('examples'):continue
                    source=Path(evidence['run_dir']);pred=source if source.is_file() else source/'predictions.jsonl'
                    if not pred.exists():continue
                    phase_rows=records(pred);latencies=[r['latency_ms'] for r in phase_rows if r.get('latency_ms') is not None]
                    measurements.append({'method':method,'k':k,'phase':evidence['phase'],
                        'evidence':str(source),'n':len(phase_rows),'latency_p50_ms':float(np.median(latencies)) if latencies else None,
                        'latency_p95_ms':float(np.quantile(latencies,.95)) if latencies else None,
                        'cost_note':'See original usage/cost ledgers including failed attempts; unknown charges are not zero.'})
            results.append(item)
    paired=[]
    for k in (5,10,15,20):
        if ('openai',k) not in complete_records:continue
        for method in METHODS:
            if method=='openai' or (method,k) not in complete_records:continue
            pairpath=CACHE/f'paired-{method}-minus-openai-k{k}.json'
            members=[r for r in results if r['k']==k and r['method'] in (method,'openai')]
            fingerprints={r['method']:r['uncertainty']['prediction_sha256'] for r in members}
            if pairpath.exists() and read_json(pairpath).get('prediction_sha256')==fingerprints:ci=read_json(pairpath)
            else:
                ci={**stratified_quality_bootstrap(complete_records[method,k],paired_records=complete_records['openai',k]),'prediction_sha256':fingerprints};write_json(pairpath,ci)
            paired.append({'method':method,'reference':'openai','k':k,'uncertainty':ci})
    completed=sum(r['status']=='completed' for r in results);draft=completed<len(results)
    return {'status':'interim' if draft else 'evaluated','conditions':results,
            'paired_effects':paired,'measurement_phases':measurements,'evidence_sha256':evidence_hashes}


def render(payload):
    REPORT.mkdir(exist_ok=True,parents=True)
    results=payload['conditions']
    expected={(method,k) for method in METHODS for k in (5,10,15,20)}
    if len(results)!=len(expected) or {(r['method'],r['k']) for r in results}!=expected:
        raise ValueError('Summary must account for all 28 method/cardinality conditions')
    completed=sum(r['status']=='completed' for r in results);draft=completed<len(results)
    write_json(REPORT/'results.json',payload)
    with (REPORT/'results.csv').open('w') as f:
        keys=['method','k','fit_per_class','status','n','expected','probability_vectors',*[m.name for m in QUALITY]]
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
        for r in results:writer.writerow({**{k:r[k] for k in keys if k in r},**{name:v['value'] for name,v in r.get('metrics',{}).items()}})
    lines=['# Banking77 v2 — extended report','',f'**{"Interim" if draft else "Evaluated"}: {completed}/{len(results)} quality conditions complete.** Missing conditions are not scored. This report is generated from saved evidence; it does not claim a v2.0.0 release.','',
        '## Design','',
        'One seed (42), nested label sets of 5/10/15/20 classes, and all 40 held-out test examples/class: 200/400/600/800 predictions per method. Class descriptions and source revision are frozen in [the protocol](protocol.md). The support pool contains 42 eligible labels. Training and test partitions remain separate.','',
        'OpenAI, Jev and Emissary routing are zero-shot. TF-IDF+LR and frozen MiniLM+LR use 50 fit examples/class with C=1; BERT uses 50/class and two CPU epochs. Emissary Qwen3-4B-Base SFT uses **100/class and one epoch**, so its comparisons with local supervised baselines do not hold supervision constant. Provider-internal use of the submitted rows is not independently audited. Quick Train remains unavailable and is not replaced by SFT.','',
        'The Qwen arm changes both model and mechanism relative to Emissary routing. The final model/settings inventory is in [the manifest](manifest.json). Checkpoint selection is the last checkpoint from the one-epoch job, without test-based selection.','',
        '## Quality and uncertainty','',
        'Accuracy intervals are 95% percentile bootstrap intervals, 2,000 resamples stratified by gold class, seed 20260925. Macro-F1 intervals and paired differences against OpenAI are saved in `results.json`. These condition on the fitted model, selected labels and one seed; they are pointwise, exploratory and not adjusted for multiple comparisons.','',
        '| Method | K | Coverage | Accuracy [95% CI] | Macro-F1 | ECE | Adaptive ECE | Log loss | Brier |',
        '| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for r in results:
        values=r.get('metrics',{});v=lambda name:values.get(name,{}).get('value')
        ci=r.get('uncertainty',{}).get('metrics',{}).get('accuracy')
        acc=f"{pct(v('accuracy'))} [{pct(ci['lower'])}, {pct(ci['upper'])}]" if ci else '—'
        lines.append(f"| {r['title']} | {r['k']} | {r['n']}/{r['expected']} ({r['status']}) | {acc} | {num(v('macro_f1'))} | {num(v('top_label_ece'))} | {num(v('adaptive_ece'))} | {num(v('multiclass_log_loss'))} | {num(v('multiclass_brier_score'))} |")
    lines+=['','## Cardinality observations','','| Method | Accuracy K=5 | Accuracy K=20 | Change (percentage points) |','| --- | ---: | ---: | ---: |']
    trends=[]
    for method,title in METHODS.items():
        a=next(r for r in results if r['method']==method and r['k']==5);b=next(r for r in results if r['method']==method and r['k']==20)
        av=a.get('metrics',{}).get('accuracy',{}).get('value');bv=b.get('metrics',{}).get('accuracy',{}).get('value')
        delta=100*(bv-av) if av is not None and bv is not None else None
        line=f'| {title} | {pct(av)} | {pct(bv)} | {num(delta)} |';lines.append(line);trends.append(line)
    lines+=['','The K=5→20 changes are descriptive. Increasing K also adds new labels and test examples, so the difference does not isolate a causal effect of cardinality. Overlapping cohorts must not be pooled as independent observations. A general claim that zero-shot necessarily degrades is not established by this single-seed experiment.','',
        '![Accuracy by cardinality](accuracy.png)','',
        '## Failures, recovery and probability validity','',
        'The first small run completed 21/24 executable conditions. Jev stopped at 381/400, 3/600 and 446/800 valid predictions for K=10/15/20 because a returned probability vector failed the sum-to-one tolerance of 0.0001. Its old invalid vectors were not persisted, so their exact deviations cannot be reconstructed. Recovery requested only the 970 missing IDs.','',
        'The follow-up preserves the strict probability tolerance. If only the sum check fails but the provider choice is a valid maximum-score label and the nonnegative finite scores have positive total mass, the label is retained for accuracy/F1. Confidence and probabilities are marked unavailable; raw scores and sums are saved in `probability_diagnostics.jsonl`. No renormalization is performed. Other invalid responses still fail. Full-cohort calibration metrics are suppressed if any distribution is unavailable. This explicitly changes the acceptance policy for label metrics in the recovery phase.','',
        'In the completed recovery, Jev has 1/2/3 unavailable probability vectors at K=10/15/20. All labels are present, so accuracy and macro-F1 cover the full test sets; calibration is unavailable for those three conditions.','',
        '## Operational measurements and costs','',
        'All inference is singleton, concurrency 1, without warmup or automatic retries. Historical and fresh measurements are not pooled. `results.json` lists separate phase sample counts and p50/p95 client latency with source paths. Individual runs retain timings, usage (including failed calls), preparation and cost reports. Provider hardware and hosted cache state are uncontrolled. Unknown Emissary charges are unavailable, not zero. No total billed cost is asserted. Reused Jev evidence may span more than one historical attempt; its phase latency is descriptive only.','',
        'Original BERT fits all completed, totaling approximately 14 minutes; the 20-class condition took about 337 seconds. Qwen training and deployment timing are separate from inference and local BERT timings.','',
        '## Limitations and reproducibility','',
        'This revised campaign is exploratory: test results were previously observed, one seed cannot estimate between-seed variability, and definitions were reviewed by the assistant rather than independently validated by a human. Initial reuse retained complete historical runs; subsequent Jev recovery retains validated partial outputs. Disclose this availability selection when discussing reliability.','',
        'All 28 agreed quality conditions completed. The follow-up used four existing Qwen jobs and 970 additional Jev requests; original predictions were retained. Raw logs and deployment/job IDs remain in the evidence directories. No additional inference is needed to regenerate the reports.','',
        'Rebuild the tables, prose and figure from the versioned summary with `PYTHONPATH=src venv/bin/python scripts/build_v2_reports.py`. To recompute metrics and bootstrap intervals from original predictions, add `--recompute`; that requires the ignored `artifacts/` evidence bundle. The [manifest](manifest.json) records source/model revisions and label sets; `results.json` retains intervals and prediction fingerprints. Original per-run manifests retain exact IDs and provider resources. Local campaign launchers are not required to render the published results. No release tag is created by report generation.']
    (REPORT/'report.md').write_text('\n'.join(lines)+'\n')
    brief=['# Banking77 v2 — brief','',f'**{"Interim results" if draft else "Evaluation results"}: {completed}/28 conditions complete.** We compare three zero-shot services and four supervised classifiers across 5, 10, 15 and 20 labels, using 40 held-out examples per label and one seed.','',
        '| Method | Accuracy K=5 | Accuracy K=20 | Change (pp) |','| --- | ---: | ---: | ---: |',*trends,'',
        'The table describes this selected label sequence; adding classes also changes the test population. It does not establish a universal or causal cardinality effect. Confidence intervals, macro-F1 and calibration metrics are in the [extended report](report.md).','',
        '**Supervision differs:** BERT, MiniLM+LR and TF-IDF+LR use 50 training examples/class. Emissary Qwen SFT uses 100/class, and also changes the model and mechanism relative to Emissary routing.','',
        'Jev’s initial K=10/15/20 runs were incomplete due to invalid probability sums. Recovery retains valid class choices with unavailable probabilities when only the sum is invalid; it never silently renormalizes. Missing conditions are not scored, and full-cohort calibration is unavailable when distributions are missing.','',
        'One seed, prior test exposure and historical output reuse limit generalization. The full report documents conditional bootstrap intervals, failures, phase-specific measurements and reproduction. Emissary costs remain unknown. This is not a v2.0.0 release announcement.']
    (REPORT/'brief.md').write_text('\n'.join(brief)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,5))
    for method,title in METHODS.items():
        points=[r for r in results if r['method']==method and r['status']=='completed']
        if points:ax.plot([r['k'] for r in points],[r['metrics']['accuracy']['value'] for r in points],marker='o',label=title)
    ax.set(xlabel='Number of classes',ylabel='Accuracy',xticks=[5,10,15,20],ylim=(0,1),title=f'Banking77 v2 — {completed}/28 complete conditions')
    ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(REPORT/'accuracy.png',dpi=180);plt.close(fig)
    print(f'Reports rebuilt offline: {completed}/28 complete; report.md, brief.md, results.csv/json and accuracy.png')

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--completion-root',type=Path,default=Path('artifacts/v2_completion'))
    parser.add_argument('--summary',type=Path,default=Path('reports/v2/results.json'))
    parser.add_argument('--output-dir',type=Path,default=REPORT)
    parser.add_argument('--recompute',action='store_true',help='Recompute metrics/CIs from the original ignored artifacts bundle')
    args=parser.parse_args()
    REPORT=args.output_dir
    render(collect(args.completion_root) if args.recompute else read_json(args.summary))
