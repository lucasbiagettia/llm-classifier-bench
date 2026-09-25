"""Offline preparation accounting evidence using real TF-IDF and known arithmetic."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess

from probe_tfidf_classifier import fixture_bundle, StaticDataset
from report_preparation import render_report
from llm_classifier_bench.classifiers import TfidfLogisticClassifier
from llm_classifier_bench.costs import recalculate_costs
from llm_classifier_bench.measurement import MeasurementConfig
from llm_classifier_bench.preparation import recalculate_preparation, amortize_preparation
from llm_classifier_bench.runner import BenchmarkRunConfig, run_benchmark


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,default=Path('artifacts/preparation_checks'))
    args=parser.parse_args()
    pricing=Path(__file__).resolve().parents[1]/'pricing/self_hosted_reference.json'
    revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True))
    classifier=TfidfLogisticClassifier()
    result=run_benchmark(StaticDataset(fixture_bundle()),classifier,BenchmarkRunConfig(
        output_root=args.output_root,run_id='local-tfidf',validation_fraction=.25,pricing_path=pricing,
        measurement=MeasurementConfig(warmup_examples=1,client_location='local fixture; no network',
                                      cache_condition='fresh TF-IDF features; one inference warmup'),
        metadata={'code_revision':revision,'working_tree_dirty':dirty,'smoke':True,
                  'limitation':'Small fixture and historical illustrative CPU rate; not production performance or an invoice'}))
    prep=recalculate_preparation(result.run_dir)
    assert prep==json.loads((result.run_dir/'preparation_report.json').read_text())
    roots=[e for e in prep['events'] if e['parent_id'] is None]
    expected_ms=sum((e['end_ns']-e['start_ns'])/1e6 for e in roots)
    hourly=float(prep['pricing']['local_hardware']['usd_per_hour'])
    assert math.isclose(prep['total_cost_usd'],expected_ms/3600000*hourly,rel_tol=1e-12)
    assert math.isclose(sum(prep['exclusive_ms_by_category'].values()),expected_ms,rel_tol=1e-12)
    assert prep['selected_configuration']['selected_configuration']['c']==classifier.selected_c
    assert prep['selected_configuration']['final_refit_performed'] is False
    assert len([e for e in prep['events'] if e['category']=='fit'])==3
    reports={}
    for scenario in ('continuous','deployed'):
        inference=recalculate_costs(result.run_dir,deployment_hours=24 if scenario=='deployed' else None)
        combined={'preparation':prep,'inference':inference,
                  'amortization':amortize_preparation(prep,inference,scenario=scenario)}
        for row in combined['amortization']['rows']:
            assert math.isclose(row['amortized_cost_per_valid_prediction_usd'],
                                (prep['total_cost_usd']+row['inference_scenario_cost_usd'])/row['successful_examples'],rel_tol=1e-12)
        (args.output_root/f'{scenario}.json').write_text(json.dumps(combined,indent=2)+'\n')
        (args.output_root/f'{scenario}.md').write_text(render_report(combined))
        reports[scenario]=combined['amortization']['rows']
    (args.output_root/'validation.json').write_text(json.dumps({
        'code_revision':revision,'working_tree_dirty':dirty,'checks_passed':True,
        'offline_only':True,'paid_requests':0,'deployed_resources':0,
        'prepared_classifier':'real TF-IDF + logistic regression',
        'preparation_cost_usd':prep['total_cost_usd'],'selected_c':classifier.selected_c,
        'final_refit_performed':False,'scenarios':reports,
        'limitation':'CPU fixture and illustrative historical rate; no live provider/GPU cost validation',
    },indent=2)+'\n')
    print(args.output_root/'validation.json')


if __name__=='__main__':
    main()
