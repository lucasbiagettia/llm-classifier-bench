import json
import pytest
from llm_classifier_bench.core import LabeledExample
from llm_classifier_bench.recovery import pending_examples


def test_only_missing_predictions_are_selected(tmp_path):
    examples = tuple(LabeledExample(str(i), f'text {i}', 'a') for i in range(3))
    path = tmp_path / 'predictions.jsonl'
    path.write_text(json.dumps({'sample_id':'1','gold_label':'a','input':'text 1'})+'\n')
    pending, completed = pending_examples(examples, [path])
    assert [e.sample_id for e in pending] == ['0','2']
    assert list(completed) == ['1']
    with pytest.raises(ValueError, match='repeated'):
        pending_examples(examples, [path,path])
    path.write_text(json.dumps({'sample_id':'1','gold_label':'a','input':'changed'})+'\n')
    with pytest.raises(ValueError, match='differs'):
        pending_examples(examples, [path])
