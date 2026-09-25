import pytest

from llm_classifier_bench.metrics.base import EvaluationRecord
from llm_classifier_bench.metrics.uncertainty import stratified_quality_bootstrap


def rows(predictions):
    return tuple(EvaluationRecord(sample_id=str(i), gold_label="ab"[i % 2], predicted_label=p, confidence=None, probabilities=None, latency_ms=None)
                 for i, p in enumerate(predictions))


def test_perfect_scores_and_identical_pair_have_degenerate_intervals():
    records = rows("abababab")
    result = stratified_quality_bootstrap(records, repetitions=100)
    assert all(m == {"estimate": 1., "lower": 1., "upper": 1.}
               for m in result["metrics"].values())
    paired = stratified_quality_bootstrap(records, paired_records=records[::-1], repetitions=100)
    assert all(m == {"estimate": 0., "lower": 0., "upper": 0.}
               for m in paired["metrics"].values())


def test_paired_effect_uses_shared_resamples_and_is_reproducible():
    first, second = rows("abababab"), rows("aaaaaaaa")
    result = stratified_quality_bootstrap(first, paired_records=second, repetitions=200)
    assert result == stratified_quality_bootstrap(first, paired_records=second, repetitions=200)
    assert result["metrics"]["accuracy"] == {"estimate": .5, "lower": .5, "upper": .5}
    assert result["metrics"]["macro_f1"]["estimate"] == pytest.approx(2 / 3)


def test_reject_mismatched_and_duplicate_ids():
    records = rows("abab")
    with pytest.raises(ValueError):
        stratified_quality_bootstrap(records, paired_records=records[:-1], repetitions=100)
    with pytest.raises(ValueError):
        stratified_quality_bootstrap(records + records, repetitions=100)


def test_reject_mismatched_gold():
    with pytest.raises(ValueError, match="gold"):
        stratified_quality_bootstrap(rows("ab"), paired_records=(
            EvaluationRecord(sample_id="0", gold_label="b", predicted_label="a", confidence=None, probabilities=None, latency_ms=None),
            EvaluationRecord(sample_id="1", gold_label="b", predicted_label="b", confidence=None, probabilities=None, latency_ms=None)), repetitions=100)
