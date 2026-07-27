from __future__ import annotations

import os
from uuid import uuid4

import pytest
from dotenv import load_dotenv

from llm_classifier_bench.classifiers.base import ClassificationInput, ClassDefinition
from llm_classifier_bench.classifiers.emissary import EmissaryClassifier, EmissaryClient


pytestmark = pytest.mark.integration


def _api_key_or_skip() -> str:
    load_dotenv(override=False)
    api_key = os.getenv("EMISSARY_API_KEY")
    if not api_key:
        pytest.skip("EMISSARY_API_KEY is not configured")
    return api_key


def test_emissary_real_api_returns_normalized_multiclass_probabilities() -> None:
    client = EmissaryClient(api_key=_api_key_or_skip())

    existing_model_id = os.getenv("EMISSARY_TEST_MODEL_ID")
    if existing_model_id:
        classifier = EmissaryClassifier(
            client=client,
            model_id=existing_model_id,
            classifier_name="emissary-integration-test",
        )
    else:
        classifier = EmissaryClassifier.create(
            client=client,
            experiment_name=f"classification-benchmark-test-{uuid4().hex[:10]}",
            classes=[
                ClassDefinition(
                    name="sports",
                    description="Sports, matches, competitions, teams, and athletes.",
                ),
                ClassDefinition(
                    name="finance",
                    description="Money, banking, markets, investments, and payments.",
                ),
                ClassDefinition(
                    name="cooking",
                    description="Recipes, ingredients, food preparation, and kitchens.",
                ),
            ],
        )

    prediction = classifier.predict(
        [
            ClassificationInput(
                sample_id="integration-1",
                text="The team won the championship after scoring in extra time.",
            )
        ]
    )[0]

    assert prediction.predicted_label in {"sports", "finance", "cooking"}
    assert prediction.confidence is not None
    assert 0.0 <= prediction.confidence <= 1.0
    assert prediction.probabilities is not None
    assert set(prediction.probabilities) == {"sports", "finance", "cooking"}
    assert sum(prediction.probabilities.values()) == pytest.approx(1.0, abs=1e-4)
    assert prediction.latency_ms > 0.0
    assert prediction.raw_response is not None
