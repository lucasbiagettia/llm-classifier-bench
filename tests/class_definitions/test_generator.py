from __future__ import annotations

import json
from types import SimpleNamespace

from llm_classifier_bench.class_definitions.generator import (
    OpenAIClassDefinitionGenerator,
    build_minimal_profile,
)


class StubCompletions:
    def __init__(self) -> None:
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        content = json.dumps(
            {
                "classes": [
                    {
                        "canonical_name": "cash_withdrawal_charge",
                        "description": "A customer query about a fee charged for withdrawing cash.",
                    },
                    {
                        "canonical_name": "cash_withdrawal_not_recognised",
                        "description": "A customer query about a cash withdrawal they do not recognize.",
                    },
                ]
            }
        )
        return SimpleNamespace(
            id="resp-generator-1",
            model="generator-snapshot",
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        )


class StubClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=StubCompletions())


def test_minimal_profile_is_deterministic_and_preserves_canonical_names() -> None:
    profile = build_minimal_profile(
        dataset_name="banking77",
        canonical_names=("cash_withdrawal_charge", "card_arrival"),
    )

    assert profile.class_names == ("cash_withdrawal_charge", "card_arrival")
    assert profile.classes[0].description == "Category: cash withdrawal charge."
    assert profile.generation["method"] == "deterministic"


def test_openai_generator_uses_only_label_inventory_and_strict_structured_output() -> None:
    client = StubClient()
    generator = OpenAIClassDefinitionGenerator(
        model="cheap-generator-model",
        reasoning_effort="minimal",
        client=client,
    )

    profile = generator.generate(
        dataset_name="banking77",
        dataset_context="Banking customer-support intent classification.",
        canonical_names=(
            "cash_withdrawal_charge",
            "cash_withdrawal_not_recognised",
        ),
    )

    assert profile.class_names == (
        "cash_withdrawal_charge",
        "cash_withdrawal_not_recognised",
    )
    assert profile.generation["requested_model"] == "cheap-generator-model"
    assert profile.generation["resolved_model"] == "generator-snapshot"
    assert profile.generation["uses_train_examples"] is False
    assert profile.generation["uses_validation_examples"] is False
    assert profile.generation["uses_test_examples"] is False

    kwargs = client.chat.completions.kwargs
    assert kwargs["reasoning_effort"] == "minimal"
    assert kwargs["response_format"]["type"] == "json_schema"
    assert kwargs["response_format"]["json_schema"]["strict"] is True

    prompt_text = "\n".join(message["content"] for message in kwargs["messages"])
    assert "cash_withdrawal_charge" in prompt_text
    assert "cash_withdrawal_not_recognised" in prompt_text
    assert "train examples" not in prompt_text.lower()
