"""Matched fit/context pools and explicit, separately counted validation budgets."""
from dataclasses import asdict, dataclass
from hashlib import sha256
import json

from llm_classifier_bench.datasets.selection import LabeledSelectionConfig, select_labeled_examples


class UnsupportedConfiguration(ValueError):
    """A declared experimental cell cannot run without changing its protocol."""


@dataclass(frozen=True, slots=True)
class MatchedBudgetConfig:
    examples: int
    unit: str
    seed: int = 42
    validation_examples: int = 0

    def __post_init__(self):
        LabeledSelectionConfig(self.examples, self.unit, self.seed)
        LabeledSelectionConfig(self.validation_examples, "total", self.seed)


def select_matched_pools(fit, validation, classes, budget):
    """Select once by identity, independent of method and supplied input order."""
    selected, selection = select_labeled_examples(
        fit, classes, LabeledSelectionConfig(budget.examples, budget.unit, budget.seed),
        validation_examples=validation,
    )
    selected_validation, validation_selection = select_labeled_examples(
        validation, classes,
        LabeledSelectionConfig(budget.validation_examples, "total", budget.seed),
        validation_examples=selected,
    )
    identity = {"selection": selection, "validation_selection": validation_selection}
    return selected, selected_validation, {
        "comparison_regime": "matched_labeled_budget",
        "budget": asdict(budget), **identity,
        "pool_sha256": sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
        "validation_policy": "additional reserved labels; never borrowed from fit/context pool",
    }
