"""Hand-computable utility and normalization regressions for llm_router."""
import importlib.util
import itertools
from pathlib import Path

import pandas as pd
import pytest


PATH = Path(__file__).resolve().parents[1] / "research/problems/llm_router/evaluator.py"
spec = importlib.util.spec_from_file_location("llm_router_evaluator", PATH)
evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluator)
MODELS = {
    "cheap": "mistralai/mistral-7b-chat",
    "mid": "mistralai/mixtral-8x7b-chat",
    "expensive": "gpt-4-1106-preview",
}


def row(qualities, costs):
    result = {"prompt": "query", "eval_name": "synthetic"}
    for model, quality, cost in zip(MODELS.values(), qualities, costs):
        result[model] = quality
        result[f"{model}|total_cost"] = cost
    return result


def evaluate(tmp_path, rows, tiers):
    data = tmp_path / "data.csv"
    pd.DataFrame(rows).to_csv(data, index=False)
    solution = tmp_path / "solution.py"
    # Iterator is only a test fixture exercising the chosen action on each row.
    solution.write_text(
        "class Solution:\n"
        f"    def __init__(self): self.tiers = iter({tiers!r})\n"
        "    def solve(self, **kwargs): return next(self.tiers)\n"
    )
    instance = evaluator.Evaluator(str(tmp_path))
    instance.trace_files = [str(data)]
    return instance.evaluate(str(solution))


@pytest.mark.parametrize("quality", [0, 0.25, 0.5, 0.75, 1])
def test_same_choice_has_same_candidate_and_oracle_utility(tmp_path, quality):
    sample = row([quality] * 3, [0.001] * 3)
    result = evaluate(tmp_path, [sample], ["cheap"])
    assert result["accuracy"] == pytest.approx(quality)
    assert result["raw_score"] == pytest.approx(quality - 0.15)
    assert result["oracle_raw_score"] == pytest.approx(result["raw_score"])
    assert result["oracle_accuracy"] == pytest.approx(quality)
    assert result["oracle_avg_cost"] == pytest.approx(0.001)


@pytest.mark.parametrize(
    "qualities,costs,expected",
    [
        ([0.25, 0.75, 1], [0.0001, 0.001, 0.01], "mid"),
        ([0.75, 1, 1], [0, 0.002, 0.003], "cheap"),
        ([0, 0, 0], [0.003, 0.001, 0.002], "mid"),
        ([0.25, 0.5, 0.75], [0, 0, 0], "expensive"),
        ([0, 0, 0], [0, 0, 0], "expensive"),
    ],
)
def test_oracle_maximizes_published_utility(qualities, costs, expected):
    model, cost = evaluator.compute_oracle_stats(row(qualities, costs), MODELS)
    assert model == MODELS[expected]
    assert cost == costs[list(MODELS).index(expected)]


def test_perfect_policy_normalizes_to_100(tmp_path):
    # Independent, hand-computed maxima: mid .60, cheap .75, mid -.15, expensive .75.
    rows = [
        row([0.25, 0.75, 1], [0.0001, 0.001, 0.01]),
        row([0.75, 1, 1], [0, 0.002, 0.003]),
        row([0, 0, 0], [0.003, 0.001, 0.002]),
        row([0.25, 0.5, 0.75], [0, 0, 0]),
    ]
    result = evaluate(tmp_path, rows, ["mid", "cheap", "mid", "expensive"])
    assert result["raw_score"] == pytest.approx((0.60 + 0.75 - 0.15 + 0.75) / 4)
    assert result["raw_score"] == pytest.approx(result["oracle_raw_score"])
    assert result["score"] == pytest.approx(100)


@pytest.mark.parametrize("quality", [0, 0.25, 0.5, 0.75, 1])
def test_every_action_reward_is_bounded_by_oracle(tmp_path, quality):
    sample = row([quality, 0.5, 0.75], [0, 0.001, 0.003])
    expected = max(quality, 0.5 - 0.15, 0.75 - 0.45)
    for tier in MODELS:
        result = evaluate(tmp_path, [sample], [tier])
        assert result["oracle_raw_score"] == pytest.approx(expected)
        assert result["raw_score"] <= result["oracle_raw_score"] + 1e-12


@pytest.mark.parametrize("bad", [None, "bad", "truthy", "", float("nan"), float("inf"), -0.1, 1.1])
def test_malformed_quality_is_rejected_by_both_paths(tmp_path, bad):
    sample = row([bad, 0.5, 1], [0, 0, 0])
    with pytest.raises(ValueError, match="correctness"):
        evaluator.compute_oracle_stats(sample, MODELS)
    with pytest.raises(ValueError, match="correctness"):
        evaluate(tmp_path, [sample], ["cheap"])


@pytest.mark.parametrize("bad", [None, "bad", float("nan"), float("inf"), -0.001])
def test_malformed_cost_is_rejected_by_both_paths(tmp_path, bad):
    sample = row([0.25, 0.5, 1], [bad, 0, 0])
    with pytest.raises(ValueError, match="cost"):
        evaluator.compute_oracle_stats(sample, MODELS)
    with pytest.raises(ValueError, match="cost"):
        evaluate(tmp_path, [sample], ["cheap"])


def test_no_valid_models():
    assert evaluator.compute_oracle_stats({}, {}) == ("no_model_correct", 0.0)
    with pytest.raises(ValueError):
        evaluator.compute_oracle_stats({}, MODELS)


def test_order_and_tier_labels_do_not_change_model_utilities():
    sample = row([0.25, 0.75, 1], [0.0001, 0.001, 0.01])
    for permutation in itertools.permutations(MODELS.values()):
        tiers = dict(zip(["one", "two", "three"], permutation))
        assert evaluator.compute_oracle_stats(sample, tiers) == (MODELS["mid"], 0.001)
    # Exact utility ties also choose the same model regardless of insertion order.
    sample = row([0.5] * 3, [0.001] * 3)
    choices = {
        evaluator.compute_oracle_stats(sample, dict(zip(MODELS, p)))[0]
        for p in itertools.permutations(MODELS.values())
    }
    assert len(choices) == 1


def test_nonpositive_normalizer_and_empty_dataset(tmp_path):
    result = evaluate(tmp_path, [row([0, 0, 0], [0, 0, 0])], ["cheap"])
    assert result["raw_score"] == 0
    assert result["score"] == 0
    instance = evaluator.Evaluator(str(tmp_path))
    instance.trace_files = []
    solution = tmp_path / "solution.py"
    assert instance.evaluate(str(solution))["error"] == "Empty dataset"


def test_numeric_strings_and_fractional_tenths():
    model = MODELS["cheap"]
    assert evaluator._model_stats(
        {model: "0.3", f"{model}|total_cost": "0.001"}, model
    ) == pytest.approx((0.3, 0.001, 0.15))
