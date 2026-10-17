import argparse
import json
import os
import sys
import traceback
import importlib.util
import pandas as pd
import gc
import math
from typing import Any, Mapping, Tuple


LAMBDA = 150.0


def _model_stats(query: Mapping[str, Any], model: str) -> Tuple[float, float, float]:
    """Return normalized quality, cost and utility using one scoring definition.

    RouterBench includes normalized ratings, not just binary correctness.
    Reject malformed data rather than turning it into free cost or full credit.
    """
    values = []
    for key, label in [(model, "correctness"), (f"{model}|total_cost", "cost")]:
        try:
            value = float(query.get(key))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Invalid {label} for {model}") from exc
        if not math.isfinite(value) or value < 0 or (label == "correctness" and value > 1):
            raise ValueError(f"Invalid {label} for {model}: {value}")
        values.append(value)
    quality, cost = values
    return quality, cost, quality - LAMBDA * cost


def compute_oracle_stats(query: Mapping[str, Any],
                         tier_to_model: Mapping[str, str]) -> Tuple[str, float]:
    """Maximize the same quality-minus-cost utility awarded to a candidate.

    Break utility ties by cost, then model name, independently of tier ordering.
    An empty candidate set retains the legacy sentinel; all-zero quality does
    not: the task requires choosing a model and paying its cost even then.
    """
    best_model = "no_model_correct"
    best_cost = 0.0
    best_key = None
    for model in tier_to_model.values():
        _, cost, utility = _model_stats(query, model)
        key = (-utility, cost, model)
        if best_key is None or key < best_key:
            best_key = key
            best_model, best_cost = model, cost
    return best_model, best_cost


class Evaluator:
    def __init__(self, problem_dir: str):
        self.problem_dir = problem_dir
        self.resources_dir = os.path.join(problem_dir, "resources")
        # Check mounted datasets directory first (from main repo datasets folder)
        mounted_datasets_dir = "/datasets/llm_router"
        if os.path.exists(mounted_datasets_dir) and os.listdir(mounted_datasets_dir):
            self.datasets_dir = mounted_datasets_dir
        else:
            # Fallback to resources/datasets if mounted directory doesn't exist
            self.datasets_dir = os.path.join(self.resources_dir, "datasets")
        ordered_names = ["routerbench_0shot_test.csv"]
        self.trace_files = [
            os.path.join(self.datasets_dir, name)
            for name in ordered_names
            if os.path.exists(os.path.join(self.datasets_dir, name))
        ]

    def evaluate(self, solution_module_path: str) -> dict:
        CANDIDATE_MODELS = ["cheap", "mid", "expensive"]

        # update based on dataset config
        TIER_TO_MODEL = {
            "cheap": "mistralai/mistral-7b-chat",
            "mid": "mistralai/mixtral-8x7b-chat",
            "expensive": "gpt-4-1106-preview",
        }
        
        # load solution and solver
        spec = importlib.util.spec_from_file_location("solution", solution_module_path)
        solution = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(solution)

        if not hasattr(solution, "Solution"):
            return {"score": 0.0, "runs_successfully": 0.0, "error": "Missing Solution"}

        solver = solution.Solution()
        
        total_queries = 0
        total_correct, oracle_correct = 0, 0
        total_cost, oracle_cost = 0.0, 0.0
        total_utility, oracle_utility = 0.0, 0.0
        for csv_path in self.trace_files:
            dataset_name = os.path.basename(csv_path)
            
            # load dataset
            df = pd.read_csv(csv_path, low_memory=False)
            
            for _, row in df.iterrows():
                
                # call solver
                chosen_tier = solver.solve(
                    query=row["prompt"],
                    eval_name=row["eval_name"],
                    candidate_models=CANDIDATE_MODELS,
                )
                if chosen_tier not in CANDIDATE_MODELS:
                    chosen_tier = "cheap"
                
                model_name = TIER_TO_MODEL[chosen_tier]
                correct, cost, utility = _model_stats(row, model_name)
                oracle, _ = compute_oracle_stats(row, TIER_TO_MODEL)
                oracle_correct_sample, oracle_cost_sample, oracle_utility_sample = (
                    _model_stats(row, oracle)
                )

                total_queries += 1
                total_correct += correct
                oracle_correct += oracle_correct_sample
                total_cost += cost
                oracle_cost += oracle_cost_sample
                total_utility += utility
                oracle_utility += oracle_utility_sample
            
            # Release memory
            del df
            gc.collect()

        # compute final score
        if total_queries == 0:
            return {"score": 0.0, "runs_successfully": 0.0, "error": "Empty dataset"}
        accuracy = total_correct / total_queries
        avg_cost = total_cost / total_queries
        raw_score = total_utility / total_queries
        oracle_accuracy = oracle_correct / total_queries
        oracle_avg_cost = oracle_cost / total_queries
        oracle_raw_score = oracle_utility / total_queries
        score_unbounded = (raw_score / oracle_raw_score) * 100 if oracle_raw_score > 0 else 0.0
        score = score_unbounded

        return {"runs_successfully": 1.0,
                "score_unbounded": score_unbounded,
                "total_queries": total_queries,
                "score": score, 
                "raw_score": raw_score,
                "accuracy": accuracy,
                "avg_cost": avg_cost,
                "oracle_raw_score": oracle_raw_score,
                "oracle_accuracy": oracle_accuracy,
                "oracle_avg_cost": oracle_avg_cost, 
                "lambda": LAMBDA,
                }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    try:
        problem_dir = os.path.dirname(os.path.abspath(__file__))
        result = Evaluator(problem_dir).evaluate(args.solution)
    except Exception as e:
        print(f"[evaluator] ERROR: {e}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        result = {"score": 0.0, "runs_successfully": 0.0, "error": str(e)}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f)
    # Format: "score score_unbounded" (space-separated)
    score = result.get('score', 0)
    score_unbounded = result.get('score_unbounded', score)
    print(f"{score} {score_unbounded}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())