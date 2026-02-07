from data_loaders.schemas import GSM8KSchema
from pathlib import Path
from data_loaders.base_loader import BaseLoader
from datasets import load_dataset
from math_utils.utils import DATASET_KEYS, RESPONSE_EXTRACTOR, RESPONSE_COMPARATOR
from typing import List
import os
from datetime import datetime
import json
# __file__ is the current script's path
# .parent gets the directory containing the script (data_loaders/)
# .parent.parent gets the project root (project_dir/)
PROJECT_DIR = Path(__file__).parent.parent

class GSM8KLoader(BaseLoader):
    def __init__(self, csv_path=None, **kwargs):
        super().__init__()
        self.input_column = "table"
        self.output_column = "ground_truth"
        self.task_name = "gsm8k"
        self.load_data()
        self.results = None
        dataset_name = "openai/gsm8k"

        self.re = RESPONSE_EXTRACTOR[dataset_name]
        self.eq = RESPONSE_COMPARATOR[dataset_name]
    

    def load_data(self):
        ds = load_dataset("datasets/gsm8k", "main")
        data = ds["test"]
        self.data = [ex["question"] for ex in data]
        self.ground_truths = [ex["answer"] for ex in data]

    def get_json_schema(self):
        return GSM8KSchema.model_json_schema()


    
    def evaluate_batch(self, responses : List[str]): 

        assert len(self.data) == len(self.ground_truths) == len(responses), f"The number of samples in the data {len(self.data)}, ground truth {len(self.ground_truths)}, and responses {len(responses)} must be the same"

        results = []
        for q, g, p in zip(self.data, self.ground_truths, responses):
            # extracting the response and then comparing the predicted and ground truth responses
            gold_ans = self.re(g)
            pred_ans = self.re(p)
            acc = 1 if self.eq(gold_ans, pred_ans) else 0
            parsed = self.is_json(p)
            composite_accuracy = acc and parsed 
            results.append({
                "question": q,
                "gold": g,
                "pred": p,
                "gold_extracted": gold_ans,
                "pred_extracted": pred_ans,
                "correct": acc,
                "parsed": parsed,
                "composite_accuracy": composite_accuracy,
            })
        self.results = results
        total_samples = len(results)
        # Summary Results
        composite_accuracy = sum(r["composite_accuracy"] for r in results) / total_samples
        parsed_accuracy = sum(r["parsed"] for r in results) / total_samples
        accuracy = sum(r["correct"] for r in results) / total_samples

        self.summary_results = {
            "composite_accuracy": composite_accuracy,
            "parsed_accuracy": parsed_accuracy,
            "accuracy": accuracy,
            "total_samples": total_samples,
        }
        return results, self.summary_results


    def get_summary_results(self):
        return self.summary_results
    
    def save_all_results(self, out_path: str, **kwargs):
        print(f"Saving results to: {out_path}")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        for key, value in kwargs.items():
            path = os.path.join(out_path, f"{key}.json")
            with open(path, "w") as f:
                json.dump(value, f, indent=4, ensure_ascii=False)
            print(f"✅ Saved {key} to: {path}")

        results_path = os.path.join(out_path, "results.json")

        with open(results_path, "w") as f:
            json.dump(self.results, f, indent=4, ensure_ascii=False)
        print(f"✅ Saved results to: {results_path}")

        summary_path = os.path.join(out_path, "summary_results.json")
        with open(summary_path, "w") as f:
            json.dump(self.summary_results, f, indent=4, ensure_ascii=False)
        print(f"✅ Saved summary results to: {summary_path}")

    