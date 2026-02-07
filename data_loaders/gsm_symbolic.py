from data_loaders.utils import get_gsm_symbolic_grammar, validate_expression_equivalence
from pathlib import Path
from data_loaders.base_loader import BaseLoader
from datasets import load_dataset
from typing import List
import os
from datetime import datetime
import json
# __file__ is the current script's path
# .parent gets the directory containing the script (data_loaders/)
# .parent.parent gets the project root (project_dir/)
PROJECT_DIR = Path(__file__).parent.parent

class GSM_Symbolic_Loader(BaseLoader):
    def __init__(self, **kwargs):
        super().__init__()
        self.task_name = "gsm_symbolic"
        self.data_folder_path = PROJECT_DIR / "data_loaders" / "utils" / 'gsm_symbolic_data' 
        self.load_data()
        self.results = None
        dataset_name = "gsm_symbolic"

    
    def load_data(self):
        files = os.listdir(self.data_folder_path)
        data = []
        for file in files:
            with open(os.path.join(self.data_folder_path, file)) as f:
                d = json.load(f)
            data.append(d)
        self.data = [ex["question_parsed"] for ex in data]
        self.ground_truths = [ex["answer_parsed"] for ex in data]
        self.variable_types = [ex["variable_types"] for ex in data]

    def get_json_schema(self):
        pass


    def get_grammar_schema(self):
        return get_gsm_symbolic_grammar()


    
    def evaluate_batch(self, responses : List[str]): 

        assert len(self.data) == len(self.ground_truths) == len(responses), f"The number of samples in the data {len(self.data)}, ground truth {len(self.ground_truths)}, and responses {len(responses)} must be the same"

        results = []
        for q, g, p, vt in zip(self.data, self.ground_truths, responses, self.variable_types):

            p = p.replace("<<", "").replace(">>", "")

            result = validate_expression_equivalence(g, p, vt)
        
            results.append({
                "question": q,
                "gold": g,
                "pred": p,
                "variable_type" : vt, 
                "correct": result,
            })


        self.results = results
        total_samples = len(results)
        # Summary Results
        accuracy = sum(r["correct"] for r in results) / total_samples

        self.summary_results = {
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

    