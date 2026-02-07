from data_loaders.utils import get_prover9_grammar, FOL_Prover9_Program
from pathlib import Path
from data_loaders.base_loader import BaseLoader
from datasets import load_dataset
from typing import List
import os
from datetime import datetime
import json
import random
# __file__ is the current script's path
# .parent gets the directory containing the script (data_loaders/)
# .parent.parent gets the project root (project_dir/)
PROJECT_DIR = Path(__file__).parent.parent

class Prover9Loader(BaseLoader):
    def __init__(self, **kwargs):
        super().__init__()
        self.task_name = "prover9"
        self.num_samples = kwargs.get("num_samples", -1)
        self.load_data()
        self.results = None
        dataset_name = "prover9"
    
    def load_data(self):
        dataset_path = str(PROJECT_DIR / "datasets" / "folio/" )
        print(f"Loading dataset from {dataset_path}")
        dataset = load_dataset(dataset_path, split="validation")
        if self.num_samples > 0:
            dataset = dataset.select(range(self.num_samples))

        data = []
        ground_truths = []
        for ex in dataset:
            context = ex["premises"]
            question = f"Based on the above information, is the following statement true, false, or uncertain? {ex['conclusion']}"
            prompt = 'Problem:\n[[PROBLEM]]\nQuestion:\n[[QUESTION]]\n###'
            prompt = prompt.replace('[[PROBLEM]]', context).replace('[[QUESTION]]', question.strip())
            data.append(prompt)
            ground_truths.append(ex['label'])
        
        self.data = data
        self.ground_truths = ground_truths
    

    def get_json_schema(self):
        pass


    def get_grammar_schema(self):
        return get_prover9_grammar()


    
    def evaluate_batch(self, responses : List[str]): 

        assert len(self.data) == len(self.ground_truths) == len(responses), f"The number of samples in the data {len(self.data)}, ground truth {len(self.ground_truths)}, and responses {len(responses)} must be the same"

        results = []
        for q, g, p in zip(self.data, self.ground_truths, responses):

            program = FOL_Prover9_Program(p)
            answer, error_message = program.execute_program()
            if answer is None:
                parsed = False
            else:
                parsed = True

            if answer is None:
                correct = False
            elif answer == 'True':
                answer = "True"
                correct = answer == g
            elif answer == 'False':
                answer = "False"
                correct = answer == g
            elif answer == 'Unknown':
                answer = "Uncertain"
                correct = answer == g
            else:
                correct = False
                error_message = "Answer not recognized"
    

            results.append({
                "question": q,
                "gold": g,
                "raw_pred": p,
                "pred": answer,
                "correct": correct,
                "parsed": parsed,
                "error_message": error_message,
            })
        
        self.results = results
        total_samples = len(results)
        # Summary Results
        accuracy = sum(r["correct"] for r in results) / total_samples
        parsed_accuracy = sum(r["parsed"] for r in results) / total_samples
        self.summary_results = {
            "accuracy": accuracy,
            "parsed_accuracy": parsed_accuracy,
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

    