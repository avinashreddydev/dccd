import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from data_loaders import (
    GSM8KLoader, Math500Loader, GSM_Symbolic_Loader, Prover9Loader
)
from algorithms import ConstrainedDecoding, ConstrainedFewShotDecoding, TwoStageDecoding, ConstrainedPromptDecoding, TwoStageDecodingScaled
from datetime import datetime
from yaml import load, Loader


def get_loader(task_name: str, task_params: dict):
    if task_name == "gsm8k":
        return GSM8KLoader(**task_params)
    elif task_name == "math500":
        return Math500Loader(**task_params)
    elif task_name == "gsm_symbolic":
        return GSM_Symbolic_Loader(**task_params)
    elif task_name == "prover9":
        return Prover9Loader(**task_params)
    else:
        raise ValueError(f"Task {task_name} not supported")


class Experiment:
    def __init__(self, algorithm_name: str, task_name: str, algorithm_params: dict, task_params: dict):
        self.algorithm_name = algorithm_name
        self.task_name = task_name
        self.algorithm_params = algorithm_params or {}
        self.task_params = task_params or {}

        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_tag = str(self.algorithm_params.get("model_name", "model")).replace("/", "_")
        self.experiment_name = f"{model_tag}_{self.algorithm_name}_{self.task_name}"
        if self.algorithm_name == "two_stage_decoding_scaled":
            self.experiment_path = os.path.join(
                "experiment_results", 
                model_tag,
                self.algorithm_name + "_n" + str(self.algorithm_params.get("stage1_num_samples", 1)),
                self.task_name,
            )
        else:
            self.experiment_path = os.path.join(
                "experiment_results", 
                model_tag,
                self.algorithm_name,
                self.task_name,
            )
        os.makedirs(self.experiment_path, exist_ok=True)

        self.loader = None
        self.algo = None

        print(f"\n--- Experiment ---")
        print(f"Algorithm: {self.algorithm_name}")
        print(f"Task:      {self.task_name}")
        print(f"Params:    {self.algorithm_params}")
        print(f"Save to:   {self.experiment_path}")

    def load(self):
        self.loader = get_loader(self.task_name, self.task_params)

        if self.algorithm_name == "constrained_decoding":
            # IMPORTANT: pass task so the system prompt is set
            if self.task_params.get("json_schema"):
                self.algo = ConstrainedDecoding(
                    json_schema=self.loader.get_json_schema(),
                    task_name = self.loader.task_name,
                    **self.algorithm_params,
                )
            elif self.task_params.get("grammar"):
                self.algo = ConstrainedDecoding(
                    grammar=self.loader.get_grammar_schema(),
                    task_name = self.loader.task_name,
                    **self.algorithm_params,
                )
            else:
                # raise `ValueError("Either json_schema or grammar must be provided")
                self.algo = ConstrainedDecoding(
                    task_name = self.loader.task_name,
                    **self.algorithm_params,
                )
            print(f"ConstrainedDecoding initialized with task: {self.loader.task_name}")
        elif self.algorithm_name == "constrained_few_shot_decoding":
            self.algo = ConstrainedFewShotDecoding(
                task_name = self.loader.task_name,
                **self.algorithm_params,
            )
            print(f"ConstrainedFewShotDecoding initialized with task: {self.loader.task_name}")
        elif self.algorithm_name == "constrained_prompt_decoding":
            self.algo = ConstrainedPromptDecoding(
                task_name = self.loader.task_name,
                **self.algorithm_params,
            )
            print(f"ConstrainedPromptDecoding initialized with task: {self.loader.task_name}")
        elif self.algorithm_name == "two_stage_decoding":
            if self.task_params.get("json_schema"):
                self.algo = TwoStageDecoding(
                    json_schema=self.loader.get_json_schema(),
                    task_name=self.loader.task_name,
                    **self.algorithm_params,
                )
            elif self.task_params.get("grammar"):
                self.algo = TwoStageDecoding(
                    grammar=self.loader.get_grammar_schema(),
                    task_name=self.loader.task_name,
                    **self.algorithm_params,
                )
            else:
                raise ValueError("Either json_schema or grammar must be provided")
           
            print(f"TwoStageDecoding initialized with task: {self.loader.task_name}")
        elif self.algorithm_name == "two_stage_decoding_scaled":
            if self.task_params.get("json_schema"):
                self.algo = TwoStageDecodingScaled(
                    json_schema=self.loader.get_json_schema(),
                    task_name=self.loader.task_name,
                    **self.algorithm_params,
                )
            else:
                raise ValueError("json_schema must be provided")
            print(f"TwoStageDecodingScaled initialized with task: {self.loader.task_name}")
        else:
            raise ValueError(f"Algorithm {self.algorithm_name} not supported")

    def execute(self):
        self.algo.load_model()

        # If your ConstrainedDecoding already loads model in __init__,
        # do NOT call load_model() again.
        # If you *do* have load_model(), keep only one pattern.

        if self.algorithm_name == "two_stage_decoding":
            responses, stage1_outputs = self.algo.generate_batch(self.loader.data)
            stage1_results, stage1_summary_results = self.loader.evaluate_batch(stage1_outputs)
            stage2_results, stage2_summary_results = self.loader.evaluate_batch(responses)

            self.loader.save_all_results(self.experiment_path, stage1_results=stage1_results, stage1_summary_results=stage1_summary_results, final_results=stage2_results, final_summary_results=stage2_summary_results)
        elif self.algorithm_name == "two_stage_decoding_scaled":
            responses, stage1_outputs = self.algo.generate_batch(self.loader.data)
            # stage1_results, stage1_summary_results = self.loader.evaluate_batch(stage1_outputs)
            stage2_results, stage2_summary_results = self.loader.evaluate_batch(responses)

            self.loader.save_all_results(self.experiment_path, stage1_outputs=stage1_outputs, final_results=stage2_results, final_summary_results=stage2_summary_results)
        elif self.algorithm_name == "constrained_decoding":
            responses = self.algo.generate_batch(self.loader.data)
            results, summary_results = self.loader.evaluate_batch(responses)
            self.loader.save_all_results(self.experiment_path, final_results=results, final_summary_results=summary_results)

        elif self.algorithm_name == "constrained_few_shot_decoding":
            responses = self.algo.generate_batch(self.loader.data)
            results, summary_results = self.loader.evaluate_batch(responses)
            self.loader.save_all_results(self.experiment_path, final_results=results, final_summary_results=summary_results)

        elif self.algorithm_name == "constrained_prompt_decoding":
            responses = self.algo.generate_batch(self.loader.data)
            results, summary_results = self.loader.evaluate_batch(responses)
            self.loader.save_all_results(self.experiment_path, final_results=results, final_summary_results=summary_results)


        # Save using your loader API; pass the experiment path/name as needed
        # (adjust these to your loader's actual signatures)

    def close(self):
        if self.algo is not None:
            close_fn = getattr(self.algo, "close", None)
            if callable(close_fn):
                close_fn()
        self.algo = None
        self.loader = None


def create_experiments(config: dict):
    experiments = []
    for algo in config["algorithms"]:
        algo_name = algo["name"]
        variants = algo.get("variants", [])
        if not variants:
            raise ValueError(f"Algorithm '{algo_name}' has no variants")

        for variant_params in variants:
            for task in config["tasks"]:
                task_name = task["name"]
                task_params = task.get("params", {})
                experiments.append(
                    Experiment(algo_name, task_name, variant_params, task_params)
                )
    return experiments


if __name__ == "__main__":
    config = load(open("configs/config.yaml", "r"), Loader=Loader)
    experiments = create_experiments(config)

    for exp in experiments:
        exp.load()
        exp.execute()
        exp.close()
        print(f"Experiment {exp.experiment_name} completed")
        print(f"Results saved to: {exp.experiment_path}")
        print(f"Summary results saved to: {exp.experiment_path}/summary_results.json")
        print(f"Results saved to: {exp.experiment_path}/results.json")
        print(f"Summary results saved to: {exp.experiment_path}/summary_results.json")

        del exp