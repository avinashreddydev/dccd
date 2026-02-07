from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
from typing import Dict, List, Optional
from pydantic import BaseModel
from dataclasses import dataclass
from algorithms.constrained_decoding_system_prompt import GSM8K_SYSTEM_PROMPT, PII_EXTRACTION_SYSTEM_PROMPT, FINANCIAL_ENTITIES_SYSTEM_PROMPT, INSURANCE_CLAIMS_SYSTEM_PROMPT, DATA_TABLE_ANALYSIS_SYSTEM_PROMPT, GSM_SYMBOLIC_SYSTEM_PROMPT, PROVER9_SYSTEM_PROMPT, PROVER9_SYSTEM_PROMPT_WITH_FEWSHOTS, MATH500_SYSTEM_PROMPT
from algorithms.few_shot_structures import GSM8K_FEW_SHOTS, MATH500_FEW_SHOTS, GSM_SYMBOLIC_FEW_SHOTS, PROVER9_FEW_SHOTS
from algorithms.base_algorithm import BaseAlgorithm
import os

class ConstrainedFewShotDecodingParams(BaseModel):
    model_name: str
    max_tokens: int = 8192
    temperature: float = 0.8
    gpu_memory_utilization: float = 0.95
    dtype: str = "bfloat16"
    enforce_eager: bool = False
    save_dir: str = "outputs"
    save_name: str = "constrained_few_shot_decoding"



class ConstrainedFewShotDecoding(BaseAlgorithm):
    """
    Constrained decoding algorithm, This is a  structured decoding algorithm that uses a JSON schema to guide the decoding process.
    """
    FEW_SHOT_PROMPT = "Here are some examples of the structures of the answer: {few_shots}"
    def __init__(self, 
        model_name: str, 
        json_schema: Optional[Dict] = None, 
        grammar: Optional[str] = None,
        max_tokens: int = 8192, 
        temperature: float = 0.8, 
        gpu_memory_utilization: float = 0.95, 
        dtype: str = "bfloat16",
        device: str = "cuda",
        enforce_eager: bool = False,
        save_dir: str = "outputs",
        save_tag: str = "cdfs",
        task_name: str = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.alg_name = "constrained_few_shot_decoding"
        self.gpu_memory_utilization = gpu_memory_utilization
        self.dtype = dtype
        self.device = device
        self.enforce_eager = enforce_eager
        self.json_schema = json_schema
        self.grammar = grammar
        if json_schema:
            structured_outputs_params_json = StructuredOutputsParams(json=json_schema)
            self.sampling_params = SamplingParams(
                structured_outputs=structured_outputs_params_json,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        elif grammar:
            structured_outputs_params_grammar = StructuredOutputsParams(grammar=grammar)
            self.sampling_params = SamplingParams(
                structured_outputs=structured_outputs_params_grammar,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        else:
            self.sampling_params = SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
            )
        model_save_name = model_name.replace("/", "_")
        self.results_save_path = os.path.join(save_dir, model_save_name, save_tag)
        self.task_system_prompt = None
        self.chat_fewshots = []
        self.set_task_system_prompt(task_name)




    def load_model(self):
        self.llm = LLM(
            model=self.model_name, 
            gpu_memory_utilization=self.gpu_memory_utilization, 
            dtype=self.dtype, 
            enforce_eager=self.enforce_eager,
        )
        self.tokenizer = self.llm.get_tokenizer()




    # There are total 5 tasks:
    # 1. GSM8k 
    # 2. Financial Entities
    # 3. Insurance Claims
    # 4. Data Table Analysis
    # 5. PII Extraction
    def set_task_system_prompt(self, task : str, **kwargs) -> str:
        if task == "gsm8k":
            self.task_system_prompt = GSM8K_SYSTEM_PROMPT 
            self.task_system_prompt += self.FEW_SHOT_PROMPT.format(few_shots="\n".join(GSM8K_FEW_SHOTS))
        if task == "math500":
            self.task_system_prompt = MATH500_SYSTEM_PROMPT
            self.task_system_prompt += self.FEW_SHOT_PROMPT.format(few_shots="\n".join(MATH500_FEW_SHOTS))
        if task == "data_table_analysis":
            self.task_system_prompt = DATA_TABLE_ANALYSIS_SYSTEM_PROMPT
        if task == "insurance_claims":
            self.task_system_prompt = INSURANCE_CLAIMS_SYSTEM_PROMPT
        if task == "financial_entities":
            self.task_system_prompt = FINANCIAL_ENTITIES_SYSTEM_PROMPT
        if task == "pii_extraction":
            self.task_system_prompt = PII_EXTRACTION_SYSTEM_PROMPT
        if task == "gsm_symbolic":
            self.task_system_prompt = GSM_SYMBOLIC_SYSTEM_PROMPT
            self.task_system_prompt += self.FEW_SHOT_PROMPT.format(few_shots="\n".join(GSM_SYMBOLIC_FEW_SHOTS))
            self.task_system_prompt = self.task_system_prompt.replace("[[START]]", kwargs.get("start_symbol", "<<")).replace("[[END]]", kwargs.get("end_symbol", ">>"))
        if task == "prover9":
            self.task_system_prompt = PROVER9_SYSTEM_PROMPT
            self.task_system_prompt += self.FEW_SHOT_PROMPT.format(few_shots="\n".join(PROVER9_FEW_SHOTS))
        return self.task_system_prompt



    def create_batch_templated_prompts(self, texts: List[str]) -> List[str]:
        prompts = []
        for text in texts:

            if self.chat_fewshots:
                prompt = [
                    {"role" : "system", "content" : self.task_system_prompt}, 
                    *self.chat_fewshots,
                    {"role" : "user", "content" : text}
                ]
            else:
                prompt = [
                    {"role" : "system", "content" : self.task_system_prompt},
                    {"role" : "user", "content" : text}
                ]
            templated_prompt = self.tokenizer.apply_chat_template(prompt, tokenize = False, add_generation_prompt=True)
            prompts.append(templated_prompt)
        return prompts
        

    def generate(self, prompt: str) -> str:
        pass

    def generate_batch(self, texts : List[str]) -> List[str]:
        prompts = self.create_batch_templated_prompts(texts)
        llm_responses = self.llm.generate(prompts = prompts, sampling_params = self.sampling_params)
        json_string_responses = [o.outputs[0].text for o in llm_responses]

        return json_string_responses


  