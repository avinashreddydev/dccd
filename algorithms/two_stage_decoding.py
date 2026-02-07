from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
from typing import Dict, List, Optional
from pydantic import BaseModel
from algorithms.base_algorithm import BaseAlgorithm
import os
from pathlib import Path
from yaml import load, Loader
from algorithms.constrained_decoding_system_prompt import PROVER9_SYSTEM_PROMPT, PROVER9_SYSTEM_PROMPT_WITH_FEWSHOTS, PROVER9_PROMPT
PROJECT_DIR = Path(__file__).parent.parent
GSM_SYMBOLIC_PROMPT_PATH = PROJECT_DIR / "algorithms" / "prompt_templates" / "gsm_symbolic.yaml"
GSM_SYMBOLIC_PROMPT = load(open(GSM_SYMBOLIC_PROMPT_PATH, "r"), Loader=Loader)


class TwoStageDecodingParams(BaseModel):
    model_name: str
    structurer_model_name: Optional[str] = None  # If None, uses same model for both stages
    max_tokens: int = 8192
    stage1_max_tokens: int = 8192
    temperature: float = 0.0
    gpu_memory_utilization: float = 0.45
    dtype: str = "bfloat16"
    enforce_eager: bool = True
    save_dir: str = "outputs"
    save_name: str = "two_stage_decoding"


# =============================================
# Stage 1 System Prompts (Free-form generation)
# =============================================

GSM8K_STAGE1_SYSTEM_PROMPT = (
    "You are a meticulous math tutor. Provide a complete, detailed solution in plain text. "
    "Show clear step-by-step reasoning and end by stating the final numeric answer. Do not use JSON."
)

MATH500_STAGE1_SYSTEM_PROMPT = (
    "You are a meticulous math tutor. Provide a complete, detailed solution in plain text. "
    "Show clear step-by-step reasoning and end by stating the final numeric answer. Do not use JSON."
)

MATH500_STAGE2_SYSTEM_PROMPT = (
    "You are a precise extractor. Convert the prior free-form solution into a JSON object with the schema: "
    "{steps: string[], answer: string}. Use only information present in the prior solution. "
    "No extra text, no code fences—emit ONLY the JSON."
)
DATA_TABLE_ANALYSIS_STAGE1_SYSTEM_PROMPT = (
    "You are a data analyst. Analyze the given CSV-like table thoroughly. "
    "Describe the table structure, count the rows (excluding header), identify column types, "
    "and note any special patterns or null values. Provide your analysis in plain text."
)

FINANCIAL_ENTITIES_STAGE1_SYSTEM_PROMPT = (
    "You are a financial analyst. Read the following financial news text carefully. "
    "Identify and describe all entities you find: companies, dates, locations, monetary values, "
    "people, products, and quantities. Provide your analysis in plain text, listing each entity you find."
)

INSURANCE_CLAIMS_STAGE1_SYSTEM_PROMPT = (
    "You are an expert insurance claim processor. Read the claim description carefully. "
    "Identify all relevant information: dates, policy details, claim amounts, parties involved, "
    "incident descriptions, and any other pertinent details. Provide your analysis in plain text."
)

PII_EXTRACTION_STAGE1_SYSTEM_PROMPT = (
    "You are a PII extraction specialist. Read the text carefully and identify all personally "
    "identifiable information (PII) present. List each piece of PII you find, categorizing them "
    "appropriately. Provide your analysis in plain text."
)


GSM_SYMBOLIC_STAGE1_SYSTEM_PROMPT = GSM_SYMBOLIC_PROMPT["cot_instruct"]["text"]
GSM_SYMBOLIC_STAGE1_SYSTEM_PROMPT = GSM_SYMBOLIC_STAGE1_SYSTEM_PROMPT.replace("[[START]]", "<<").replace("[[END]]", ">>")


GSM_SYMBOLIC_STAGE2_SYSTEM_PROMPT = GSM_SYMBOLIC_PROMPT["std_instruct"]["gsm"]
GSM_SYMBOLIC_STAGE2_SYSTEM_PROMPT = GSM_SYMBOLIC_STAGE2_SYSTEM_PROMPT.replace("[[START]]", "<<").replace("[[END]]", ">>")


PROVER9_STAGE1_SYSTEM_PROMPT = PROVER9_SYSTEM_PROMPT
PROVER9_STAGE1_SYSTEM_PROMPT_WITH_FEWSHOTS = PROVER9_SYSTEM_PROMPT_WITH_FEWSHOTS

# PROVER9_STAGE2_SYSTEM_PROMPT = PROVER9_PROMPT["parser_prompt"]["prover9"]
PROVER9_STAGE2_SYSTEM_PROMPT = PROVER9_SYSTEM_PROMPT 
# =============================================
# Stage 2 System Prompts (Extraction to JSON)
# =============================================

GSM8K_STAGE2_SYSTEM_PROMPT = (
    "You are a precise extractor. Convert the prior free-form solution into a JSON object with the schema: "
    "{steps: string[], answer: string}. Use only information present in the prior solution. "
    "No extra text, no code fences—emit ONLY the JSON."
)

DATA_TABLE_ANALYSIS_STAGE2_SYSTEM_PROMPT = (
    "You are a precise extractor. Convert the prior analysis into a structured JSON object "
    "following the provided response format. Use only information present in the prior analysis. "
    "Do not guess: if a value does not exist or is not applicable, return null. "
    "No extra text, no code fences—emit ONLY the JSON."
)

FINANCIAL_ENTITIES_STAGE2_SYSTEM_PROMPT = (
    "You are a precise extractor. Convert the prior entity analysis into a structured JSON object "
    "with categories: Company, Date, Location, Money, Person, Product, Quantity. "
    "Each category should be a list of strings or null if none found. "
    "Use only entities mentioned in the prior analysis. "
    "No extra text, no code fences—emit ONLY the JSON."
)

INSURANCE_CLAIMS_STAGE2_SYSTEM_PROMPT = (
    "You are a precise extractor. Convert the prior claim analysis into a structured JSON object "
    "following the provided response format. For dates, use YYYY-MM-DD format. "
    "Use only information present in the prior analysis. If information is missing, use null. "
    "No extra text, no code fences—emit ONLY the JSON."
)

PII_EXTRACTION_STAGE2_SYSTEM_PROMPT = (
    "You are a precise extractor. Convert the prior PII analysis into a structured JSON object "
    "following the provided response format. Use only information present in the prior analysis. "
    "If a PII entity is not found, set that field to null. "
    "No extra text, no code fences—emit ONLY the JSON."
)


class TwoStageDecoding(BaseAlgorithm):
    """
    Two-stage decoding algorithm:
    Stage 1: Generate free-form solution (unconstrained)
    Stage 2: Extract into structured JSON using schema
    """
    
    def __init__(
        self,
        model_name: str,
        json_schema: Optional[Dict] = None,
        grammar: Optional[str] = None,
        structurer_model_name: Optional[str] = None,
        max_tokens: int = 8192,
        stage1_max_tokens: int = 8192,
        temperature: float = 0.8,
        gpu_memory_utilization: float = 0.95,
        dtype: str = "bfloat16",
        device: str = "cuda",
        enforce_eager: bool = False,
        save_dir: str = "outputs",
        save_tag: str = "tsd",
        task_name: str = None,
        question_in_stage2: bool = False,
    ):
        super().__init__()
        self.model_name = model_name
        self.structurer_model_name = structurer_model_name if structurer_model_name else model_name
        self.gpu_memory_utilization = gpu_memory_utilization
        self.dtype = dtype
        self.device = device
        self.enforce_eager = enforce_eager
        self.json_schema = json_schema
        self.grammar = grammar
        # Stage 1: Free-form sampling params (unconstrained)
        self.stage1_sampling_params = SamplingParams(
            max_tokens=stage1_max_tokens,
            temperature=temperature,
            stop=["\n------"]
        )
        self.alg_name = "two_stage_decoding"

        self.chat_fewshots_stage1 = []
        self.chat_fewshots_stage2 = []
        self.question_in_stage2 = question_in_stage2


        # Stage 2: Structured sampling params (JSON schema guided)
        if json_schema:
            structured_outputs_params_json = StructuredOutputsParams(json=json_schema)
            self.stage2_sampling_params = SamplingParams(
                structured_outputs=structured_outputs_params_json,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        elif grammar:
            structured_outputs_params_grammar = StructuredOutputsParams(grammar=grammar)
            self.stage2_sampling_params = SamplingParams(
                structured_outputs=structured_outputs_params_grammar,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        else:
            raise ValueError("Either json_schema or grammar must be provided")
        
        model_save_name = model_name.replace("/", "_")
        self.results_save_path = os.path.join(save_dir, model_save_name, save_tag)
        
        # Task-specific prompts
        self.stage1_system_prompt = None
        self.stage2_system_prompt = None
        self.set_task_system_prompts(task_name)
        
        # Will be set in load_model
        self.structurer_llm = None
        self.structurer_tokenizer = None

    def load_model(self):
        """Load the LLM(s) for both stages."""
        self.llm = LLM(
            model=self.model_name,
            gpu_memory_utilization=self.gpu_memory_utilization,
            dtype=self.dtype,
            enforce_eager=self.enforce_eager,
        )
        self.tokenizer = self.llm.get_tokenizer()
        
        # Load separate structurer model if specified
        if self.structurer_model_name != self.model_name:
            self.structurer_llm = LLM(
                model=self.structurer_model_name,
                gpu_memory_utilization=self.gpu_memory_utilization,
                dtype=self.dtype,
                enforce_eager=self.enforce_eager,
            )
            self.structurer_tokenizer = self.structurer_llm.get_tokenizer()
        else:
            self.structurer_llm = self.llm
            self.structurer_tokenizer = self.tokenizer

    def set_task_system_prompts(self, task: str) -> None:
        """Set system prompts for both stages based on task."""
        if task == "gsm8k":
            self.stage1_system_prompt = GSM8K_STAGE1_SYSTEM_PROMPT
            self.stage2_system_prompt = GSM8K_STAGE2_SYSTEM_PROMPT
        elif task == "math500":
            self.stage1_system_prompt = MATH500_STAGE1_SYSTEM_PROMPT
            self.stage2_system_prompt = MATH500_STAGE2_SYSTEM_PROMPT
        elif task == "data_table_analysis":
            self.stage1_system_prompt = DATA_TABLE_ANALYSIS_STAGE1_SYSTEM_PROMPT
            self.stage2_system_prompt = DATA_TABLE_ANALYSIS_STAGE2_SYSTEM_PROMPT
        elif task == "insurance_claims":
            self.stage1_system_prompt = INSURANCE_CLAIMS_STAGE1_SYSTEM_PROMPT
            self.stage2_system_prompt = INSURANCE_CLAIMS_STAGE2_SYSTEM_PROMPT
        elif task == "financial_entities":
            self.stage1_system_prompt = FINANCIAL_ENTITIES_STAGE1_SYSTEM_PROMPT
            self.stage2_system_prompt = FINANCIAL_ENTITIES_STAGE2_SYSTEM_PROMPT
        elif task == "pii_extraction":
            self.stage1_system_prompt = PII_EXTRACTION_STAGE1_SYSTEM_PROMPT
            self.stage2_system_prompt = PII_EXTRACTION_STAGE2_SYSTEM_PROMPT
        elif task == "gsm_symbolic":
            self.stage1_system_prompt = GSM_SYMBOLIC_STAGE1_SYSTEM_PROMPT
            self.stage2_system_prompt = GSM_SYMBOLIC_STAGE2_SYSTEM_PROMPT
        elif task == "prover9":
            self.stage1_system_prompt = PROVER9_STAGE1_SYSTEM_PROMPT
            self.stage2_system_prompt = PROVER9_STAGE2_SYSTEM_PROMPT
            for fewshot in PROVER9_SYSTEM_PROMPT_WITH_FEWSHOTS:
                self.chat_fewshots_stage1.append({"role": "user", "content": fewshot['question']})
                self.chat_fewshots_stage1.append({"role": "assistant", "content": fewshot['response']})
            for fewshot in PROVER9_STAGE1_SYSTEM_PROMPT_WITH_FEWSHOTS:
                user_content = (
                    f"For the Question: {fewshot['question']}, Your prior answer is:"
                    f"{fewshot['response']}"
                    "Now, arrange your answer in Predicates, Premises, and Conclusion fields.  Just write the 'Predicates:', 'Premises:', and 'Conclusion:' fields. DO NOT OUTPUT ANYTHING ELSE OTHER THAN THESE 3 FIELDS! After then end of Conclusion, write \\n------"
                )
                self.chat_fewshots_stage2.append({"role": "user", "content":  user_content})
                self.chat_fewshots_stage2.append({"role": "assistant", "content": fewshot['response']})

    def create_stage1_prompts(self, texts: List[str]) -> List[str]:
        """Create prompts for Stage 1 (free-form generation)."""
        prompts = []
        for text in texts:
            if self.chat_fewshots_stage1:
                prompt = [
                    {"role": "system", "content": self.stage1_system_prompt},
                    *self.chat_fewshots_stage1,
                    {"role": "user", "content": text + " After then end of conclusion, write \\n------"}
                ]
            else:
                prompt = [
                    {"role": "system", "content": self.stage1_system_prompt},
                    {"role": "user", "content": text}
                ]
            templated_prompt = self.tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )
            prompts.append(templated_prompt)
        return prompts

    def create_stage2_prompts(self, questions: List[str], stage1_outputs: List[str]) -> List[str]:
        """Create prompts for Stage 2 (extraction to JSON)."""
        prompts = []
        for question, prior_answer in zip(questions, stage1_outputs):
            if self.question_in_stage2:
                user_content = (
                    f"For the Question: {question}, Your prior answer is:"
                    f"{prior_answer}"
                    "Now, arrange your answer in Predicates, Premises, and Conclusion fields.  Just write the 'Predicates:', 'Premises:', and 'Conclusion:' fields. DO NOT OUTPUT ANYTHING ELSE OTHER THAN THESE 3 FIELDS! After then end of Conclusion, write \\n------"
                )
            else:
                user_content = (
                    f"Prior solution (source text; do not add new facts):\n{prior_answer}"
                )
            if self.chat_fewshots_stage2:
                prompt = [
                    {"role": "system", "content": self.stage2_system_prompt},
                    *self.chat_fewshots_stage2,
                    {"role": "user", "content":  user_content}
                ]
            else:
                prompt = [
                    {"role": "system", "content": self.stage2_system_prompt},
                    {"role": "user", "content": user_content}
                ]
            templated_prompt = self.structurer_tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )
            prompts.append(templated_prompt)
        return prompts

    def generate_stage1(self, texts: List[str]) -> List[str]:
        """Stage 1: Generate free-form solutions."""
        prompts = self.create_stage1_prompts(texts)
        llm_responses = self.llm.generate(
            prompts=prompts, 
            sampling_params=self.stage1_sampling_params
        )
        return [o.outputs[0].text for o in llm_responses]

    def generate_stage2(self, questions: List[str], stage1_outputs: List[str]) -> List[str]:
        """Stage 2: Extract structured JSON from free-form solutions."""
        prompts = self.create_stage2_prompts(questions, stage1_outputs)
        llm_responses = self.structurer_llm.generate(
            prompts=prompts,
            sampling_params=self.stage2_sampling_params
        )
        return [o.outputs[0].text for o in llm_responses]

    def generate(self, prompt: str) -> str:
        """Single example generation (both stages)."""
        stage1_output = self.generate_stage1([prompt])[0]
        stage2_output = self.generate_stage2([prompt], [stage1_output])[0]
        return stage2_output

    def generate_batch(self, texts: List[str]) -> List[str]:
        """
        Batch generation with two stages.
        Returns the final structured JSON outputs.
        """
        # Stage 1: Free-form generation
        stage1_outputs = self.generate_stage1(texts)
        
        # Stage 2: Extract to structured JSON
        json_string_responses = self.generate_stage2(texts, stage1_outputs)
        
        return json_string_responses, stage1_outputs

    def generate_batch_with_intermediate(self, texts: List[str]) -> Dict[str, List[str]]:
        """
        Batch generation returning both intermediate and final outputs.
        Useful for debugging and analysis.
        """
        # Stage 1: Free-form generation
        stage1_outputs = self.generate_stage1(texts)
        
        # Stage 2: Extract to structured JSON
        stage2_outputs = self.generate_stage2(texts, stage1_outputs)
        
        return {
            "stage1_freeform": stage1_outputs,
            "stage2_structured": stage2_outputs,
        }

    def close(self):
        """Release resources for both LLMs."""
        if self._closed:
            return
        
        # Close structurer LLM if it's different from main LLM
        if self.structurer_llm is not None and self.structurer_llm != self.llm:
            try:
                self.structurer_tokenizer = None
                engine = getattr(self.structurer_llm, "llm_engine", None)
                if engine is not None:
                    executor = getattr(engine, "executor", None)
                    for obj in (executor, engine, self.structurer_llm):
                        for m in ("shutdown", "close", "destroy", "stop"):
                            fn = getattr(obj, m, None)
                            if callable(fn):
                                try:
                                    fn()
                                except Exception:
                                    pass
            except Exception:
                pass
            self.structurer_llm = None
        
        # Call parent close for main LLM
        super().close()

