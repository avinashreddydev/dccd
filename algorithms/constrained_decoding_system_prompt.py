from pathlib import Path
import yaml
from yaml import load, Loader
PROJECT_DIR = Path(__file__).parent.parent
GSM_SYMBOLIC_PROMPT_PATH = PROJECT_DIR / "algorithms" / "prompt_templates" / "gsm_symbolic.yaml"
GSM_SYMBOLIC_PROMPT = load(open(GSM_SYMBOLIC_PROMPT_PATH, "r"), Loader=Loader)

PROVER9_PROMPT_PATH = PROJECT_DIR / "algorithms" / "prompt_templates" / "fol.yaml"
PROVER9_PROMPT = load(open(PROVER9_PROMPT_PATH, "r"), Loader=Loader)

GSM8K_SYSTEM_PROMPT =  (
                "You are a meticulous math tutor. Solve the problem step by step and OUTPUT ONLY a single JSON object "
                    "conforming to the provided schema, '{steps: string[], answer: string}'. No extra text, no code fences, no commentary outside JSON."
                    "result in 'answer'. No extra text, no code fences, no commentary outside JSON."
            )

MATH500_SYSTEM_PROMPT = (
    "You are a meticulous math tutor. Solve the problem step by step and OUTPUT ONLY a single JSON object "
    "conforming to the provided schema, '{steps: string[], answer: string}'. No extra text, no code fences, no commentary outside JSON."
    "result in 'answer'. No extra text, no code fences, no commentary outside JSON."
)
DATA_TABLE_ANALYSIS_SYSTEM_PROMPT = (
                "You are given a CSV-like string representation of a table (with header row, no index)."
                "Extract a structured JSON object following the provided response format class."
                "Do not guess: if a value does not exist or is not applicable, return null."
                "Count rows excluding the header. "
                "Infer each column type as 'str', 'int', or 'float'. "
                "For string columns, set min/max to null. "
                "If the 'Identifier' column is missing, set all Identifier-related fields to null. "
                "For null/None entries, set string columns to '', and numerical to None. "
                "Return only the structured JSON object."
            )
    
FINANCIAL_ENTITIES_SYSTEM_PROMPT = """Identify and extract entities from the following financial news text into the following categories:

Entity 1: Company 
⋆ Definition: Denotes the official or unofficial name of a registered company or a brand.
⋆ Example entities: {Apple Inc.; Uber; Bank of America}

Entity 2: Date 
⋆ Definition: Represents a specific time period, whether explicitly mentioned (e.g., "year ended March 2020") or implicitly referred to (e.g., "last month"), in the past, present, or future.
⋆ Example entities: {June 2nd, 2010; quarter ended 2021; last week; prior year; Wednesday}

Entity 3: Location 
⋆ Definition: Represents geographical locations, such as political regions, countries, states, cities, roads, or any other location, even when used as adjectives.
⋆ Example entities: {California; Paris; 1280 W 12th Blvd; Americas; Europe}

Entity 4: Money 
⋆ Definition: Denotes a monetary value expressed in any world currency, including digital currencies.
⋆ Example entities: {$76.3 million; $4 Bn; Rs 33.80 crore; 1.2 BTC}

Entity 5: Person 
⋆ Definition: Represents the name of an individual.
⋆ Example entities: {Meg Whitman; Mr. Baker; Warren Buffet}

Entity 6: Product 
⋆ Definition: Refers to any physical object or service manufactured or provided by a company to consumers, excluding references to businesses or sectors within the financial context.
⋆ Example entities: {iPhone; Tesla model X; cloud services; Microsoft Windows 10; laptops; medical equipment; computer software; online classes; eye surgery}

Entity 7: Quantity 
⋆ Definition: Represents any numeric value that is not categorized as Money, such as percentages, numbers, measurements (e.g., weight, length), or other similar quantities. Note that unit of measurements are also part of the entity.
⋆ Example entities: {15%; 25,000 units; 2.75in; 100 tons}

For each category:
- Extract all relevant entities as a list of strings, preserving the wording from the text
- Use None if no entities are found in that category
- Only extract entities that are explicitly mentioned in the text itself, do not make inferences or reason about what entities might be implied based on URLs, domain names, or other indirect references
- Extract individual items rather than compound or ranged entities (e.g., if a range or compound entity is mentioned, extract each individual item separately)

Return the extracted information as a JSON object with all categories included, using None for cases where no entities are found.
"""

INSURANCE_CLAIMS_SYSTEM_PROMPT = """You are an expert insurance claim processor. Extract structured information from insurance claim descriptions.
For dates, use YYYY-MM-DD format.
If a piece of information does not exist in the claim description, return null instead of making assumptions.
Be thorough in extracting all available information and categorize appropriately."""



PII_EXTRACTION_SYSTEM_PROMPT =  """Your task is to extract structured information about PII entities from the text provided by the user.
For each field in the response format, extract the corresponding PII entity if it exists in the text.
If a particular PII entity is not present in the text, set that field to null.
Return the complete structured response with all fields, setting missing entities to null."""



GSM_SYMBOLIC_SYSTEM_PROMPT_WITH_FEWSHOTS =  GSM_SYMBOLIC_PROMPT["fewshots"]["std"]["text"]

GSM_SYMBOLIC_SYSTEM_PROMPT_WITH_FEWSHOT_TEXT = "\n Fewshots:\n"

for fewshot in GSM_SYMBOLIC_SYSTEM_PROMPT_WITH_FEWSHOTS:
    GSM_SYMBOLIC_SYSTEM_PROMPT_WITH_FEWSHOT_TEXT += f"Question: {fewshot['question']}\nResponse: {fewshot['response']}\n"


GSM_SYMBOLIC_SYSTEM_PROMPT = GSM_SYMBOLIC_PROMPT["std_instruct"]["text"].replace("[[START]]", "<<").replace("[[END]]", ">>")

PROVER9_SYSTEM_PROMPT = PROVER9_PROMPT["task_specification"]

PROVER9_SYSTEM_PROMPT_WITH_FEWSHOTS = PROVER9_PROMPT["fewshots"]["std"]["prover9"]




