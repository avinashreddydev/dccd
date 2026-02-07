#!/usr/bin/env python3
import argparse
import json
import os
from datetime import datetime

from vllm import LLM, SamplingParams
from prompts import tldr_prompts_detailed  # your 100-prompt dict
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Let user/environment decide
# ---------------------------
# CLI
# ---------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--unconstrained", action="store_true",
                    help="If set, convert prompts to unconstrained (essay-style) before generation.")
parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B-Instruct",
                    help="Path or name of the model for vLLM.")
parser.add_argument("--max-tokens", type=int, default=256,
                    help="Max new tokens to generate.")
parser.add_argument("--outdir", type=str, default="outputs",
                    help="Directory to write JSON results.")
parser.add_argument("--gpu-mem-util", type=float, default=0.95,
                    help="vLLM gpu_memory_utilization parameter.")
parser.add_argument("--dtype", type=str, default="bfloat16",
                    help="Model dtype (e.g., float16, bfloat16, auto).")
args = parser.parse_args()

# ---------------------------
# LLM + Tokenizer
# ---------------------------
llm = LLM(
    model=args.model,
    gpu_memory_utilization=args.gpu_mem_util,
    dtype=args.dtype
)
tokenizer = llm.get_tokenizer()

sampling_params = SamplingParams(
    max_tokens=args.max_tokens,
)

# ---------------------------
# Prompt templating
# ---------------------------
def to_unconstrained(text: str) -> str:
    """
    Convert a TL;DR-constrained prompt into an unconstrained 'essay' variant.
    We conservatively replace common phrasings.
    """
    replacements = [
        ("a ~256-word TL;DR", "an essay"),
        ("~256-word TL;DR", "essay"),
        ("a 256-word TL;DR", "an essay"),
        ("256-word TL;DR", "essay"),
        ("~256 words", "an essay"),
        ("≈256 words", "an essay"),
        ("~256 word", "an essay"),
        ("256 words", "an essay"),
    ]
    new_text = text
    for old, new in replacements:
        new_text = new_text.replace(old, new)
    return new_text

def create_prompt(prompt, unconstrained: bool = False) -> str:
    if unconstrained:
        prompt = to_unconstrained(prompt)

    prompt_message = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": prompt},
    ]


    templated = tokenizer.apply_chat_template(
        prompt_message,
        tokenize=False,
        add_generation_prompt=True
    )
    return templated




def create_prompt_single_stage_constrained(prompt: str, max_tokens: int, unit: str = "tokens") -> str:
    """
    Single-stage content generation under a hard budget.
    The model is asked to produce a final answer <= max_tokens (unit), with clear coverage and coherence.
    """
    system_msg = (
        "You are a specialist Content-Generation agent. Produce a comprehensive, high-quality answer that "
        f"fits within <= {max_tokens} {unit} while preserving essential meaning.\n"
        "Requirements:\n"
        "1) Accuracy: ensure all factual/analytical content is correct.\n"
        "2) Coverage: address all key aspects of the prompt with sufficient depth.\n"
        "3) Coherence: complete sentences; avoid redundancy; clear logical structure.\n"
        "4) Plain text only: no headings, no JSON, no markdown, no preamble or postscript.\n"
        "5) Style: concise, professional; prefer clarity over flourish.\n"
        "6) Termination: end on a sentence boundary; never cut mid-sentence.\n"
        f"7) Length target: aim for ~95-100% of the {max_tokens}-{unit} budget without exceeding it."
    )

    user_msg = (
        "PROMPT:\n"
        f"{prompt}\n\n"
        "TASK:\n"
        f"Write the final answer that fits within <= {max_tokens} {unit}, preserving essential information. "
        "Output ONLY the answer text."
    )

    prompt_message = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    templated = tokenizer.apply_chat_template(
        prompt_message, tokenize=False, add_generation_prompt=True
    )
    return templated


def create_prompt_stage1_unconstrained(prompt: str, target_downstream_budget: int, unit: str = "tokens") -> str:
    """
    Two-stage pipeline: Stage 1 (unconstrained).
    Generate a full, detailed answer (no explicit budget) that maximizes correctness and coverage.
    Note: we mention the downstream budget so the model includes enough detail for later compression, but we DO NOT constrain length here.
    """
    system_msg = (
        "You are a specialist Content-Generation agent. Generate a comprehensive, high-quality answer to the prompt "
        "without any explicit length limitation.\n"
        "Requirements:\n"
        "1) Accuracy: ensure all factual/analytical content is correct.\n"
        "2) Coverage: address every key aspect thoroughly; include critical definitions, numbers, and caveats.\n"
        "3) Reasoning: present a clear logical structure; explain steps/causal links where useful.\n"
        "4) Clarity: complete, well-formed sentences; avoid unnecessary repetition.\n"
        "5) Plain text only: no headings, no JSON, no markdown.\n"
        f"6) Note: this answer may be summarized later into ~{target_downstream_budget} {unit}; provide enough detail to support faithful compression."
    )

    user_msg = (
        "PROMPT:\n"
        f"{prompt}\n\n"
        "TASK:\n"
        "Generate a full, detailed, unconstrained answer that completely and accurately addresses the prompt. "
        "Do not limit length or oversimplify. Output ONLY the final comprehensive text."
    )

    prompt_message = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    templated = tokenizer.apply_chat_template(
        prompt_message, tokenize=False, add_generation_prompt=True
    )
    return templated







def create_prompt_for_constraint_enforcing(prompt, prev_answer, max_tokens, unit="tokens"):
    """
    Build a strong 'Constraint Enforcer' prompt that compresses a previous answer
    into a faithful TL;DR within a hard budget (default: tokens).
    - Preserves essentials (claim, key points, critical numbers, conclusion)
    - No new facts, no extra formatting
    - Ends on a sentence boundary (avoid mid-sentence truncation)
    """
    system_msg = (
        "You are a specialist Constraint-Enforcer agent. Your task is to compress a prior answer "
        f"into a high-quality TL;DR that is <= {max_tokens} {unit} while preserving essential meaning.\n"
        "Requirements:\n"
        "1) Faithful: use ONLY information from the provided prior answer; do NOT invent new facts.\n"
        "2) Coverage-first: include the core claim,  key points, critical numbers, and the conclusion.\n"
        "3) Coherent: complete sentences; no bullet lists unless already present; avoid redundancy.\n"
        "4) Plain text only: no headings, no JSON, no markdown, no preamble or postscript.\n"
        "5) Style: concise, active voice; remove fluff; merge overlapping ideas.\n"
        "6) Termination: end on a sentence boundary; never cut a sentence mid-way.\n"
        "7) Trade-offs: if you must drop content to fit, prefer breadth of essential points over minor details.\n"
        f"8) Length target: aim for ~95-100% of the {max_tokens}-{unit} budget without exceeding it."
    )

    user_msg = (
        "PROMPT:\n"
        f"{prompt}\n\n"
        "PREVIOUS ANSWER (source content; compress faithfully without adding new facts):\n"
        f"{prev_answer}\n\n"
        "TASK:\n"
        f"Write a TL;DR that fits within <= {max_tokens} {unit}, retaining only essential information needed to answer the prompt. "
        "Output ONLY the final TL;DR text."
    )

    prompt_message = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    templated = tokenizer.apply_chat_template(
        prompt_message,
        tokenize=False,
        add_generation_prompt=True
    )
    return templated



# ---------------------------
# Generate
# ---------------------------
results = {}
for category, prompts in tldr_prompts_detailed.items():
    if args.unconstrained:
        templated_prompts = [create_prompt_stage1_unconstrained(p, args.max_tokens) for p in prompts]
    else:
        templated_prompts = [create_prompt_single_stage_constrained(p, args.max_tokens) for p in prompts]
    if args.unconstrained:
        sampling_params = SamplingParams(
            max_tokens=8192,
        )
    else:
        sampling_params = SamplingParams(
            max_tokens=args.max_tokens,
        )
    outputs = llm.generate(templated_prompts, sampling_params=sampling_params)
    # vLLM returns a list of RequestOutput; each has .outputs (list) with candidates
    responses = [o.outputs[0].text for o in outputs]
    
    response_lengths =[ len(o.outputs[0].token_ids) for o in outputs]
    
    # Store prompt-response pairs for this category
    cat_items = []
    if args.unconstrained:
        new_templated_prompts = [create_prompt_for_constraint_enforcing(p, a, args.max_tokens) for p,a in zip(prompts, responses)]
        sampling_params = SamplingParams(
            max_tokens=args.max_tokens,
        )
        new_outputs = llm.generate(new_templated_prompts, sampling_params=sampling_params)
        new_responses = [o.outputs[0].text for o in new_outputs]
        new_lens =  [ len(o.outputs[0].token_ids) for o in new_outputs]

        for p, r, l, nr, nl in zip(prompts, responses, response_lengths, new_responses, new_lens):
            cat_items.append({
                "prompt": p,
                "response": r, 
                "response_len" :l,
                "constrained_response" : nr, 
                "constrained_response_len" : nl
            })
    else:
        for p, r, l in zip(prompts, responses, response_lengths):
            cat_items.append({
                "prompt": p,
                "response": r, 
                "response_len" :l
            })



    results[category] = cat_items

# ---------------------------
# Save JSON only
# ---------------------------
os.makedirs(args.outdir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
mode_tag = "unconstrained" if args.unconstrained else "constrained"
json_path = os.path.join(args.outdir, f"{mode_tag}_results_{timestamp}.json")

payload = {
    "meta": {
        "mode": mode_tag,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "dtype": args.dtype,
        "timestamp": timestamp
    },
    "results": results
}

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=4, ensure_ascii=False)

print(f"✅ Saved JSON results to: {json_path}")
