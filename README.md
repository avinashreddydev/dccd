# Draft-Conditioned Constrained Decoding for Structured Generation in LLMs

**Authors:** Avinash Reddy, Thayne T. Walker, James S. Ide, Amrit Singh Bedi  

**Affiliations:**
*   Department of Computer Science, University of Central Florida, FL, US
*   Lockheed Martin AI Center, CT, US

## Abstract

Large language models (LLMs) are increasingly used to generate executable outputs, JSON objects, and API calls, where a single syntax error can make the output unusable. Constrained decoding enforces validity token-by-token via masking and renormalization, but it can distort generation when the model assigns low probability mass to valid continuations, pushing decoding toward locally valid yet semantically incorrect trajectories. We propose **Draft-Conditioned Constrained Decoding (DCCD)**, a simple two-step, training-free inference procedure that decouples semantic planning from structural enforcement: an unconstrained draft is generated first, and constrained decoding is then applied, conditioned on this draft, to guarantee validity. We analyze DCCD through a KL-projection view, showing that draft conditioning increases feasible mass and reduces the cumulative “projection tax” induced by hard constraints, with an optional best-of-K draft selection. Across structured reasoning benchmarks, DCCD improves strict structured accuracy by up to +24 percentage points over standard constrained decoding (e.g., 15.2% to 39.0% on GSM8K with a 1B model), and enables smaller model pairs to match or exceed much larger constrained baselines, yielding substantial gains in parameter efficiency.

## Repository Overview

This repository contains code for **Draft-Conditioned Constrained Decoding (DCCD)** and other constrained decoding algorithms for structured generation tasks using LLMs.

### Algorithms Implemented
*   **DCCD (Two-Stage Decoding)**: Decouples semantic planning (drafting) from structural enforcement (constraints).
*   **Standard Constrained Decoding**: Token-by-token masking and renormalization.
*   **Few-Shot Constrained Prompting**: Using examples to guide structured output.

## Installation

1.  Clone the repository:
    ```bash
    git clone <repo_url>
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Main Experiment Loader (`main.py`)

This script loads configurations and runs experiments for different algorithms and tasks.

```bash
python main.py
```

Configuration files are located in `configs/`.

### Generation Script (`generate.py`)

This script uses vLLM to generate responses, either constrained or unconstrained.

```bash
python generate.py --model meta-llama/Llama-3.2-3B-Instruct --outdir outputs/
```

**Arguments:**
*   `--model`: Path to the model or HuggingFace model ID (default: `meta-llama/Llama-3.2-3B-Instruct`).
*   `--unconstrained`: If set, converts prompts to unconstrained "essay-style" before generation.
*   `--max-tokens`: Max new tokens to generate (default: 256).
*   `--outdir`: Directory to save JSON results.
*   `--gpu-mem-util`: vLLM GPU memory utilization (default: 0.95).

## Directory Structure

*   `algorithms/`: Implementation of decoding algorithms (including DCCD).
*   `data_loaders/`: Data loaders for various tasks (GSM8K, etc.).
*   `configs/`: Configuration files (YAML).
*   `prompts/`: Prompt templates.

## Requirements

*   Python 3.11+
*   PyTorch
*   vLLM
*   Transformers
*   Pydantic
