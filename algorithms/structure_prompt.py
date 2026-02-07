GSM8K_STRUCTURE_PROMPT = """You must output a single JSON object and nothing else.

Output schema (strict):
{
  "steps": [string, string, ...],
  "answer": string
}

Rules:
- The output must be valid JSON (double quotes for keys/strings, no trailing commas).
- "steps" must be a JSON array of strings, each string describing one small reasoning step.
- "answer" must be a single string containing only the final numeric answer (no units, no extra text).
- Do not include the question, do not include markdown, and do not add any extra keys beyond "steps" and "answer".
- Keep steps concise and minimal; include only what is necessary to reach the answer.
"""

MATH500_STRUCTURE_PROMPT = """You must output a single JSON object and nothing else.

Output schema (strict):
{
  "steps": [string, string, ...],
  "answer": string
}

Rules:
- The output must be valid JSON (double quotes for keys/strings, no trailing commas).
- "steps" must be a JSON array of strings. Inline math is allowed using single-dollar LaTeX like $x^2$.
- Each step should describe a logically valid transformation (e.g., set up equations, simplify, factor, solve).
- "answer" must be a single string containing only the final result (e.g., "98", "5/6", "-3, 2", "3:5").
- Do not include any extra explanation outside the JSON.
- Do not include markdown, and do not add any extra keys beyond "steps" and "answer".
"""

GSM_SYMBOLIC_STRUCTURE_PROMPT = """You must output only a single symbolic expression wrapped in << and >>, and nothing else.

Rules:
- Output format: <<EXPR>>
- EXPR must be an algebraic expression using the provided variables (e.g., t, tf, c, nc, etc.).
- Do not evaluate, simplify aggressively, or substitute numbers; keep it in symbolic form.
- Use standard operators: +, -, *, /, and parentheses when needed.
- Do not output JSON, do not add words, and do not include multiple candidates—only one <<...>> expression.
"""

PROVER9_STRUCTURE_PROMPT = """You must output a Prover9-style logical specification string with exactly these three sections in this order:
Predicates:
Premises:
Conclusion:

Formatting rules:
- The output must be plain text (not JSON, not markdown).
- Each predicate line must follow: PredicateName(args) ::: natural language description.
- Each premise line must be a well-formed formula using Prover9-style tokens:
  {forall}, {exists}, {implies}, {and}, {or}, {not}, {xor}
- Each premise line must end with: ::: natural language paraphrase.
- The conclusion section must contain exactly one formula line in the same style, ending with ::: paraphrase.
- Use consistent constants for entities (e.g., rina, miroslav) and consistent predicate names across the item.
- Do not include proofs, truth labels (true/false/uncertain), or any extra sections.
- End the output with a final line containing exactly: ------ 
"""
