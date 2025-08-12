def structure_check_prompt(report):
    return """
You are an evaluator. You must reply ONLY with a valid JSON object — no explanations, no extra text, no tags, no reasoning.

Does the report contain all 5 of Porter's Five Forces?
The answer should be either "Present" or "Missing" for each force.

Return strictly this JSON format (replace only the values with your answer):
{{
  "Threat of New Entrants": "Present" or "Missing",
  "Bargaining Power of Suppliers": "Present" or "Missing",
  "Bargaining Power of Buyers": "Present" or "Missing",
  "Threat of Substitutes": "Present" or "Missing",
  "Industry Rivalry": "Present" or "Missing"
}}

/no_think

Report:
{report}
""".replace("{", "{{").replace("}", "}}").replace("{{report}}", report)



def coherence_check_prompt(report):
    return """
You are a writing evaluator.

Your task is to assess the **coherence and logical flow** of the following analytical report.

Specifically, consider:
- Logical structure and flow between ideas and sections
- Clarity of reasoning (are conclusions justified?)
- Redundancy or repetition
- Sentence-level readability (confusing phrasing, overly long constructions)

Do **not** penalize:
- Stylistic tone (unless it affects clarity)
- Use of bullet points or formatting, unless it breaks the logic

Return strictly JSON in the format below. Do not include any natural language commentary before or after.

{{
  "flow_score": integer from 1 to 10,  // 1 = incoherent, 10 = excellent flow
  "issues": [
    "brief description of any logic, clarity, or organization issues",
    ...
  ]
}}

Report:
{report}
""".replace("{", "{{").replace("}", "}}").replace("{{report}}", report)


    return f"""
Evaluate this report's coherence.
Return strictly JSON:
{{
  "flow_score": integer 1-10,
  "issues": ["brief description of any logic, clarity, redundancy issues"]
}}

Report:
{report}
"""

def hallucination_check_prompt(report, rags):
    rag_text = "\n\n".join([f"{k}:\n{v}" for k, v in rags.items()])
    return """
You are an expert fact-checker.

Your task is to validate whether each statement in the report is grounded in the provided source RAGs.

Accept statements that:
- Are directly supported by the RAGs.
- Express the same idea as the RAGs, even if phrased differently (e.g., paraphrased, summarized).

Flag statements as hallucinations if:
- The idea is not present in the RAGs at all.
- The claim introduces new information or assumptions not supported by the RAGs.
- The tone, interpretation, or implication goes beyond what's stated in the RAGs.

Be strict but fair: the goal is to detect **unjustified invention**, not stylistic difference.

RAGs:
{rag_text}

Report:
{report}

Return strictly JSON in the format below. Do not include any explanation or natural language before or after the JSON.

{{
  "hallucinations": [
    {{
      "sentence": "...",
      "issue": "Not found in RAGs"  // Or "Idea present but rephrased beyond safe interpretation"
    }}
  ]
}}
""".replace("{", "{{").replace("}", "}}").replace("{{report}}", report).replace("{{rag_text}}", rag_text)


    rag_text = "\n\n".join([f"{k}:\n{v}" for k, v in rags.items()])
    return f"""
Fact-check this report based on the source RAGs.

RAGs:
{rag_text}

Report:
{report}

Return strictly JSON:
{{
  "hallucinations": [
    {{
      "sentence": "...",
      "issue": "Not found in RAGs"
    }}
  ]
}}
"""



def final_judge_prompt(structure_json: str, coherence_json: str, hallucination_json: str) -> str:
    return """
You are an expert judge. Given these evaluation results:

Structure Evaluation:
{structure_json}

Coherence Evaluation:
{coherence_json}

Hallucination Evaluation:
{hallucination_json}

Determine if the report PASSES all criteria:
- All Porter forces present
- Coherence flow score ≥ 7
- No hallucinations detected

Return ONLY a JSON with this schema:
{{
  "passed": boolean,
  "reasons": [string],
  "details": {{
    "structure": object,
    "coherence": object,
    "hallucination": object
  }}
}}
""".replace("{", "{{").replace("}", "}}").replace("{{structure_json}}", structure_json).replace("{{coherence_json}}", coherence_json).replace("{{hallucination_json}}", hallucination_json)