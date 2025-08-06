import json
import requests
import re
import time
import os
from merge_Report import generate_report



# ==== CONFIG ====
MISTRAL_API_KEY = "wycDAXufddC3GSFF6TyvC4yXUlmUyFC1"
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
MODEL = "open-mistral-7b"

MERGED_REPORT_PATH = "merged_report.json"
RAGS_PATH = "rags.json"
VERDICT_PATH = "outputs/verdict.json"

# ==== PROMPTS ====
def structure_check_prompt(report):
    return f"""
You are an evaluator. Reply ONLY with a JSON object.

Does the report contain all 5 of Porter's Five Forces?
Return strictly this JSON: 
{{
  "Threat of New Entrants": "Present" or "Missing",
  "Bargaining Power of Suppliers": "Present" or "Missing",
  "Bargaining Power of Buyers": "Present" or "Missing",
  "Threat of Substitutes": "Present" or "Missing",
  "Industry Rivalry": "Present" or "Missing"
}}

Report:
{report}
"""

def coherence_check_prompt(report):
    return f"""
You are a writing evaluator.

Your task is to assess the **coherence and logical flow** of the following analytical report.

Return strictly JSON in the format below:
{{
  "flow_score": integer from 1 to 10,
  "issues": [
    "brief description of any logic, clarity, or organization issues"
  ]
}}

Report:
{report}
"""

def hallucination_check_prompt(report, rags):
    rag_text = "\n\n".join([f"{k}:\n{v}" for k, v in rags.items()])
    return f"""
You are an expert fact-checker.

Validate whether each statement in the report is grounded in the provided source RAGs.

RAGs:
{rag_text}

Report:
{report}

if it's not in the provided source RAGs return strictly JSON:
{{
  "hallucinations": [
    {{
      "sentence": "...",
      "issue": "Not found in RAGs"
    }}
  ]
}}
"""

# ==== UTILITIES ====
def extract_json(raw_text: str, label: str = "LLM"):
    print(f"🔍 {label} Raw Output:\n", raw_text)
    match = re.search(r'\{[\s\S]*\}', raw_text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            print(f" {label} JSON invalid.")
            print(match.group())
            exit()
    else:
        print(f" No JSON detected in {label}.")
        print(raw_text)
        exit()

def ask_mistral(prompt):
    response = requests.post(
        MISTRAL_API_URL,
        headers={
            "Authorization": f"Bearer {MISTRAL_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 1024
        }
    )
    try:
        result = response.json()
        return result['choices'][0]['message']['content']
    except Exception as e:
        print(" Mistral API error:")
        print(json.dumps(response.json(), indent=2))
        raise e

def load_report_text():
    if not os.path.exists(MERGED_REPORT_PATH):
        print(f" File not found: {MERGED_REPORT_PATH}")
        exit()
    with open(MERGED_REPORT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    report_text = data.get("content", "").strip()
    if not report_text:
        print(" 'content' field is empty in merged_report.json")
        exit()
    return report_text

def load_rags():
    if not os.path.exists(RAGS_PATH):
        print(f" File not found: {RAGS_PATH}")
        exit()
    with open(RAGS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

# ==== MAIN PIPELINE ====
def evaluate():
    generate_report()  
    merged_report = load_report_text()
    rags = load_rags()

    # Run checks
    structure_out = extract_json(ask_mistral(structure_check_prompt(merged_report)), "Structure")
    coherence_out = extract_json(ask_mistral(coherence_check_prompt(merged_report)), "Coherence")
    hallucination_out = extract_json(ask_mistral(hallucination_check_prompt(merged_report, rags)), "Hallucination")

    # Determine verdict
    missing_sections = [k for k, v in structure_out.items() if v == "Missing"]
    hallucinations = hallucination_out.get("hallucinations", [])
    flow_score = coherence_out.get("flow_score", 5)

    verdict = "Fail" if missing_sections or hallucinations or flow_score < 6 else "Pass"

    # Save output
    output = {
        "verdict": verdict,
        "structure": structure_out,
        "coherence": coherence_out,
        "hallucinations": hallucinations
    }

    os.makedirs("outputs", exist_ok=True)
    with open(VERDICT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(" Verdict generated at outputs/verdict.json")

def check_verdict():
    if not os.path.exists(VERDICT_PATH):
        print("❌ verdict.json not found.")
        return "Fail"
    with open(VERDICT_PATH, "r", encoding="utf-8") as f:
        return json.load(f).get("verdict", "Fail")

# ==== LOOP UNTIL PASS ====
if __name__ == "__main__":
    attempt = 1
    while True:
        print(f"\n🔁 Tentative #{attempt}")
        evaluate()
        print("evaluation OK")
        verdict = check_verdict()
        if verdict == "Pass":
            print(" Verdict final : PASS \n")
            break
        else:
            print(" Verdict: FAIL - Retrying...\n")
            attempt += 1
            time.sleep(2)
