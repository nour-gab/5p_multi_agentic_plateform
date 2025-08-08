import json
import re
import requests
from typing import Dict

# Prompts are imported from utils
from utils.judge_prompts import (
    structure_check_prompt,
    coherence_check_prompt,
    hallucination_check_prompt,
    final_judge_prompt  # Optional: if needed later
)

class LLMJudge:
    def __init__(
        self,
        api_key: str,
        model: str = "open-mistral-7b",
        api_url: str = "https://api.mistral.ai/v1/chat/completions"
    ):
        self.api_key = api_key
        self.model = model
        self.api_url = api_url

    def _ask_model(self, prompt: str) -> str:
        response = requests.post(
            self.api_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "max_tokens": 1024
            }
        )

        try:
            result = response.json()
            return result['choices'][0]['message']['content']
        except Exception as e:
            print("❌ Error with Mistral API response:")
            print(json.dumps(response.json(), indent=2))
            raise e

    def _extract_json(self, raw_text: str, label: str) -> Dict:
        print(f"🔍 {label} Raw Output:\n", raw_text)
        match = re.search(r'\{[\s\S]*\}', raw_text)
        if not match:
            raise ValueError(f"❌ No JSON found in {label} output.")
        
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            print(f"❌ Invalid JSON in {label}:")
            print(match.group())
            raise

    def evaluate(self, merged_report: str, rags: Dict) -> Dict:
        # Structure Check
        structure_prompt = structure_check_prompt(merged_report)
        structure_raw = self._ask_model(structure_prompt)
        structure_out = self._extract_json(structure_raw, label="Structure")

        # Coherence Check
        coherence_prompt = coherence_check_prompt(merged_report)
        coherence_raw = self._ask_model(coherence_prompt)
        coherence_out = self._extract_json(coherence_raw, label="Coherence")

        # Hallucination Check
        hallucination_prompt = hallucination_check_prompt(merged_report, rags)
        hallucination_raw = self._ask_model(hallucination_prompt)
        hallucination_out = self._extract_json(hallucination_raw, label="Hallucination")

        # Final verdict
        missing_sections = [k for k, v in structure_out.items() if v == "Missing"]
        hallucinations = hallucination_out.get("hallucinations", [])
        flow_score = coherence_out.get("flow_score", 5)

        verdict = "Fail" if missing_sections or hallucinations or flow_score < 6 else "Pass"

        result = {
            "verdict": verdict,
            "structure": structure_out,
            "coherence": coherence_out,
            "hallucinations": hallucinations
        }

        # Save result
        self._save_verdict(result)
        return result

    def _save_verdict(self, result: Dict, path: str = "outputs/verdict.json"):
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"✅ Verdict saved to {path}")
    def run(self, merged_report: str, rags: Dict) -> Dict:
        """ Main entry point to evaluate the merged report against RAGs. """
        try:
            return self.evaluate(merged_report, rags)
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            return {
                "status": "error",
                "message": str(e)
            }

