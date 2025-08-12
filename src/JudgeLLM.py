import json
import re
from typing import Dict
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
load_dotenv()

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
        model: str = "qwen/qwen3-32b",
    ):
        if not api_key:
            raise ValueError("Groq API key is required.")

        self.llm = ChatGroq(
            model=model,
            temperature=0.2,
            max_tokens=1024,
            api_key=api_key
        )
        self.parser = JsonOutputParser()

    def _ask_model(self, prompt: str) -> Dict:
        chain = PromptTemplate.from_template(prompt) | self.llm | self.parser
        try:
            result = chain.invoke({})
            return result
        except Exception as e:
            print(f"❌ Groq error: {e}")
            raise

    def evaluate(self, merged_report: str, rags: Dict) -> Dict:
        # Structure Check
        structure_prompt = structure_check_prompt(merged_report)
        structure_out = self._ask_model(structure_prompt)

        # Coherence Check
        coherence_prompt = coherence_check_prompt(merged_report)
        coherence_out = self._ask_model(coherence_prompt)

        # Hallucination Check
        hallucination_prompt = hallucination_check_prompt(merged_report, rags)
        hallucination_out = self._ask_model(hallucination_prompt)

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

if __name__ == "__main__":
    import os
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Please set the GROQ_API_KEY environment variable.")

    judge = LLMJudge(api_key=api_key)
    merged_report_path = "data/report/merged_report.txt"
    with open(merged_report_path, "r", encoding="utf-8") as f:
        merged_report = f.read()

    rags_path = "data/report/report.json"
    with open(rags_path, "r", encoding="utf-8") as f:
        rags = json.load(f)
    
    result = judge.run(merged_report, rags)
    print(json.dumps(result, indent=2, ensure_ascii=False))