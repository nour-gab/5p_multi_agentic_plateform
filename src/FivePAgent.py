import os
import json
import google.generativeai as genai
from typing import Optional
from dotenv import load_dotenv

load_dotenv()
# api_key = os.getenv("GOOGLE_API_KEY")

class FivePAgent:
    """
    Agent that generates a strategic analysis report based on Porter's Five Forces,
    using structured RAG (retrieved augmented generation) notes.
    """

    def __init__(
        self,
        api_key: str,
        rag_path: str = "data/report/report.json",
        output_path: str = "data/report/merged_report.txt",
        model_name: str = "gemini-2.5-flash"
    ):
        if not api_key:
            raise ValueError("Gemini API key is required.")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.rag_path = rag_path
        self.output_path = output_path

    def load_rags(self) -> str:
        """Loads and formats the RAGs into a structured text block."""
        if not os.path.exists(self.rag_path):
            raise FileNotFoundError(f"RAG file not found: {self.rag_path}")

        with open(self.rag_path, "r", encoding="utf-8") as f:
            rags = json.load(f)

        rag_text = "\n\n".join([f"{force}:\n{content}" for force, content in rags.items()])
        return rag_text

    def build_prompt(self, rag_text: str) -> str:
        """Builds the full analysis prompt with instructions."""
        return f"""
You are a strategy analyst. Using the following notes based on Porter's Five Forces, write a coherent, non-redundant and well-structured analytical report.

Guidelines:
- Use a formal tone and analytical style.
- Ensure coherence and logical flow between sections.
- Do not hallucinate information; stick strictly to the RAGs.
- Use headings for each force.
- Begin with a short introduction (2-3 sentences).
- End with a short conclusion summarizing competitive insights.

Source Notes (RAGs):
{rag_text}

Return only the final report as plain text (no JSON or explanations).
"""

    def generate_report(self, prompt: str) -> Optional[str]:
        """Calls Gemini API to generate the strategy report."""
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"❌ Gemini error: {e}")
            return None

    def save_report(self, report: str) -> None:
        """Saves the final report to a .txt file."""
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            f.write(report)

    def run(self) -> dict:
        """Main method to load RAGs, generate the report, and save it."""
        try:
            rag_text = self.load_rags()
            prompt = self.build_prompt(rag_text)
            report = self.generate_report(prompt)

            if not report:
                return {
                    "status": "error",
                    "message": "Failed to generate report from Gemini."
                }

            self.save_report(report)

            return {
                "status": "success",
                "message": f"Report saved to '{self.output_path}'",
                "report_text": report
            }

        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

def main():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Please set the GOOGLE_API_KEY environment variable.")

    agent = FivePAgent(
        api_key=api_key,
        rag_path="data/report/report.json",
        output_path="data/report/merged_report.txt"
    )
    result = agent.run()
    print(json.dumps(result, indent=2, ensure_ascii=False))
if __name__ == "__main__":
    main()