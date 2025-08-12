import os
import json
from typing import Dict, Any
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()

class DashboardAgent:
    def __init__(self, api_key: str, model: str = "deepseek-r1-distill-llama-70b"):
        self.llm = ChatGroq(model=model, temperature=0.2, api_key=api_key)

    def generate_analysis(self, verdict: Dict[str, Any]) -> Dict[str, Any]:
        prompt_template = """
Analyze the following verdict from the Judge LLM and generate an in-depth dashboard report.
Include:
1. Key Insights: Summarize the main findings from structure, coherence, and hallucinations.
2. Recommendations: Suggest improvements based on issues found.
3. Visualizations: Describe several visualizations (e.g., bar charts for scores, pie charts for issues) and provide Python code snippets using matplotlib to generate them.

Verdict:
{verdict}

Return a JSON with keys: "insights", "recommendations", "visualizations" (list of dicts with "description" and "code").
"""
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | self.llm
        response = chain.invoke({"verdict": json.dumps(verdict)}).content
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"error": "Failed to parse LLM response"}

    def run(self, verdict_path: str = "outputs/verdict.json") -> Dict[str, Any]:
        with open(verdict_path, "r", encoding="utf-8") as f:
            verdict = json.load(f)
        analysis = self.generate_analysis(verdict)
        # Optionally execute visualization code using code_execution tool, but for now return the analysis
        return analysis

# Example usage
if __name__ == "__main__":
    api_key = os.getenv("GROQ_API_KEY")
    agent = DashboardAgent(api_key=api_key)
    result = agent.run()
    print(json.dumps(result, indent=2))