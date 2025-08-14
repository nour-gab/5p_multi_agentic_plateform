# dashboard_llm_fixed.py

import json
import re
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
# from langchain.output_parsers import OutputFixingParser, JsonOutputParser
from langchain.output_parsers import OutputFixingParser
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

load_dotenv()


# ---------------------
# Pydantic model for dashboard
# ---------------------
class Visualization(BaseModel):
    description: str
    code: str


class ForceSection(BaseModel):
    insights: list[str]
    recommendations: list[str]
    visualizations: list[Visualization]


class DashboardData(BaseModel):
    Rivalry: ForceSection
    Buyer_Power: ForceSection
    Supplier_Power: ForceSection
    Threat_of_New_Entrants: ForceSection
    Threat_of_Substitute_Products: ForceSection
    overall_summary: dict


# ---------------------
# Helper to extract JSON safely
# ---------------------
def extract_json_block(text: str) -> str:
    """
    Extract the first valid JSON object from a text string.
    Removes <think>...</think> and markdown fences.
    """
    # # Remove <think> blocks
    # text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # # Remove markdown fences
    # text = re.sub(r"```(?:json)?", "", text)
    # # Extract first {...} block
    # match = re.search(r"\{(?:[^{}]|(?R))*\}", text, re.DOTALL)
    # if not match:
    #     raise ValueError("No JSON object found in model output")
    # return match.group(0)
    
    start = text.find("{")
    if start == -1:
        return None
    brace_count = 0
    for i, char in enumerate(text[start:], start=start):
        if char == "{":
            brace_count += 1
        elif char == "}":
            brace_count -= 1
            if brace_count == 0:
                return text[start:i+1]
    return None



# ---------------------
# Main class
# ---------------------
class DashboardAgent:
    def __init__(self, api_key: str, model: str = "deepseek-r1-distill-llama-70b"):
        if not api_key:
            raise ValueError("Groq API key is required.")
        self.llm = ChatGroq(model=model, temperature=0.2, max_tokens=2048, api_key=api_key)
        json_parser = JsonOutputParser(pydantic_object=DashboardData)
        self.parser = OutputFixingParser.from_llm(parser=json_parser, llm=self.llm)

    def generate_dashboard(self, report: str) -> dict:
        prompt_text = """
You are an AI assistant. You will be given a FinTech Porter's Five Forces report.
You must output ONLY a valid JSON object following the schema below:
{format_instructions}

CRITICAL RULES:
- Output JSON only, no explanations, no <think> tags, no markdown.
- Ensure it is strictly valid JSON.
- All fields must follow the schema exactly.
- Visualization code must be valid Python (matplotlib).

Report:
{report}
"""
        prompt = PromptTemplate(
            input_variables=["report"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
            template=prompt_text
        )
        chain = prompt | self.llm
        raw_output = chain.invoke({"report": report}).content

        try:
            json_str = extract_json_block(raw_output)
            return json.loads(json_str)
        except Exception as e:
            raise ValueError(f"Failed to parse LLM response: {e}\nRaw Output:\n{raw_output}")


# ---------------------
# Script entry
# ---------------------
if __name__ == "__main__":
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Please set the GROQ_API_KEY environment variable.")

    report_path = "data/report/merged_report.txt"
    with open(report_path, "r", encoding="utf-8") as f:
        report_text = f.read()

    llm = DashboardAgent(api_key=api_key)
    dashboard_data = llm.generate_dashboard(report_text)

    # Save to file
    os.makedirs("outputs", exist_ok=True)
    out_path = "outputs/dashboard.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dashboard_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Dashboard saved to {out_path}")
