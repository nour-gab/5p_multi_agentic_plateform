"""
MetaPromptAgent: Generates strategic, force-specific instructions for DeepCrawlerAgent in AI/Data domains.
Uses Gemini 2.5 Flash via Google Generative AI API. Reads enhanced idea from VerifAgent's output.
Outputs prompt to ./prompts/prompt.txt.
"""
import os
import json
from typing import Dict
import google.generativeai as genai
from google.generativeai import GenerativeModel
from dotenv import load_dotenv

# Load environment and configure Gemini API key ONCE at module level
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment.")
genai.configure(api_key=api_key)

class MetaPromptAgent:
    """
    Generates a meta-prompt for the DeepCrawlerAgent based on VerifAgent's enhanced idea and Porter Force.
    Outputs the prompt to ./prompts/prompt.txt for downstream use.
    """
    def __init__(self, verif_report_path: str = "idea/idea_report.json"):
        self.model = GenerativeModel("gemini-2.5-flash")
        self.verif_report_path = verif_report_path
        self.prompt_template = """
        You are a strategy analyst generating a crawling strategy for the DeepCrawlerAgent to analyze the Financial Technology (FinTech) industry using Porter's Five Forces framework.

        Based on the provided Porter Force and business context, generate a JSON object with:
        - A list of 5-10 high-impact, force-specific search keywords tailored for the FinTech sector.
        - A brief crawling strategy description (100-150 words).
        - Recommended APIs/data platforms (e.g., Crunchbase, CB Insights, SEC EDGAR).

        Inputs:
        - **Porter Force**: {porter_force}
        - **Business Context**: {project_idea}

        Force-Specific Guidance:
        - *Buyer Power*: Keywords like "customer retention", "switching costs", "price sensitivity", focus on customer segments and demand.
        - *Supplier Power*: Keywords like "cloud providers", "regulatory compliance", focus on technology vendors.
        - *New Entrants*: Keywords like "startup funding", "barriers to entry", focus on VC trends and regulations.
        - *Substitute Products*: Keywords like "open banking", "alternative platforms", focus on disruptive technologies.
        - *Rivalry*: Keywords like "market share", "pricing wars", focus on major players and acquisitions.

        Constraints:
        - Optimize for performance (response in <60s).
        - Keywords must be specific, relevant, and actionable for Tavily/Gemini APIs.
        - Focus on data from the last 6-12 months.

        Return a JSON object with:
        {{
          "porter_force": str,
          "keywords": [str],
          "strategy": str,
          "recommended_apis": [str]
        }}
        """
        # """
        # You are the DeepCrawler Agent embedded within an agentic system that analyzes the Financial Technology (FinTech) industry using Porter's Five Forces framework.
        # Your mission is to autonomously plan and execute a deep market and ecosystem crawl using tools like Tavily, Gemini, and MCP APIs.

        # You must dynamically tailor your crawling logic based on the strategic context provided, gathering high-fidelity insights specific to FinTech trends, technologies, regulatory shifts, and ecosystem dynamics.

        # Instructions:
        # 1. Based on the selected **Porter Force** (e.g., Buyer Power, New Entrants), generate a focused data acquisition and crawling strategy customized for the FinTech sector.
        # 2. Use your expertise in FinTech domains—such as digital banking, blockchain, embedded finance, neobanks, regtech, insurtech, and payment platforms—to tailor keyword strategies, APIs, and crawl targets.
        # 3. Generate a list of force-specific, high-impact search queries optimized for Tavily, Gemini, and other tools.
        # 4. Recommend the best APIs and data platforms (e.g., Crunchbase, CB Insights, SEC EDGAR, global regulatory portals, LinkedIn company intelligence, financial news feeds) that yield the most relevant insights.
        # 5. Return structured and ranked results, including link metadata, sentiment analysis, signal strength, and force-specific relevance.

        # Inputs:
        # - **Porter Force**: {porter_force}
        # - **Project Idea / Business Context**: {project_idea}

        # Force-Specific Guidance:
        # - *Buyer Power*: Focus on customer segments, their bargaining leverage, switching costs, and demand for customized FinTech solutions.
        # - *Supplier Power*: Analyze key technology providers, cloud infrastructure dependencies, and regulatory compliance vendors.
        # - *New Entrants*: Identify barriers to entry, emerging startups, and venture capital trends shaping FinTech.
        # - *Substitutes*: Examine alternative technologies, non-FinTech competitors, and open banking innovations disrupting legacy FinTech.
        # - *Rivalry*: Benchmark major players across payments, credit, wealth management, insurtech, and regtech. Include VC activity, acquisitions, user growth, and pricing wars.

        # Constraints:
        # - Optimize for performance—complete response in under 60 seconds.
        # - Use cosine similarity or Gemini-based validation to filter low-relevance sources.
        # - Focus on global results with emphasis on regional relevance and data from the last 6–12 months.
        # - Respect API call limits and cache results when appropriate.

        # Goal:
        # Provide strategic analysts with actionable, force-specific insights that enable deep understanding of the FinTech ecosystem surrounding the provided business context.

        # Begin execution using:
        # - {porter_force}
        # - {project_idea}
        # """
        # self.output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'prompts', 'prompt.txt'))

    def load_enhanced_idea(self) -> str:
        """
        Loads the enhanced idea from VerifAgent's output (idea/idea_report.json).
        """
        if not os.path.exists(self.verif_report_path):
            raise FileNotFoundError(f"VerifAgent report not found at {self.verif_report_path}")
        with open(self.verif_report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        enhanced_idea = report.get("idea", "")
        if not enhanced_idea:
            raise ValueError("No enhanced idea found in VerifAgent report")
        return enhanced_idea
    

    def load_enhanced_idea(self) -> str:
        """
        Loads the enhanced idea from VerifAgent's output (idea/idea_report.json).
        Returns the enhanced idea or raises an error if not found.
        """
        if not os.path.exists(self.verif_report_path):
            raise FileNotFoundError(f"VerifAgent report not found at {self.verif_report_path}")
        with open(self.verif_report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        enhanced_idea = report.get("idea", "")
        if not enhanced_idea:
            raise ValueError("No enhanced idea found in VerifAgent report")
        return enhanced_idea


    
    def generate_prompt(self, porter_force: str) -> Dict:
        """
        Generates the meta-prompt and saves it to prompts/{porter_force}_prompt.json.
        Returns the JSON object.
        """
        enhanced_idea = self.load_enhanced_idea()
        prompt = self.prompt_template.format(
            porter_force=porter_force,
            project_idea=enhanced_idea
        )
        response = self.model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(response_mime_type="application/json")
        )
        try:
            instructions = json.loads(response.text)
        except Exception as e:
            raise ValueError(f"Failed to parse Gemini response as JSON: {e}")
        
        # Ensure the output has the required structure
        output = {
            "porter_force": porter_force,
            "keywords": instructions.get("keywords", []),
            "strategy": instructions.get("strategy", ""),
            "recommended_apis": instructions.get("recommended_apis", [])
        }
        
        # Save to JSON file
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'prompts'))
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{porter_force}_prompt.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        return output

if __name__ == "__main__":
    #input("Enter Porter Force (e.g., Buyer Power): ").strip() or
    porter_force =  "Buyer Power"
    agent = MetaPromptAgent()
    result = agent.generate_prompt(porter_force)
    print("\nGenerated DeepCrawler Agent MetaPrompt (written to ./prompts/prompt.json):\n")
    print(result)