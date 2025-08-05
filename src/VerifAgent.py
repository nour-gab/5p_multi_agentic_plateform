import os
import json
import nltk
import time
from keybert import KeyBERT
import google.generativeai as genai
from typing import List, Dict, Optional, Tuple

# Ensure required NLTK data is downloaded
# nltk.download('punkt', quiet=True)
nltk.download('punkt_tab')

# Path to save report
REPORT_PATH = "idea/idea_report.json"

class KeywordExtractor:
    """Extracts top keywords from text using KeyBERT."""
    def __init__(self):
        try:
            self.model = KeyBERT(model="paraphrase-MiniLM-L6-v2")
        except Exception as e:
            print(f"Failed to load KeyBERT model: {e}")
            self.model = None

    def extract(self, text: str, top_n: int = 10) -> List[str]:
        if not self.model:
            return []
        try:
            keywords = self.model.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 2),
                stop_words='english',
                top_n=top_n
            )
            return [kw[0] for kw in keywords]
        except Exception as e:
            print(f"Keyword extraction error: {e}")
            return []


class FintechIdeaAnalyzer:
    def __init__(self, api_key: Optional[str]):
        if not api_key:
            raise ValueError("API key is required for Gemini")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.keyword_extractor = KeywordExtractor()

    def validate_idea(self, text: str) -> Tuple[bool, Optional[str]]:
        """Validates that the idea has enough content for analysis."""
        if not text or len(text.strip()) < 20:
            return False, "Please provide a more detailed idea (min. 20 characters)."

        try:
            tokens = nltk.word_tokenize(text)
            if len(tokens) < 5:
                return False, "The idea is too vague. Provide more context."
        except Exception as e:
            print(f"Tokenization error: {e}")
            return False, "Error during input analysis."

        return True, None

    def create_prompt(self, idea_text: str, keywords: List[str]) -> str:
        """Constructs analysis prompt for Gemini model."""
        keywords_str = ', '.join(keywords) if keywords else 'No keywords extracted'

        return f"""
You are a fintech expert analyzing innovative financial ideas.

IDEA:
{idea_text}

KEYWORDS:
{keywords_str}

FINTECH CONTEXT:
Includes: digital payments, blockchain, crypto, digital banking, credit/lending, investment platforms, insurtech, regtech, neobanks, crowdfunding, robo-advisors, KYC/AML, banking APIs, embedded finance.

TASK:
Analyze this idea based on:

1. FINTECH RELEVANCE (0-10)
   - Is it clearly within financial services?
   - Does it use tech to innovate finance?

2. COMPLETENESS (0-10)
   - Clear target audience?
   - Specific problem?
   - Clear tech solution?
   - Clear value proposition?

3. FEASIBILITY (0-10)
   - Is it technically feasible?
   - Does the market exist?

RESPONSE FORMAT (Valid JSON):
{{
  "is_fintech": true/false,
  "fintech_score": 0-10,
  "completeness_score": 0-10,
  "feasibility_score": 0-10,
  "overall_score": 0-10,
  "category": "main_fintech_category",
  "missing_elements": ["element1", "element2"],
  "strengths": ["strength1", "strength2"],
  "recommendation": "APPROVE" | "DISCUSS" | "REJECT",
  "feedback": "brief feedback"
}}

IMPORTANT: Only return valid JSON. No explanation text.
"""

    def analyze(self, idea_text: str) -> Dict:
        is_valid, error = self.validate_idea(idea_text)
        if not is_valid:
            return {"status": "error", "message": error}

        keywords = self.keyword_extractor.extract(idea_text)
        prompt = self.create_prompt(idea_text, keywords)

        try:
            response = self.model.generate_content(prompt)
            analysis = json.loads(response.text.strip())
        except Exception as e:
            print(f"Analysis error: {e}")
            return {"status": "error", "message": "Failed to analyze the idea."}

        return self.process_analysis_result(idea_text, analysis)

    def process_analysis_result(self, idea_text: str, analysis: Dict) -> Dict:
        """Adds metadata and writes to JSON file."""
        result = {
            "idea": idea_text,
            "analysis": analysis,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
        try:
            with open(REPORT_PATH, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Failed to write report: {e}")
            return {"status": "error", "message": "Could not save the analysis report."}

        return {
            "status": "success",
            "message": "Idea analyzed and report saved.",
            "report_path": REPORT_PATH
        }


# Usage
if __name__ == "__main__":
    api_key = "AIzaSyDpP8HI8SdZ0APj5HcAQMvtyI2ECFwqso8"  # Replace with a valid Gemini API key
    idea = input("Enter your fintech idea (in English): ")

    analyzer = FintechIdeaAnalyzer(api_key)
    result = analyzer.analyze(idea)

    print(json.dumps(result, indent=2))
