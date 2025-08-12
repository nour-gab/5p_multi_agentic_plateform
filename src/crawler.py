import json
import os
import re
from utils.search_api import query_sources
from utils.extractors import extract_content
from utils.loggers import setup_logger


class DeepCrawlerAgent:
    """
    Agent for extracting and summarizing market data using Tavily, MCP, and Gemini.
    """
    def __init__(self, porter_force: str, prompt_file: str = None):
        """
        Initialize the agent with a Porter force. Optionally accepts prompt_file for compatibility.
        Reads keywords from prompts/{porter_force}_prompt.json.
        """
        self.porter_force = porter_force
        self.logger = setup_logger("DeepCrawlerAgent")
        self.prompt_file = f"prompts/{porter_force}_prompt.json"
        self.keywords = self._load_keywords()

    def _load_keywords(self) -> str:
        """
        Load keywords from the JSON prompt file.
        Returns a comma-separated string of keywords.
        """
        try:
            with open(self.prompt_file, "r", encoding="utf-8") as f:
                prompt_data = json.load(f)
            keywords = prompt_data.get("keywords", [])
            if not keywords:
                self.logger.warning("No keywords found in prompt JSON, using Porter force as query")
                return self.porter_force
            return ", ".join(keywords)
        except Exception as e:
            self.logger.error(f"Failed to load prompt from {self.prompt_file}: {e}")
            return self.porter_force

    def _construct_tavily_query(self, keywords: str) -> str:
        """
        Construct a query for Tavily search from keywords.
        """
        return f"{self.porter_force} {keywords} FinTech market trends"

    def run(self):
        """
        Execute the crawling process: search, extract content, and save to JSON.
        """
        self.logger.info(f"Starting crawl for Porter force: {self.porter_force}")
        if not self.keywords:
            self.logger.warning("No keywords available, using Porter force as query")
            query = self.porter_force
        else:
            query = self._construct_tavily_query(self.keywords)
        self.logger.info(f"Final Tavily query: {query}")
        try:
            sources = query_sources(self.porter_force, query)
        except Exception as e:
            self.logger.error(f"Tavily search failed: {e}")
            sources = []
        all_data = []

        for source in sources:
            url = source['url']
            self.logger.info(f"Extracting raw content from: {url}")
            content = extract_content(url)
            entry = {
                "source_name": source.get('name', 'Unknown'),
                "url": url,
                "porter_force": self.porter_force,
                "keywords": self.keywords,
                "raw_content": content,
                "timestamp": __import__('datetime').datetime.utcnow().isoformat() + 'Z'
            }
            all_data.append(entry)

        os.makedirs("data/raw", exist_ok=True)
        output_path = f"data/raw/{self.porter_force}_crawl.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Crawl complete with {len(all_data)} entries. Output: {output_path}")
        return all_data

    def _summarize_with_llm(self, source, content):
        """
        Use Gemini LLM to summarize and structure the extracted content.
        """
        from google.generativeai import GenerativeModel
        model = GenerativeModel("gemini-2.5-flash")
        with open(self.prompt_file, "r", encoding="utf-8") as f:
            prompt_data = json.load(f)
        prompt = f"""
        Summarize the following content for Porter's {self.porter_force} analysis in the FinTech sector:
        {content}
        Use these keywords: {prompt_data.get("keywords", [])}
        Return a JSON object with:
        {{
          "source_name": str,
          "porter_force": str,
          "summary": str,
          "key_metrics": [str],
          "data_points": [str]
        }}
        """
        response = model.generate_content(prompt)
        try:
            summary_json = json.loads(response.text)
        except Exception:
            summary_json = {
                "source_name": source.get('name', 'Unknown'),
                "porter_force": self.porter_force,
                "summary": response.text,
                "key_metrics": [],
                "data_points": []
            }
        return summary_json

if __name__ == "__main__":
    agent = DeepCrawlerAgent(porter_force="Buyer Power")
    agent.run()