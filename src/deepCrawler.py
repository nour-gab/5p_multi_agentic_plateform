"""
Main module for the DeepCrawlerAgent.
Handles the orchestration of searching, extracting, and summarizing market data.
"""

import os
import json
from utils.search_api import query_sources
from utils.extractors import extract_content
from utils.loggers import setup_logger

class DeepCrawlerAgent:
    """
    Agent for extracting and summarizing market data using Tavily, MCP, and Gemini.
    """
        
    # def __init__(self, prompt: str, top_k: int = 20):
    #     """
    #     Initializes the DeepCrawlerAgent with a meta-prompt and config.
    #     """
    #     self.prompt = prompt
    #     self.top_k = top_k
    #     self.context = self._parse_prompt(prompt)
    #     self.search_queries = self.context.get("search_queries", [])
    #     # self.api_results: List[Dict[str, Any]] = []
    #     # self.entities: List[Dict[str, Any]] = []
    #     self.logger = setup_logger("DeepCrawlerAgent")

    def __init__(self, porter_force: str, keywords: str):
        """
        Initialize the agent with a Porter force and search keywords.
        """
        self.porter_force = porter_force
        self.keywords = keywords
        self.logger = setup_logger("DeepCrawlerAgent")
    

    def _generate_better_keywords(self, idea: str, porter_force: str) -> str:
        """
        Use Gemini to suggest improved/expanded keyword searches based on the idea and Porter force.
        """
        from google.generativeai import GenerativeModel
        model = GenerativeModel("gemini-2.5-flash")
        prompt = (
            f"Given the business idea: '{idea}' and the Porter force: '{porter_force}', "
            "suggest a list of improved and expanded keyword searches to maximize relevant data extraction from the web. "
            "Return a comma-separated list of keywords or queries."
        )
        response = model.generate_content(prompt)
        # Return as a single string (comma-separated)
        return response.text.strip()

    def run(self):
        """
        Run the full pipeline: generate better keywords, search, extract raw data, and save results.
        """
        self.logger.info(f"Starting crawl for: {self.porter_force} / {self.keywords}")
        # Use Gemini to generate better keywords
        improved_keywords = self._generate_better_keywords(self.keywords, self.porter_force)
        self.logger.info(f"Improved keywords suggested by Gemini: {improved_keywords}")
        # Parse Gemini's response into a list, trim to fit Tavily's 400-char limit
        keyword_list = [k.strip('" ') for k in improved_keywords.split(',') if k.strip()]
        query = ''
        for k in keyword_list:
            next_query = (query + ' OR ' if query else '') + k
            if len(next_query) > 400:
                break
            query = next_query
        if not query:
            query = self.keywords  # fallback
        self.logger.info(f"Final Tavily query: {query}")
        sources = query_sources(self.porter_force, query)
        all_data = []

        for source in sources:
            url = source['url']
            self.logger.info(f"Extracting raw content from: {url}")
            content = extract_content(url)
            # Save all raw info, including timestamp, duplicates, etc.
            entry = {
                "source_name": source.get('name', 'Unknown'),
                "url": url,
                "porter_force": self.porter_force,
                "keywords": improved_keywords,
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
        with open("prompts/prompt.txt", encoding="utf-8") as f:
            prompt = f.read()
        # prompt = prompt_template.format(
        #     URL=source['url'],
        #     CONTENT=content,
        #     PORTER_FORCE=self.porter_force,
        #     KEYWORDS=self.keywords,
        #     SOURCE_NAME=source.get('name', 'Unknown')
        # )
        response = model.generate_content(prompt)
        # Expecting the LLM to return a JSON string as per the prompt template
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
    # Example usage
    agent = DeepCrawlerAgent(porter_force="Buyer Power", keywords="Chatbot assisstant for sales")
    agent.run()