"""
Main module for the DeepCrawlerAgent.
Handles the orchestration of searching, extracting, and summarizing market data.
"""

import os
import json
import re
from utils.search_api import query_sources
from utils.extractors import extract_content
from utils.loggers import setup_logger

class DeepCrawlerAgent:
    """
    Agent for extracting and summarizing market data using Tavily, MCP, and Gemini.
    """
        
    def __init__(self, porter_force: str, prompt_file: str = "prompts/prompt.txt"):
        """
        Initialize the agent with a Porter force and path to the metaprompter's prompt file.
        """
        self.porter_force = porter_force
        self.prompt_file = prompt_file
        self.logger = setup_logger("DeepCrawlerAgent")
        self.prompt = self._load_prompt()
        self.keywords = self._extract_keywords_from_prompt()

    def _load_prompt(self) -> str:
        """
        Load the prompt from the specified prompt file.
        """
        try:
            with open(self.prompt_file, encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            self.logger.error(f"Failed to load prompt from {self.prompt_file}: {e}")
            return ""

    def _extract_keywords_from_prompt(self) -> str:
        """
        Extract the keywords from the 'Force-Specific Keyword Search Strategy' section of the prompt.
        Returns a comma-separated string of keywords.
        """
        if not self.prompt:
            self.logger.warning("No prompt loaded, falling back to empty keywords")
            return ""

        # Look for the section starting with "Force-Specific Keyword Search Strategy"
        match = re.search(
            r"\*\*Force-Specific Keyword Search Strategy.*?\n(.*?)(?=\n\*\*|$)",
            self.prompt,
            re.DOTALL
        )
        if not match:
            self.logger.warning("No keywords found in prompt")
            return ""

        # Extract the list of keywords (assuming they are numbered list items)
        keyword_section = match.group(1)
        keywords = re.findall(r"^\d+\.\s*(.*?)$", keyword_section, re.MULTILINE)
        keywords = [k.strip('" ') for k in keywords if k.strip()]
        return ", ".join(keywords)

    def _construct_tavily_query(self, keywords: str) -> str:
        """
        Construct a Tavily-compatible query from keywords, respecting the 400-character limit.
        Prioritizes keywords to maximize relevance within the limit.
        """
        keyword_list = [k.strip('" ') for k in keywords.split(',') if k.strip()]
        query = ""
        max_length = 400 - len(self.porter_force) - 1  # Account for Porter force and space
        for i, keyword in enumerate(keyword_list):
            # Add Porter force to each keyword for relevance
            next_segment = f"{self.porter_force} {keyword}"
            next_query = (query + " OR " if query else "") + next_segment
            if len(next_query) > max_length:
                if i == 0:
                    # If even the first keyword is too long, truncate it
                    self.logger.warning(f"Single keyword too long, truncating: {keyword}")
                    return f"{self.porter_force} {keyword[:max_length - len(self.porter_force) - 1]}"
                break
            query = next_query
        if not query:
            self.logger.warning("No valid query constructed, falling back to Porter force")
            query = self.porter_force
        return query

    def run(self):
        """
        Run the full pipeline: use prompt keywords, search, extract raw data, and save results.
        """
        self.logger.info(f"Starting crawl for: {self.porter_force} / {self.keywords}")
        # Use keywords directly from the prompt
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
        with open(self.prompt_file, encoding="utf-8") as f:
            prompt = f.read()
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
    agent = DeepCrawlerAgent(porter_force="Rivalry")
    agent.run()