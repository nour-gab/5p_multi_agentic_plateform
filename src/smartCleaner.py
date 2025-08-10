"""
Cleaner module for processing DeepCrawlerAgent output using LangGraph.
Cleans data, extracts entities and dates, summarizes with LLM, and prepares for RAG and visualization.
"""

import re
import uuid
import pandas as pd
import json
from datetime import datetime
from typing import Dict, Any, Optional, List, TypedDict, Annotated
import os
from dateutil.parser import parse as parse_date
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import operator
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# Load spaCy model
try:
    import spacy
    nlp = spacy.load("en_core_web_lg")
except (ImportError, OSError):
    print("spaCy model 'en_core_web_lg' not found. Entity extraction disabled.")
    nlp = None

class CleanerState(TypedDict):
    """State for the LangGraph workflow."""
    raw_data: List[Dict[str, Any]]
    processed_insights: List[Dict[str, Any]]
    porter_force: str
    output_path: str
    deduplication_keys: List[str]

class CleanerAgent:
    """
    Agent for cleaning and structuring data from DeepCrawlerAgent using LangGraph.
    """

    def __init__(self):
        self.porter_forces = ["Rivalry", "Buyer Power", "Supplier Power", "New Entrants", "Substitutes"]
        self.llm = ChatGroq(model="qwen/qwen3-32b", 
                            temperature=0.2,
                            api_key=GROQ_API_KEY)
        self.summarize_prompt = ChatPromptTemplate.from_template(
            "Summarize the following content in 100-150 words, focusing on key insights relevant to {porter_force} in the MLOps market for SMBs. Highlight competitors, market trends, or strategic moves. Ensure the summary is concise, clear, and suitable for a RAG report and visualization dashboard:\n\n{content}"
        )

    def load_data(self, state: CleanerState) -> CleanerState:
        """Load crawler output from JSON file."""
        input_path = f"data/raw/{state['porter_force']}_crawl.json"
        try:
            with open(input_path, "r", encoding="utf-8") as f:
                state["raw_data"] = json.load(f)
            print(f"✅ Data loaded. Rows: {len(state['raw_data'])}")
        except FileNotFoundError:
            print(f"Error: Input file {input_path} not found.")
            state["raw_data"] = []
        return state

    def clean_and_structure_insight(self, raw_insight: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Clean and structure a single raw insight."""
        raw_content = raw_insight.get("raw_content", "")
        if not raw_content or len(raw_content) < 50 or any(phrase in raw_content.lower() for phrase in ["unavailable", "go back", "not found"]):
            return None

        cleaned_insight = {
            "id": str(uuid.uuid4()),
            "source_name": self._clean_string(raw_insight.get("source_name")),
            "url": self._normalize_url(raw_insight.get("url")),
            "keywords": [],
            "content_summary": None,
            "timestamp": self._clean_string(raw_insight.get("timestamp")),
            "date_cleaned": datetime.now().isoformat() + "Z",
            "source": raw_insight.get("source_name", "Unknown"),
            "entities": {"persons": [], "orgs": [], "locations": [], "products": []},
            "dates": [],
            "porter_force": raw_insight.get("porter_force", "Unknown"),
            "raw_content": raw_content  # Add raw_content for summarization
        }

        # Process keywords
        raw_keywords = raw_insight.get("keywords", "")
        if isinstance(raw_keywords, str):
            cleaned_insight["keywords"] = [self._clean_string(kw.strip()) for kw in raw_keywords.split(",") if kw.strip()]
        elif isinstance(raw_keywords, list):
            cleaned_insight["keywords"] = [self._clean_string(kw) for kw in raw_keywords if kw]
        return cleaned_insight

    def extract_entities_and_dates(self, state: CleanerState) -> CleanerState:
        """Extract entities and dates from raw content."""
        processed_insights = []
        for raw_insight in state["raw_data"]:
            cleaned_insight = self.clean_and_structure_insight(raw_insight)
            if cleaned_insight and nlp:
                raw_content = cleaned_insight["raw_content"]
                doc = nlp(raw_content)
                entities = {"persons": [], "orgs": [], "locations": [], "products": []}
                for ent in doc.ents:
                    if ent.label_ == "PERSON":
                        entities["persons"].append(ent.text)
                    elif ent.label_ == "ORG":
                        entities["orgs"].append(ent.text)
                    elif ent.label_ == "GPE":
                        entities["locations"].append(ent.text)
                    elif ent.label_ == "PRODUCT":
                        entities["products"].append(ent.text)
                cleaned_insight["entities"] = entities
                cleaned_insight["dates"] = self._extract_dates(raw_content)
            if cleaned_insight:
                processed_insights.append(cleaned_insight)
        state["processed_insights"] = processed_insights
        return state

    def _extract_dates(self, text: str) -> List[str]:
        """Extract dates from text using dateutil.parser."""
        try:
            dates = []
            for token in text.split():
                try:
                    parsed_date = parse_date(token, fuzzy=True)
                    dates.append(parsed_date.isoformat())
                except (ValueError, TypeError):
                    continue
            return list(set(dates))  # Remove duplicates
        except Exception:
            return []

    def summarize_with_llm(self, state: CleanerState) -> CleanerState:
        """Summarize content using LLM for each insight."""
        for insight in state["processed_insights"]:
            raw_content = insight.get("raw_content", "")
            if raw_content:
                porter_force = insight["porter_force"]  # Use per-insight porter_force
                prompt = self.summarize_prompt.format(porter_force=porter_force, content=raw_content)
                try:
                    summary = self.llm.invoke(prompt).content
                    insight["content_summary"] = summary
                except Exception as e:
                    print(f"Error summarizing content: {e}")
                    insight["content_summary"] = "Summary unavailable due to processing error."
            else:
                insight["content_summary"] = "No content available"
        return state

    def deduplicate_and_save(self, state: CleanerState) -> CleanerState:
        """Deduplicate insights and save to JSON."""
        df = pd.DataFrame(state["processed_insights"])
        if df.empty:
            print("Error: No valid insights to process.")
            return state

        # Filter low-quality insights
        df = df[
            (df["content_summary"].str.len() > 50) &
            (~df["content_summary"].str.contains("unavailable|go back|not found", case=False, na=False))
        ]
        print(f"🧹 Insights after filtering low-quality: {len(df)}")

        # Deduplicate
        for key in state["deduplication_keys"]:
            if key in df.columns:
                df[key] = df[key].astype(str).str.lower()
        df = df.drop_duplicates(subset=state["deduplication_keys"], keep="last")
        print(f"🔄 Insights after deduplication: {len(df)}")

        # Remove raw_content before saving
        if "raw_content" in df.columns:
            df = df.drop(columns=["raw_content"])

        # Save results
        report = {
            "report_date": datetime.now().isoformat() + "Z",
            "keywords": df["keywords"].explode().unique().tolist() if "keywords" in df.columns else [],
            "insights": df.to_dict(orient="records")
        }
        os.makedirs(os.path.dirname(state["output_path"]), exist_ok=True)
        with open(state["output_path"], "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4, ensure_ascii=False)
        print(f"💾 Report saved to: {state['output_path']}")
        return state

    def _clean_string(self, text: Optional[str]) -> Optional[str]:
        """Clean a string by normalizing whitespace and removing invalid characters."""
        if text is None:
            return None
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[^\w\s.,\-@:/]', '', text)
        return text if text else None

    def _normalize_url(self, url: Optional[str]) -> Optional[str]:
        """Normalize a URL by ensuring it has a protocol and is valid."""
        if url is None:
            return None
        url = url.strip()
        if not re.match(r'^[a-zA-Z]+://', url):
            url = 'https://' + url
        if re.match(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', url):
            return url
        return None

    def build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow for cleaning data."""
        workflow = StateGraph(CleanerState)
        workflow.add_node("load_data", self.load_data)
        workflow.add_node("extract_entities_and_dates", self.extract_entities_and_dates)
        workflow.add_node("summarize_with_llm", self.summarize_with_llm)
        workflow.add_node("deduplicate_and_save", self.deduplicate_and_save)
        workflow.add_edge("load_data", "extract_entities_and_dates")
        workflow.add_edge("extract_entities_and_dates", "summarize_with_llm")
        workflow.add_edge("summarize_with_llm", "deduplicate_and_save")
        workflow.add_edge("deduplicate_and_save", END)
        workflow.set_entry_point("load_data")
        return workflow.compile()

    def run(self, porter_force: str):
        """Run the cleaning workflow for a given Porter force."""
        state = {
            "raw_data": [],
            "processed_insights": [],
            "porter_force": porter_force,
            "output_path": f"data/cleaned/{porter_force}_cleaned.json",
            "deduplication_keys": ["url", "source_name"]
        }
        workflow = self.build_workflow()
        final_state = workflow.invoke(state)
        return final_state
    


def main(porter_force: str):
    """Execute the cleaner workflow for a given Porter force."""
    print(f"🚀 Processing crawler output for Porter force: {porter_force}")
    agent = CleanerAgent()
    final_state = agent.run(porter_force)
    print(f"\n🎉 Cleaning completed successfully for {porter_force}!")
    print(f"📊 Total insights processed: {len(final_state['processed_insights'])}")

if __name__ == "__main__":
    # Example usage with the Porter force from the crawler
    main(porter_force="Rivalry")