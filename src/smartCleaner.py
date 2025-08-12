"""
Cleaner module for processing DeepCrawlerAgent output using LangGraph with Llama-3.1-8B-Instant.
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
import time
import asyncio

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
    Agent for cleaning and structuring data from DeepCrawlerAgent using LangGraph with Llama-3.1-8B-Instant.
    """
    def __init__(self):
        self.porter_forces = ["Rivalry", "Buyer Power", "Supplier Power", "New Entrants", "Substitute Products"]
        self.llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2, api_key=GROQ_API_KEY)
        self.summarize_prompt = ChatPromptTemplate.from_template(
            "Summarize the following content in 100-150 words, focusing on key insights relevant to {porter_force} in the FinTech market. Highlight competitors, market trends, or strategic moves. Ensure the summary is concise, clear, and suitable for a RAG report and visualization dashboard:\n\n{content}"
        )

    def load_data(self, state: CleanerState) -> CleanerState:
        """Load crawler output from JSON file (synchronous)."""
        input_path = f"data/raw/{state['porter_force']}_crawl.json"
        try:
            with open(input_path, "r", encoding="utf-8") as f:
                state["raw_data"] = json.load(f)
        except Exception as e:
            print(f"Failed to load data from {input_path}: {e}")
            state["raw_data"] = []
        return state

    def extract_entities_and_dates(self, state: CleanerState) -> CleanerState:
        """Extract entities and dates from raw content (synchronous)."""
        processed_insights = []
        for item in state["raw_data"]:
            content = item.get("raw_content", "")
            entities = {"organizations": [], "people": [], "locations": []}
            dates = []
            if nlp and content:
                doc = nlp(content)
                entities["organizations"] = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
                entities["people"] = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
                entities["locations"] = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
                dates = [match.group() for match in re.finditer(r'\b\d{4}-\d{2}-\d{2}\b|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b', content)]
            processed_insights.append({
                "id": str(uuid.uuid4()),
                "source_name": item.get("source_name", "Unknown"),
                "url": item.get("url", ""),
                "porter_force": state["porter_force"],
                "keywords": item.get("keywords", ""),
                "content": content,
                "entities": entities,
                "dates": dates,
                "timestamp": item.get("timestamp", datetime.utcnow().isoformat() + 'Z')
            })
        state["processed_insights"] = processed_insights
        return state

    async def summarize_with_llm(self, state: CleanerState) -> CleanerState:
        """Summarize content using LLM with rate limiting."""
        async def summarize_single(insight: Dict[str, Any]) -> Dict[str, Any]:
            for attempt in range(3):  # Retry up to 3 times
                try:
                    chain = self.summarize_prompt | self.llm
                    summary = await chain.ainvoke({
                        "porter_force": state["porter_force"],
                        "content": insight["content"][:2000]  # Limit input size
                    })
                    insight["content_summary"] = summary.content
                    return insight
                except Exception as e:
                    if "Rate limit reached" in str(e):
                        wait_time = 10 * (2 ** attempt)  # Exponential backoff
                        print(f"Rate limit hit, retrying in {wait_time}s: {e}")
                        await asyncio.sleep(wait_time)
                    else:
                        print(f"Error summarizing content: {e}")
                        insight["content_summary"] = "Summary failed due to error."
                        return insight
            insight["content_summary"] = "Summary failed after retries."
            return insight

        tasks = [summarize_single(insight) for insight in state["processed_insights"]]
        state["processed_insights"] = await asyncio.gather(*tasks, return_exceptions=True)
        return state

    def deduplicate_and_save(self, state: CleanerState) -> CleanerState:
        """Deduplicate insights and save to JSON (synchronous)."""
        insights = state["processed_insights"]
        seen = set()
        deduplicated = []
        for insight in insights:
            key = tuple(insight.get(k, "") for k in state["deduplication_keys"])
            if key not in seen:
                seen.add(key)
                deduplicated.append(insight)
        print(f"🧹 Insights after filtering low-quality: {len(insights)}")
        print(f"🔄 Insights after deduplication: {len(deduplicated)}")
        state["processed_insights"] = deduplicated
        os.makedirs(os.path.dirname(state["output_path"]), exist_ok=True)
        with open(state["output_path"], "w", encoding="utf-8") as f:
            json.dump({"insights": deduplicated}, f, indent=2, ensure_ascii=False)
        print(f"💾 Report saved to: {state['output_path']}")
        return state

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

    async def run(self, porter_force: str):
        """Run the cleaning workflow for a given Porter force."""
        state = {
            "raw_data": [],
            "processed_insights": [],
            "porter_force": porter_force,
            "output_path": f"data/cleaned/{porter_force}_cleaned.json",
            "deduplication_keys": ["url", "source_name"]
        }
        workflow = self.build_workflow()
        final_state = await workflow.ainvoke(state)
        return final_state

def main(porter_force: str):
    """Execute the cleaner workflow for a given Porter force."""
    print(f"🚀 Processing crawler output for Porter force: {porter_force}")
    agent = CleanerAgent()
    final_state = asyncio.run(agent.run(porter_force))
    print(f"\n🎉 Cleaning completed successfully for {porter_force}!")
    print(f"📊 Total insights processed: {len(final_state['processed_insights'])}")

if __name__ == "__main__":
    main(porter_force="Buyer Power")