import asyncio
import os
import json
from typing import Dict, List, Any, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.pregel import Pregel
from dotenv import load_dotenv

load_dotenv()

# Import all agents and necessary components
from VerifAgent import FintechIdeaAnalyzer
from metaPrompter import MetaPromptAgent
from crawler import DeepCrawlerAgent
from smartCleaner import CleanerAgent
from ForceRAG import configure_document, get_response, save_report
from FivePAgent import FivePAgent
from JudgeLLM import LLMJudge
from dashboard_llm import DashboardAgent

# Define Porter forces
PORTER_FORCES = [
    "Rivalry",
    "Buyer Power",
    "Supplier Power",
    "New Entrants",
    "Substitute Products"
]

# Pipeline State
class PipelineState(TypedDict):
    initial_idea: str
    enhanced_idea: str
    per_force_reports: List[Dict[str, Any]]  # List of {"force": str, "report_path": str, "report": str}
    merged_report: str
    verdict: Dict[str, Any]
    dashboard_analysis: Dict[str, Any]

# Node: Verify and Enhance Idea
async def verif_node(state: PipelineState) -> PipelineState:
    analyzer = FintechIdeaAnalyzer(api_key=os.getenv("GOOGLE_API_KEY"))
    result = await asyncio.to_thread(analyzer.analyze, state["initial_idea"])
    if result["status"] != "success":
        raise ValueError(f"Idea verification failed: {result['message']}")
    with open(result["report_path"], "r", encoding="utf-8") as f:
        report = json.load(f)
    state["enhanced_idea"] = report["idea"]
    return state

# Async function to process one force
async def process_force(force: str, enhanced_idea: str) -> Dict[str, Any]:
    # MetaPrompter
    meta_agent = MetaPromptAgent()
    await asyncio.to_thread(meta_agent.generate_prompt, force)
    
    # Crawler
    crawler = DeepCrawlerAgent(porter_force=force)
    await asyncio.to_thread(crawler.run)
    
    # Cleaner
    cleaner = CleanerAgent()
    await cleaner.run(porter_force=force)
    
    # ForceRAG
    document_path = f"data/cleaned/{force}_cleaned.json"
    index_name = f"{force}-index"
    index_path = "./index"
    await asyncio.to_thread(configure_document, document_path, index_name, index_path)
    query = f"Analyze this competitive intelligence data using Porter's {force} Force"
    for attempt in range(3):  # Retry for rate limits
        try:
            response = await asyncio.to_thread(get_response, index_name, index_path, query, force)
            if not response.get("status", False):
                print(f"ForceRAG failed for {force}: {response.get('message', 'Unknown error')}")
                return {"force": force, "report_path": "", "report": f"Error: {response.get('message', 'ForceRAG failed')}"}
            if "answer" not in response:
                print(f"ForceRAG response missing 'answer' key for {force}: {response}")
                return {"force": force, "report_path": "", "report": "Error: No answer in ForceRAG response"}
            break
        except Exception as e:
            if "Rate limit reached" in str(e):
                wait_time = 10 * (2 ** attempt)
                print(f"Rate limit hit in ForceRAG for {force}, retrying in {wait_time}s: {e}")
                await asyncio.sleep(wait_time)
            else:
                print(f"ForceRAG error for {force}: {e}")
                return {"force": force, "report_path": "", "report": f"Error: {str(e)}"}
    else:
        print(f"ForceRAG failed after retries for {force} due to rate limits")
        return {"force": force, "report_path": "", "report": "Error: ForceRAG failed after retries"}
    
    report_path = f"data/report/{force}_report.json"
    await asyncio.to_thread(save_report, {"analysis": response["answer"]}, report_path)
    
    return {"force": force, "report_path": report_path, "report": response["answer"]}

# Node: Parallel Process Forces
async def parallel_forces_node(state: PipelineState) -> PipelineState:
    tasks = [process_force(force, state["enhanced_idea"]) for force in PORTER_FORCES]
    state["per_force_reports"] = await asyncio.gather(*tasks, return_exceptions=True)
    # Filter out any exceptions or invalid reports
    state["per_force_reports"] = [
        report for report in state["per_force_reports"]
        if isinstance(report, dict) and report.get("report_path")
    ]
    return state

# Node: Merge Reports with 5pAgent
async def five_p_agent_node(state: PipelineState) -> PipelineState:
    rag_data = {report["force"]: report["report"] for report in state["per_force_reports"]}
    rag_path = "data/report/report.json"
    os.makedirs(os.path.dirname(rag_path), exist_ok=True)
    with open(rag_path, "w", encoding="utf-8") as f:
        json.dump(rag_data, f, indent=2)
    
    agent = FivePAgent(
        api_key=os.getenv("GOOGLE_API_KEY"),
        rag_path=rag_path,
        output_path="data/report/merged_report.txt"
    )
    result = await asyncio.to_thread(agent.run)
    if result["status"] != "success":
        raise ValueError(f"5pAgent failed: {result['message']}")
    state["merged_report"] = result["report_text"]
    return state

# Node: Judge Merged Report
async def judge_node(state: PipelineState) -> PipelineState:
    rags = {report["force"]: report["report"] for report in state["per_force_reports"]}
    judge = LLMJudge(api_key=os.getenv("GROQ_API_KEY"))
    for attempt in range(3):  # Retry for rate limits
        try:
            verdict = await asyncio.to_thread(judge.run, state["merged_report"], rags)
            break
        except Exception as e:
            if "Rate limit reached" in str(e):
                wait_time = 10 * (2 ** attempt)
                print(f"Rate limit hit in JudgeLLM, retrying in {wait_time}s: {e}")
                await asyncio.sleep(wait_time)
            else:
                raise
    else:
        raise ValueError("JudgeLLM failed after retries due to rate limits")
    
    state["verdict"] = verdict
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/verdict.json", "w", encoding="utf-8") as f:
        json.dump(verdict, f, indent=2)
    return state

# Node: Generate Dashboard Analysis
async def dashboard_node(state: PipelineState) -> PipelineState:
    agent = DashboardAgent(api_key=os.getenv("GROQ_API_KEY"))
    for attempt in range(3):  # Retry for rate limits
        try:
            analysis = await asyncio.to_thread(agent.run, "outputs/verdict.json")
            break
        except Exception as e:
            if "Rate limit reached" in str(e):
                wait_time = 10 * (2 ** attempt)
                print(f"Rate limit hit in DashboardAgent, retrying in {wait_time}s: {e}")
                await asyncio.sleep(wait_time)
            else:
                raise
    else:
        raise ValueError("DashboardAgent failed after retries due to rate limits")
    
    state["dashboard_analysis"] = analysis
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/dashboard_analysis.json", "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)
    return state

# Build LangGraph Workflow
workflow = StateGraph(PipelineState)

workflow.add_node("verif", verif_node)
workflow.add_node("parallel_forces", parallel_forces_node)
workflow.add_node("five_p_agent", five_p_agent_node)
workflow.add_node("judge", judge_node)
workflow.add_node("dashboard", dashboard_node)

workflow.set_entry_point("verif")
workflow.add_edge("verif", "parallel_forces")
workflow.add_edge("parallel_forces", "five_p_agent")
workflow.add_edge("five_p_agent", "judge")
workflow.add_edge("five_p_agent", "dashboard")
workflow.add_edge("dashboard", END)

graph: Pregel = workflow.compile()

# Run the pipeline asynchronously
async def run_pipeline(initial_idea: str):
    initial_state = PipelineState(
        initial_idea=initial_idea,
        enhanced_idea="",
        per_force_reports=[],
        merged_report="",
        verdict={},
        dashboard_analysis={}
    )
    final_state = await graph.ainvoke(initial_state)
    return final_state

# Entry point
if __name__ == "__main__":
    initial_idea = input("Enter your fintech idea: ") or "AI-powered optimization platform for car rental fleet usage and predictive maintenance"
    final_state = asyncio.run(run_pipeline(initial_idea))
    print("Pipeline completed. Dashboard analysis:")
    print(json.dumps(final_state["dashboard_analysis"], indent=2))