# app.py
import os
import asyncio
import json
from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from src.pipeline import run_pipeline, verif_node, PipelineState
from src.PortalyzeBot import PortalyzeBot
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

app = FastAPI(title="Fintech Analysis Platform")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or ["*"] for all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# bot = PortalyzeBot()

@app.post("/verify_idea")
async def verify_idea(data: dict = Body(...)):
    """Takes initial idea, runs VerifAgent, returns enhanced idea."""
    state = PipelineState(
        initial_idea=data["idea"],
        enhanced_idea="",
        per_force_reports=[],
        merged_report="",
        verdict={},
        dashboard_analysis={}
    )
    verified_state = await verif_node(state)
    return {"enhanced_idea": verified_state["enhanced_idea"]}

@app.post("/run_pipeline")
async def run_full_pipeline(data: dict = Body(...)):
    """Runs the entire pipeline and returns dashboard.json data."""
    final_state = await run_pipeline(data["idea"])
    return JSONResponse(content=final_state["dashboard_analysis"])

@app.get("/dashboard_data")
async def get_dashboard_data():
    """Fetch dashboard_analysis.json for chart rendering."""
    try:
        with open("outputs/dashboard_analysis.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        return JSONResponse(content=data)
    except FileNotFoundError:
        return JSONResponse(content={"error": "No dashboard data found"}, status_code=404)
    
@app.get("/idea_report")
async def get_idea_report():
    """Fetches the enhanced idea report."""
    try:
        with open("idea/idea_report.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        return JSONResponse(content=data)
    except FileNotFoundError:
        return JSONResponse(content={"error": "No idea report found"}, status_code=404)

# @app.post("/chatbot")
# async def chatbot_query(data: dict = Body(...)):
#     """RAG chatbot over merged_report.txt."""
#     answer = bot.ask(data["question"], data.get("chat_history", []))
#     return {"answer": answer}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8090, reload=True)
