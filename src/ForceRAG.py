import faiss
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import JSONLoader
from langchain_community.docstore.in_memory import InMemoryDocstore
from dotenv import load_dotenv
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
import os
import json
import re
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import traceback

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


class IndexProcessingError(Exception):
    """Raised when FAISS index processing fails."""
    pass

class RetrievalError(Exception):
    """Raised when configuring document to the vector database."""
    pass


def get_load_document(file_path):
    loader = JSONLoader(file_path, jq_schema=".[]", content_key="raw_content")
    pages = []
    for page in loader.lazy_load():
        pages.append(page)

    return pages


def split_document(pages):
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators = ["\n\n", "."],
        is_separator_regex="\n\n",
    )
    chunks = text_splitter.split_documents(pages)

    return chunks


def get_embedding_model():
    model_name = "BAAI/bge-small-en-v1.5"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}

    hf_embedding = HuggingFaceBgeEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )

    return hf_embedding


def store_to_vectorstore(index_name, chunks, hf_embedding, index_path="./index", ):
    # If index doesn't exist then create the index
    try:
        if not (os.path.exists(os.path.join(index_path, f"{index_name}.faiss")) and os.path.exists(os.path.join(index_path, f"{index_name}.pkl"))):
            index = faiss.IndexFlatL2(len(hf_embedding.embed_query("hello world")))
            vector_store = FAISS(
                embedding_function=hf_embedding,
                index=index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
            )
        else:
            vector_store = FAISS.load_local(folder_path=index_path, embeddings=hf_embedding, index_name=index_name, allow_dangerous_deserialization=True)

        # Now, index is exist, it's time to insert the data to the index
        vector_store.add_documents(documents=chunks)
        vector_store.save_local(folder_path=index_path, index_name=index_name)

    except Exception as e:
        print(traceback.format_exc())
        raise IndexProcessingError(f"Exception occurred while creating index : {e}")


def get_vectorstore_as_retriever(index_name, index_path="./index"):
    try:
        embeddings = get_embedding_model()
        vector_store = FAISS.load_local(folder_path=index_path, index_name=index_name, embeddings=embeddings, allow_dangerous_deserialization=True)
        retriever = vector_store.as_retriever()

        return retriever
    except Exception as e:
        print(traceback.format_exc())
        raise RetrievalError(f"Exception occurred while configuring retriever: {e}")



def get_llm_instance():
    llm = ChatGroq(
        model="deepseek-r1-distill-llama-70b",
        temperature=0,
        max_tokens=None,
        reasoning_format="parsed",
        timeout=None,
        max_retries=2,
        api_key=GROQ_API_KEY,
        # other params...
    )

    return llm



def get_rag_chain(retriever, llm, question):
    
#     PORTER_FORCE_TEMPLATES = {
#     "Competitive Rivalry": """
#             Analyze competitive rivalry in FinTech using:
#             {context}
            
#             Extract and calculate:
#             1. Market Concentration Score (1-10) based on:
#                - Number of competitors mentioned
#                - Market share percentages
#                - Growth rate comparisons
            
#             2. Trend Analysis:
#                - Identify 3 key trends with supporting statistics
#                - Calculate trend velocity (change over time)
            
#             3. Financial Health Indicators:
#                - Profit margins mentioned
#                - Investment amounts
#                - Revenue growth rates
            
#             4. Competitive Strategy Recommendations
#             """,
    
#     "Buyer Power": """
#             Analyze the buyer power dynamics in FinTech using:
#             {context}
            
#             Extract and calculate:
#             1. Customer Concentration Score (1-10) based on:
#                - Number of major customers mentioned
#                - Switching cost indicators
#                - Price sensitivity evidence
            
#             2. Lead Scoring Model:
#                - Intent signals (frequency of buying-related terms)
#                - Engagement metrics (references to interactions)
#                - Budget indicators (dollar amounts mentioned)
            
#             3. Financial Metrics:
#                - Extract all numerical data (percentages, dollar amounts)
#                - Calculate average customer acquisition cost if possible
            
#             4. Strategic Recommendations:
#                - Customer retention strategies
#                - Pricing model adjustments
#             """,
    
#     "Supplier Power": """
#             Analyze supplier power in FinTech using:
#             {context}
            
#             Extract and calculate:
#             1. Supplier Criticality Score (1-10):
#                - Number of alternative suppliers mentioned
#                - Switching cost indicators
#                - Unique technology dependencies
            
#             2. Cost Structure Analysis:
#                - Extract all cost-related figures
#                - Calculate cost volatility if possible
            
#             3. Risk Assessment:
#                - Single points of failure
#                - Contract duration mentions
#             """,
    
#     "Threat of Substitutes": """
#             Analyze substitution threats in FinTech:
#             - Traditional finance alternatives
#             - Blockchain/crypto alternatives
#             - Emerging tech disruptions

#             Context:
#             {context}

#             Generate:
#             1. SUBSTITUTION THREAT SCORE (1-10) based on:
#                 - Relative price performance
#                 - Switching willingness
#                 - Feature parity

#             2. DISRUPTION ANALYSIS:
#                 - Technology adoption curves
#                 - Regulatory shifts
#                 - Consumer behavior changes

#             3. INNOVATION ROADMAP:
#                 - Defensive R&D priorities
#                 - Strategic partnerships
#                 - Ecosystem development

#             4. DASHBOARD METRICS:
#                 - Alternative solution growth rates
#                 - Customer preference shifts
#                 - Cross-industry threat matrix
#     """,
    
#     "Threat of New Entrants": """
#             Evaluate market entry barriers:
#             - Regulatory hurdles
#             - Capital requirements
#             - Incumbent advantages

#             Context:
#             {context}

#             Generate:
#             1. ENTRY THREAT SCORE (1-10) based on:
#                 - Regulatory complexity
#                 - Minimum efficient scale
#                 - Brand dominance

#             2. ENTRANT PROFILING:
#                 - Likely player types
#                 - Potential entry vectors
#                 - Timing projections

#             3. DEFENSIVE STRATEGIES:
#                 - Barrier strengthening
#                 - Early acquisition targets
#                 - Regulatory engagement

#             4. DASHBOARD METRICS:
#                 - New FinTech licenses issued
#                 - Venture capital inflows 
#                 - Patent activity trends
#     """
# }
    PORTER_FORCE_TEMPLATES = {
    "Competitive Rivalry": """
            Conduct a comprehensive analysis of competitive rivalry within the FinTech utilizing the provided context:
            {context}
            
            Extract, calculate, and elaborate on the following elements to generate a detailed report suitable for strategic decision-making and dashboard visualization:
            
            1. Market Concentration Score (1-10): 
               - Compute based on the number of competitors referenced, their implied market shares (derive from mentions of dominance, funding, or user base), and concentration ratios (e.g., CR4 if data allows).
               - Provide supporting evidence from the context, including quantitative estimates where possible, and explain the scoring rationale with references to key data points.
            
            2. Trend Analysis:
               - Identify and describe at least 5 key emerging trends, supported by specific examples, statistics, or qualitative insights from the context.
               - Assess trend velocity by evaluating mentions of growth rates, adoption timelines, or market shifts; include projected impacts on SMBs over the next 12-24 months.
               - Suggest visualization formats, such as line charts for trend progression or heat maps for impact assessment.
            
            3. Financial Health Indicators:
               - Extract and analyze all mentioned financial metrics, including profit margins, investment/funding amounts, revenue growth rates, and valuation estimates.
               - Calculate comparative benchmarks (e.g., average growth rate across competitors) and highlight outliers or risks.
               - Recommend dashboard metrics like bar charts for revenue comparisons or pie charts for funding distribution.
            
            4. Competitive Strategy Recommendations:
               - Propose 4-6 actionable strategies for differentiation, such as partnerships, feature innovations, or pricing adjustments, grounded in the analyzed data.
               - Include potential ROI estimates or risk assessments where feasible, and align with SMB resource constraints.
               - Suggest visualization aids, such as SWOT matrices or strategy roadmaps.
            """,
    
    "Buyer Power": """
            Perform an in-depth evaluation of buyer power dynamics in FinTech based on:
            {context}
            
            Extract, quantify, and expand upon these components to produce an extensive report optimized for dashboard integration and strategic insights:
            
            1. Customer Concentration Score (1-10): 
               - Determine using the count of major customers or buyer segments noted, indicators of switching costs (e.g., integration complexities), and evidence of price sensitivity (e.g., negotiation power mentions).
               - Substantiate the score with detailed context excerpts, numerical derivations, and a breakdown of influencing factors.
            
            2. Lead Scoring Model:
               - Develop a model incorporating intent signals (e.g., frequency of purchase-related terms), engagement metrics (e.g., interaction counts or sentiment analysis), and budget indicators (e.g., explicit dollar values or implied spending capacity).
               - Assign scores to potential leads and provide aggregated statistics, such as average lead quality.
               - Recommend dashboard elements like funnel visualizations or heat maps for lead distribution.
            
            3. Financial Metrics:
               - Compile all numerical data related to costs, including customer acquisition costs (CAC), lifetime value (LTV), and pricing structures; compute ratios like LTV:CAC if data permits.
               - Analyze trends in buyer bargaining power through metrics like discount rates or contract lengths.
               - Propose visualizations such as scatter plots for CAC vs. LTV or trend lines for pricing evolution.
            
            4. Strategic Recommendations:
               - Offer 5-7 tailored strategies for enhancing customer retention and value capture, such as loyalty programs, customized offerings, or value-based pricing.
               - Include implementation steps, potential challenges, and measurable KPIs for success.
               - Suggest dashboard tracking via retention rate charts or customer satisfaction gauges.
            """,
    
    "Supplier Power": """
            Undertake a thorough assessment of supplier power in FinTech employing:
            {context}
            
            Extract, compute, and detail the following aspects to deliver a robust report primed for dashboard analytics and operational strategies:
            
            1. Supplier Criticality Score (1-10):
               - Evaluate based on the availability of alternative suppliers, switching cost implications (e.g., migration efforts), and dependencies on unique technologies or services.
               - Support the score with comprehensive evidence, including supplier counts and qualitative dependency descriptions.
            
            2. Cost Structure Analysis:
               - Gather all cost-associated figures, such as pricing models, contract values, or expense ratios; calculate potential volatility using variance in mentioned costs.
               - Identify cost drivers and their impact on SMB operations.
               - Recommend visualizations like cost breakdown pies or volatility trend graphs.
            
            3. Risk Assessment:
               - Pinpoint single points of failure, supply chain vulnerabilities, and contract-related risks (e.g., duration, exclusivity).
               - Quantify risks where possible (e.g., probability scores) and prioritize based on severity.
               - Propose risk matrices or heat maps for dashboard representation.
            
            4. Strategic Recommendations:
               - Formulate 4-6 mitigation strategies, including diversification, long-term contracts, or vertical integration.
               - Detail benefits, costs, and timelines, aligned with SMB capabilities.
               - Suggest monitoring via supplier performance dashboards or risk alert systems.
            """,
    
    "Threat of Substitutes": """
            Execute a detailed examination of substitution threats in FinTech, focusing on:
            {context}
            
            Generate an expanded report by extracting, calculating, and elaborating on:
            
            1. Substitution Threat Score (1-10): 
               - Base on relative price-performance ratios, buyer switching willingness (e.g., ease of adoption), and feature parity between alternatives.
               - Provide in-depth justification with context-specific examples and derived metrics.
            
            2. Disruption Analysis:
               - Outline at least 5 disruption vectors, including technology adoption curves, regulatory changes, and shifts in consumer behavior.
               - Evaluate disruption potential with timelines and impact forecasts.
               - Recommend line graphs for adoption curves or scenario trees for disruptions.
            
            3. Innovation Roadmap:
               - Develop defensive R&D priorities, strategic partnerships, and ecosystem enhancements.
               - Include phased plans with milestones and resource allocations.
               - Suggest roadmap timelines or Gantt charts for visualization.
            
            4. Dashboard Metrics:
               - Track alternative growth rates, preference shifts, and cross-industry threats via extracted metrics.
               - Propose composite indices or comparative bar charts.
            """,
    
    "Threat of New Entrants": """
            Conduct an exhaustive review of entry threats in FinTech, emphasizing:
            {context}
            
            Produce a comprehensive report through extraction, computation, and analysis of:
            
            1. Entry Threat Score (1-10): 
               - Derived from regulatory complexities, capital requirements, and incumbent advantages like brand strength.
               - Elaborate with supporting data points and calculation logic.
            
            2. Entrant Profiling:
               - Characterize potential entrants by type, entry methods, and projected timelines.
               - Include profiles with strengths, weaknesses, and market fit.
               - Recommend entity diagrams or timelines for dashboard display.
            
            3. Defensive Strategies:
               - Recommend 5-7 approaches to fortify barriers, such as regulatory advocacy or preemptive acquisitions.
               - Detail execution plans, expected outcomes, and monitoring metrics.
               - Suggest strategy efficacy scorecards or progress trackers.
            
            4. Dashboard Metrics:
               - Monitor new licenses, VC inflows, and patent trends with extracted numerical data.
               - Propose trend dashboards with alerts for threshold breaches.
            """
    }

    def get_force_template(force_name: str, context: str) -> str:
        template = PORTER_FORCE_TEMPLATES.get(force_name.split("Force")[0].strip())
        if not template:
            raise ValueError(f"Unknown Porter Force: {force_name}")
        return template.format(context=context)
    
    template = get_force_template("Buyer Power Force", "{context}")

    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever,  "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


    return rag_chain


def configure_document(document_path, index_name, index_path):
    try:
        pages = get_load_document(document_path)
        chunks = split_document(pages)
        hf_embedding_model = get_embedding_model()
        store_to_vectorstore(chunks=chunks, hf_embedding=hf_embedding_model, index_name=index_name, index_path=index_path)
        return {"status": True, "message": "Document ingested successfully, now you can perform the QA over the document."}

    except Exception as e:
        print(f"Failed to configure the document: {e}")
        print(traceback.format_exc())
        return {"status": False, "message": f"Failed to configure the document: {e}"}



def get_response(index_name, index_path, query):
    try:
        retriever = get_vectorstore_as_retriever(index_name=index_name, index_path=index_path)
        llm = get_llm_instance()
        rag_chain = get_rag_chain(retriever, llm, query)
        response = rag_chain.invoke(query)

        return {"status": True, "answer":response, "message":"successfully generated the response for the query."}

    except Exception as e:
        print(f"Exception occurred : {e}")
        print(traceback.format_exc())
        return {"status": False, "message":f"Exception occurred : {e}"}

def save_report(report: Dict, file_path: str):
    try:
        with open(file_path, 'w') as f:
            json.dump(report, f, indent=4)
        return {"status": True, "message": "Report saved successfully."}
    except Exception as e:
        print(f"Failed to save report: {e}")
        return {"status": False, "message": f"Failed to save report: {e}"}


if __name__ == "__main__":
    document_path = "data/raw/Buyer Power_crawl.json"
    pages = get_load_document(document_path)
    chunks = split_document(pages)
    hf_embedding_model = get_embedding_model()

    index_name = "document-index"
    index_path = "./index"
    store_to_vectorstore(chunks=chunks, hf_embedding=hf_embedding_model,index_name=index_name, index_path=index_path )
    retriever = get_vectorstore_as_retriever(index_name=index_name, index_path=index_path)
    llm = get_llm_instance()
    query = "Analyze this competitive intelligence data using Porter's Buyer's Power Force"
    rag_chain = get_rag_chain(retriever, llm, query)
    response = rag_chain.invoke(query)
    save_report_response = save_report(response, "data/report/report.json")
    if save_report_response["status"]:
        print("Report saved successfully.")
    else:
        print("Failed to save report:", save_report_response["message"])

    print(response)
