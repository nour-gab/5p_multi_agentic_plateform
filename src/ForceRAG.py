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
from langchain_core.documents import Document

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


class IndexProcessingError(Exception):
    """Raised when FAISS index processing fails."""
    pass

class RetrievalError(Exception):
    """Raised when configuring document to the vector database."""
    pass


def get_load_document(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    documents = []
    for insight in data.get("insights", []):
        content = (
            insight.get("content_summary", "") + "\n\n"
            + "Entities: " + json.dumps(insight.get("entities", {})) + "\n\n"
            + "Dates: " + json.dumps(insight.get("dates", [])) + "\n\n"
            + "Keywords: " + json.dumps(insight.get("keywords", []))
        )
        metadata = {
            "source_name": insight.get("source_name"),
            "url": insight.get("url"),
            "timestamp": insight.get("timestamp"),
            "porter_force": insight.get("porter_force"),
            "id": insight.get("id")
        }
        documents.append(Document(page_content=content, metadata=metadata))
    
    return documents


def split_document(pages):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "."],
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


def store_to_vectorstore(index_name, chunks, hf_embedding, index_path="./index"):
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
    )

    return llm


def get_rag_chain(retriever, llm, question, porter_force):
    
    PORTER_FORCE_TEMPLATES = {
    "Competitive Rivalry": """
            Analyze competitive rivalry in FinTech using:
            {context}
            
            Extract and calculate:
            1. Market Concentration Score (1-10) based on:
               - Number of competitors mentioned
               - Market share percentages
               - Growth rate comparisons
            
            2. Trend Analysis:
               - Identify 3 key trends with supporting statistics
               - Calculate trend velocity (change over time)
            
            3. Financial Health Indicators:
               - Profit margins mentioned
               - Investment amounts
               - Revenue growth rates
            
            4. Competitive Strategy Recommendations
            """,
    
    "Buyer Power": """
            Analyze the buyer power dynamics in FinTech using:
            {context}
            
            Extract and calculate:
            1. Customer Concentration Score (1-10) based on:
               - Number of major customers mentioned
               - Switching cost indicators
               - Price sensitivity evidence
            
            2. Lead Scoring Model:
               - Intent signals (frequency of buying-related terms)
               - Engagement metrics (references to interactions)
               - Budget indicators (dollar amounts mentioned)
            
            3. Financial Metrics:
               - Extract all numerical data (percentages, dollar amounts)
               - Calculate average customer acquisition cost if possible
            
            4. Strategic Recommendations:
               - Customer retention strategies
               - Pricing model adjustments
            """,
    
    "Supplier Power": """
            Analyze supplier power in FinTech using:
            {context}
            
            Extract and calculate:
            1. Supplier Criticality Score (1-10):
               - Number of alternative suppliers mentioned
               - Switching cost indicators
               - Unique technology dependencies
            
            2. Cost Structure Analysis:
               - Extract all cost-related figures
               - Calculate cost volatility if possible
            
            3. Risk Assessment:
               - Identify 4-6 supply chain risks with probability estimates
               - Compute risk impact scores
            
            4. Mitigation Strategies:
               - Recommend diversification approaches
               - Contract negotiation tactics
               - Alternative sourcing options
            """,
    
    "Threat of New Entrants": """
            Execute an exhaustive review of entry threats in FinTech, emphasizing:
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
            """
    }

    mapping = {
        "Rivalry": "Competitive Rivalry",
        "Industry Rivalry": "Competitive Rivalry",
        "Buyer Power": "Buyer Power",
        "Supplier Power": "Supplier Power",
        "New Entrants": "Threat of New Entrants",
        "Substitute Products": "Threat of Substitutes"
    }

    def get_force_template(force_name: str, context: str) -> str:
        key = mapping.get(force_name, force_name)
        template = PORTER_FORCE_TEMPLATES.get(key)
        if not template:
            raise ValueError(f"Unknown Porter Force: {force_name}")
        return template.format(context=context)
    
    template = get_force_template(porter_force, "{context}")

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


def get_response(index_name, index_path, query, porter_force):
    try:
        retriever = get_vectorstore_as_retriever(index_name=index_name, index_path=index_path)
        llm = get_llm_instance()
        rag_chain = get_rag_chain(retriever, llm, query, porter_force)
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


def run():
    porter_force = "Buyer Power"  # Example; can be parameterized
    document_path = f"data/cleaned/{porter_force}_cleaned.json"
    
    # Load data to extract porter_force if needed, but since uniform, use the input
    with open(document_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    actual_porter_force = data["insights"][0]["porter_force"] if data["insights"] else porter_force
    
    pages = get_load_document(document_path)
    chunks = split_document(pages)
    hf_embedding_model = get_embedding_model()

    index_name = "document-index"
    index_path = "./index"
    store_to_vectorstore(chunks=chunks, hf_embedding=hf_embedding_model, index_name=index_name, index_path=index_path)
    retriever = get_vectorstore_as_retriever(index_name=index_name, index_path=index_path)
    llm = get_llm_instance()
    query = f"Analyze this competitive intelligence data using Porter's {actual_porter_force} Force"
    rag_chain = get_rag_chain(retriever, llm, query, actual_porter_force)
    response = rag_chain.invoke(query)
    save_report_response = save_report({"analysis": response}, "data/report/report.json")
    if save_report_response["status"]:
        print("Report saved successfully.")
    else:
        print("Failed to save report:", save_report_response["message"])

    print(response)

    
if __name__ == "__main__":
    run()