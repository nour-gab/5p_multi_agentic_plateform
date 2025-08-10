"""
Module for querying Tavily and MCP APIs for relevant sources.
"""

import os
from dotenv import load_dotenv

load_dotenv()

def query_sources(porter_force: str, keywords: str):
    """
    Query Tavily and MCP APIs for relevant sources.
    Returns a list of dicts: [{'url': ..., 'name': ...}, ...]
    """
    sources = []

    # Tavily search (broad web search)
    try:
        from tavily import TavilyClient
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        tavily_client = TavilyClient(api_key=tavily_api_key)
        tavily_results = tavily_client.search(f"{porter_force} {keywords}", max_results=50)
        for res in tavily_results['results']:
            sources.append({'url': res['url'], 'name': res.get('title', 'Tavily Result')})
    except Exception as e:
        print(f"Tavily search failed: {e}")

    # MCP search (targeted APIs)
    # try:
    #     from mcp import MCPClient
    #     mcp_api_key = os.getenv("MCP_API_KEY")
    #     mcp_client = MCPClient(api_key=mcp_api_key)
    #     apis = porter_force_apis.get(porter_force, [])
    #     for api in apis:
    #         try:
    #             mcp_results = mcp_client.search(f"{keywords}", api=api, limit=3)
    #             for res in mcp_results.get('results', []):
    #                 sources.append({'url': res['url'], 'name': res.get('title', f'{api} Result')})
    #         except Exception as api_e:
    #             print(f"MCP search failed for {api}: {api_e}")
    # except Exception as e:
    #     print(f"MCP search failed: {e}")

    return sources