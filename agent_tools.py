"""
AI Agent Tools Module
These are the "skills" the agent can use to accomplish tasks
"""

import requests
from datetime import datetime
import json
from typing import Dict, Any, List
import os
from serpapi import GoogleSearch

class AgentTools:
    """Collection of tools the AI agent can use"""
    
    def __init__(self, vector_store=None):
        self.vector_store = vector_store
        self.tool_history = []  # Track what tools were used
    
    # ═══════════════════════════════════════════════════════════
    # TOOL 1: Web Search (Simulated - explains limitation)
    # ═══════════════════════════════════════════════════════════

    def web_search(self, query: str) -> Dict[str, Any]:
        try:
            # 🔹 Step 1: Prepare search parameters
            params = {
                "q": query,   # what user wants to search
                "api_key": os.getenv("SERPAPI_KEY"),  # get API key from environment
                "engine": "google",   # use Google search
                "num": 5   # number of results to fetch
            }

            # 🔹 Step 2: Call SerpAPI
            search = GoogleSearch(params)
            results = search.get_dict()  
            # 👉 This sends request to Google via SerpAPI and returns JSON data

            # 🔹 Step 3: Extract only organic (normal) search results
            organic_results = results.get("organic_results", [])

            # 🔹 Step 4: Prepare clean output
            formatted_results = []   # will store cleaned results
            related = []             # store titles for suggestions

            # 🔹 Step 5: Loop through results
            for r in organic_results[:5]:
                title = r.get("title", "")      # page title
                link = r.get("link", "")        # URL
                snippet = r.get("snippet", "")  # short description

                # 👉 Save structured result
                formatted_results.append({
                    "title": title,
                    "link": link,
                    "snippet": snippet
                })

                # 👉 Save titles as related topics
                related.append(title)

            # 🔹 Step 6: Create final output (important for your agent)
            output = {
                "success": True,  # indicates success
                "query": query,   # original user query
                
                # 👉 combine all snippets into one text (used by Groq)
                "abstract": "\n\n".join([r["snippet"] for r in formatted_results]),
                
                "results": formatted_results,  # full structured results
                
                "related": related,  # list of related topics
                
                # 👉 first link as main URL (optional)
                "url": formatted_results[0]["link"] if formatted_results else "",
                
                "demo_mode": False   # now it's real search (not demo)
            }

            # 🔹 Step 7: Log tool usage (for debugging/history)
            self._log_tool_use("web_search", query, output)

            # 🔹 Step 8: Return result
            return output

        except Exception as e:
            # ❌ If anything fails (API error, key missing, etc.)
            return {
                "success": False,
                "error": str(e),
                "demo_mode": False
            }
        
    # ═══════════════════════════════════════════════════════════
    # TOOL 2: Advanced Calculator
    # ═══════════════════════════════════════════════════════════
    def calculator(self, expression: str) -> Dict[str, Any]:
        """
        Perform mathematical calculations
        
        Args:
            expression: Math expression to evaluate
            
        Returns:
            Result of calculation
        """
        try:
            # Security: only allow safe characters
            allowed = set('0123456789+-*/().% ')
            if not all(c in allowed for c in expression):
                return {"success": False, "error": "Invalid characters in expression"}
            
            result = eval(expression)
            
            output = {
                "success": True,
                "expression": expression,
                "result": result
            }
            
            self._log_tool_use("calculator", expression, output)
            return output
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # ═══════════════════════════════════════════════════════════
    # TOOL 3: RAG Document Search
    # ═══════════════════════════════════════════════════════════
    def search_documents(self, query: str, n_results: int = 3) -> Dict[str, Any]:
        """
        Search uploaded documents using RAG
        
        Args:
            query: What to search for in documents
            n_results: Number of relevant chunks to return
            
        Returns:
            Relevant document chunks
        """
        try:
            if not self.vector_store:
                return {"success": False, "error": "No documents uploaded"}
            
            # Check if any documents exist
            stats = self.vector_store.get_stats()
            if stats['total_chunks'] == 0:
                return {"success": False, "error": "No documents in database"}
            
            # Search
            results = self.vector_store.search(query, n_results=n_results)
            
            if not results:
                return {"success": False, "error": "No relevant documents found"}
            
            # Format results
            chunks = []
            sources = set()
            for r in results:
                chunks.append({
                    "text": r['text'],
                    "source": r['metadata'].get('source', 'unknown'),
                    "relevance": 1 - r['distance']  # Convert distance to relevance score
                })
                sources.add(r['metadata'].get('source', 'unknown'))
            
            output = {
                "success": True,
                "query": query,
                "chunks": chunks,
                "sources": list(sources),
                "total_found": len(chunks)
            }
            
            self._log_tool_use("search_documents", query, output)
            return output
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # ═══════════════════════════════════════════════════════════
    # TOOL 4: Get Current Time/Date
    # ═══════════════════════════════════════════════════════════
    def get_current_time(self) -> Dict[str, Any]:
        """
        Get current date and time
        
        Returns:
            Current datetime information
        """
        now = datetime.now()
        
        output = {
            "success": True,
            "timestamp": now.isoformat(),
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "day": now.strftime("%A"),
            "formatted": now.strftime("%A, %B %d, %Y at %I:%M %p")
        }
        
        self._log_tool_use("get_current_time", None, output)
        return output
    
    # ═══════════════════════════════════════════════════════════
    # TOOL 5: File System Operations
    # ═══════════════════════════════════════════════════════════
    def list_uploaded_files(self) -> Dict[str, Any]:
        """
        List all uploaded documents
        
        Returns:
            List of uploaded files with metadata
        """
        try:
            upload_dir = "uploaded_files"
            
            if not os.path.exists(upload_dir):
                return {"success": True, "files": [], "total": 0}
            
            files = []
            for filename in os.listdir(upload_dir):
                filepath = os.path.join(upload_dir, filename)
                stat = os.stat(filepath)
                
                files.append({
                    "name": filename,
                    "size_bytes": stat.st_size,
                    "size_kb": round(stat.st_size / 1024, 2),
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
            
            output = {
                "success": True,
                "files": files,
                "total": len(files)
            }
            
            self._log_tool_use("list_uploaded_files", None, output)
            return output
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # ═══════════════════════════════════════════════════════════
    # TOOL 6: Database Statistics
    # ═══════════════════════════════════════════════════════════
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the RAG database
        
        Returns:
            Database statistics
        """
        try:
            if not self.vector_store:
                return {"success": False, "error": "Vector store not initialized"}
            
            stats = self.vector_store.get_stats()
            files = self.list_uploaded_files()
            
            output = {
                "success": True,
                "total_chunks": stats['total_chunks'],
                "total_files": files.get('total', 0),
                "avg_chunks_per_file": round(stats['total_chunks'] / max(files.get('total', 1), 1), 2),
                "files": files.get('files', [])
            }
            
            self._log_tool_use("get_database_stats", None, output)
            return output
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # ═══════════════════════════════════════════════════════════
    # TOOL REGISTRY - Descriptions for the Agent
    # ═══════════════════════════════════════════════════════════
    def get_available_tools(self) -> List[Dict[str, str]]:
        """
        Return list of available tools with descriptions
        This helps the agent know what it can do
        """
        return [
            {
                "name": "web_search",
                "description": "Search the internet for current information, news, facts, or general knowledge",
                "parameters": "query (string): what to search for",
                "example": "web_search('latest AI news')"
            },
            {
                "name": "calculator",
                "description": "Perform mathematical calculations, arithmetic, equations",
                "parameters": "expression (string): math expression to evaluate",
                "example": "calculator('25 * 4 + 10')"
            },
            {
                "name": "search_documents",
                "description": "Search through uploaded PDF/TXT documents for specific information",
                "parameters": "query (string): what to search for, n_results (int, optional): number of results",
                "example": "search_documents('vacation policy')"
            },
            {
                "name": "get_current_time",
                "description": "Get the current date, time, day of week",
                "parameters": "none",
                "example": "get_current_time()"
            },
            {
                "name": "list_uploaded_files",
                "description": "List all documents that have been uploaded to the system",
                "parameters": "none",
                "example": "list_uploaded_files()"
            },
            {
                "name": "get_database_stats",
                "description": "Get statistics about the document database (how many files, chunks, etc)",
                "parameters": "none",
                "example": "get_database_stats()"
            }
        ]
    
    # ═══════════════════════════════════════════════════════════
    # HELPER: Logging
    # ═══════════════════════════════════════════════════════════
    def _log_tool_use(self, tool_name: str, input_data: Any, output_data: Any):
        """Track which tools were used"""
        self.tool_history.append({
            "tool": tool_name,
            "input": input_data,
            "output": output_data,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_tool_history(self) -> List[Dict]:
        """Get history of tool usage"""
        return self.tool_history
    
    def clear_tool_history(self):
        """Clear the tool usage history"""
        self.tool_history = []


# ═══════════════════════════════════════════════════════════
# Tool Executor - Calls the right tool based on agent's decision
# ═══════════════════════════════════════════════════════════
def execute_tool(tools: AgentTools, tool_name: str, **kwargs) -> Dict[str, Any]:
    """
    Execute a tool by name
    
    Args:
        tools: AgentTools instance
        tool_name: Name of the tool to execute
        **kwargs: Arguments to pass to the tool
        
    Returns:
        Tool execution result
    """
    tool_map = {
        "web_search": lambda: tools.web_search(kwargs.get("query", "")),
        "calculator": lambda: tools.calculator(kwargs.get("expression", "")),
        "search_documents": lambda: tools.search_documents(
            kwargs.get("query", ""),
            kwargs.get("n_results", 3)
        ),
        "get_current_time": lambda: tools.get_current_time(),
        "list_uploaded_files": lambda: tools.list_uploaded_files(),
        "get_database_stats": lambda: tools.get_database_stats()
    }
    
    if tool_name in tool_map:
        return tool_map[tool_name]()
    else:
        return {"success": False, "error": f"Unknown tool: {tool_name}"}