"""
AI Agent - The Brain
Decides which tools to use and orchestrates multi-step tasks
"""

import json
import re
from typing import Dict, List, Any, Optional
from agent_tools import AgentTools, execute_tool


class AIAgent:
    """
    AI Agent that can use tools to accomplish tasks
    
    The agent:
    1. Analyzes the user's request
    2. Decides which tools are needed
    3. Executes tools in sequence
    4. Combines results into a coherent response
    """
    
    def __init__(self, llm_function, tools: AgentTools, max_iterations: int = 5):
        """
        Args:
            llm_function: Function to call the LLM (phi3)
            tools: AgentTools instance
            max_iterations: Max number of tool-use cycles before stopping
        """
        self.llm = llm_function
        self.tools = tools
        self.max_iterations = max_iterations
        self.thinking_steps = []  # Track the agent's reasoning
    
    def run(self, user_query: str, verbose: bool = False) -> Dict[str, Any]:
        """
        Main agent loop
        
        Args:
            user_query: What the user asked
            verbose: If True, return detailed thinking steps
            
        Returns:
            {
                "response": final answer to user,
                "tools_used": list of tools that were called,
                "thinking": agent's reasoning process (if verbose)
            }
        """
        self.thinking_steps = []
        self.tools.clear_tool_history()
        
        # Step 1: Analyze query and make a plan
        plan = self._create_plan(user_query)
        self._log_thinking("PLAN", plan)
        
        # Step 2: Execute the plan
        results = self._execute_plan(plan, user_query)
        
        # Step 3: Synthesize final answer
        final_response = self._synthesize_response(user_query, results)
        
        output = {
            "response": final_response,
            "tools_used": [step["tool"] for step in self.tools.get_tool_history()],
            "tool_results": self.tools.get_tool_history()
        }
        
        if verbose:
            output["thinking"] = self.thinking_steps
        
        return output
    
    def _create_plan(self, query: str) -> List[str]:
        """
        Decide which tools are needed for this query
        
        Returns:
            List of tool names to use
        """
        # Build a prompt that helps the agent decide what tools it needs
        available_tools = self.tools.get_available_tools()
        tools_description = "\n".join([
            f"- {tool['name']}: {tool['description']}"
            for tool in available_tools
        ])
        
        planning_prompt = f"""You are an AI agent that can use tools to help users.

Available tools:
{tools_description}

User query: "{query}"

Analyze the query and decide which tools you need. Respond with ONLY a JSON list of tool names.

Examples:
Query: "What's 2+2 and the weather in Paris?"
Response: ["calculator", "web_search"]

Query: "Search my documents for vacation policy"
Response: ["search_documents"]

Query: "What time is it?"
Response: ["get_current_time"]

Query: "List my files and search them for 'meeting notes'"
Response: ["list_uploaded_files", "search_documents"]

Now analyze: "{query}"
Response (JSON list only):"""
        
        # 🔹 Ask LLM what tools to use
        llm_response = self.llm(planning_prompt)

        try:
            # 🔹 Extract JSON from LLM response
            json_match = re.search(r'\[.*?\]', llm_response, re.DOTALL)

            if json_match:
                tools_needed = json.loads(json_match.group())

                # ✅ Ensure it's a list
                if not isinstance(tools_needed, list):
                    tools_needed = []
            else:
                tools_needed = []

        except Exception as e:
            print("Planning error:", e)
            tools_needed = []

        # 🔥 FORCE web search for real-time queries
        query_lower = query.lower()
        if any(word in query_lower for word in ["latest", "news", "today", "current"]):
            if "web_search" not in tools_needed:
                tools_needed.append("web_search")

        print("🔥 TOOLS SELECTED:", tools_needed)

        return tools_needed
                                
    def _fallback_plan(self, query: str) -> List[str]:
        """Backup plan if LLM planning fails - use keyword matching"""
        query_lower = query.lower()
        tools = []
        
        # Calculator keywords
        if any(word in query_lower for word in ['calc', 'calculate', '*', '+', '-', '/', 'math']):
            tools.append("calculator")
        
        # Time keywords
        if any(word in query_lower for word in ['time', 'date', 'today', 'now', 'when']):
            tools.append("get_current_time")
        
        # Document search keywords
        if any(word in query_lower for word in ['document', 'file', 'pdf', 'search', 'find in']):
            tools.append("search_documents")
        
        # Web search keywords
        if any(word in query_lower for word in ['search web', 'google', 'look up', 'latest', 'news', 'current']):
            tools.append("web_search")
        
        # File list keywords
        if any(word in query_lower for word in ['list files', 'show files', 'what files', 'uploaded']):
            tools.append("list_uploaded_files")
        
        # Stats keywords
        if any(word in query_lower for word in ['stats', 'statistics', 'how many', 'database info']):
            tools.append("get_database_stats")
        
        return tools if tools else []
    
    def _execute_plan(self, plan: List[str], query: str) -> List[Dict[str, Any]]:
        """
        Execute each tool in the plan
        
        Returns:
            List of tool results
        """
        results = []
        
        for tool_name in plan:
            self._log_thinking("EXECUTING", f"Tool: {tool_name}")
            
            # Extract parameters from query for this tool
            params = self._extract_parameters(tool_name, query)
            
            # Execute the tool
            result = execute_tool(self.tools, tool_name, **params)
            results.append({
                "tool": tool_name,
                "params": params,
                "result": result
            })
            
            self._log_thinking("RESULT", f"{tool_name}: {result.get('success', False)}")
        
        return results
    
    def _extract_parameters(self, tool_name: str, query: str) -> Dict[str, Any]:
        """
        Extract parameters needed for each tool from the query
        
        This is a simple implementation - could be improved with LLM
        """
        params = {}
        
        if tool_name == "calculator":
            # Find math expression in query
            # Look for patterns like "calc 2+2" or "calculate 5*10"
            match = re.search(r'(?:calc|calculate)\s+(.+?)(?:\?|$|and|then)', query, re.IGNORECASE)
            if match:
                params["expression"] = match.group(1).strip()
            else:
                # Try to find any math expression
                math_match = re.search(r'(\d+[\s\d\+\-\*/\(\)\.]+\d+)', query)
                if math_match:
                    params["expression"] = math_match.group(1).strip()
        
        elif tool_name == "web_search":
            # Extract search query
            # Remove common prefixes
            search_query = re.sub(r'(?:search for|google|look up|find)\s+', '', query, flags=re.IGNORECASE)
            params["query"] = search_query.strip()
        
        elif tool_name == "search_documents":
            # Extract what to search for in documents
            search_query = re.sub(r'(?:search|find|look for)\s+(?:in\s+)?(?:my\s+)?(?:documents?|files?)\s+(?:for\s+)?', '', query, flags=re.IGNORECASE)
            params["query"] = search_query.strip() if search_query.strip() else query
            params["n_results"] = 3
        
        # Other tools don't need parameters
        
        return params
    
    def _synthesize_response(self, query: str, results: List[Dict[str, Any]]) -> str:
        """
        Combine all tool results into a coherent answer
        
        This is where the agent "thinks" about what it learned
        """
        if not results:
            # No tools were used - just answer normally
            return self.llm(f"Answer this question: {query}")
        
        # Build context from tool results
        context_parts = []
        
        for item in results:
            tool = item["tool"]
            result = item["result"]
            
            if not result.get("success"):
                context_parts.append(f"⚠️ {tool} failed: {result.get('error', 'unknown error')}")
                continue
            
            # Format each tool's result
            if tool == "calculator":
                context_parts.append(f"Calculation: {result['expression']} = {result['result']}")
            
            elif tool == "web_search":
                if result.get("abstract"):
                    context_parts.append(f"Web search found: {result['abstract']}")
                if result.get("related"):
                    context_parts.append(f"Related topics: {', '.join(result['related'][:2])}")
            
            elif tool == "search_documents":
                context_parts.append(f"From documents: {len(result['chunks'])} relevant sections found")
                for i, chunk in enumerate(result['chunks'][:2], 1):
                    context_parts.append(f"  {i}. {chunk['text'][:200]}...")
                context_parts.append(f"Sources: {', '.join(result['sources'])}")
            
            elif tool == "get_current_time":
                context_parts.append(f"Current time: {result['formatted']}")
            
            elif tool == "list_uploaded_files":
                if result['total'] > 0:
                    files = ", ".join([f['name'] for f in result['files']])
                    context_parts.append(f"Uploaded files ({result['total']}): {files}")
                else:
                    context_parts.append("No files uploaded yet")
            
            elif tool == "get_database_stats":
                context_parts.append(f"Database: {result['total_files']} files, {result['total_chunks']} chunks")
        
        # Combine everything
        context = "\n".join(context_parts)
        
        # Ask LLM to synthesize final answer
        synthesis_prompt = f"""Based on the information gathered, answer the user's question.

User question: "{query}"

Information gathered:
{context}

Provide a natural, conversational answer that directly addresses the question.
If multiple pieces of information were gathered, combine them logically.
Be concise but complete.

Answer:"""
        
        final_answer = self.llm(synthesis_prompt)
        
        return final_answer.strip()
    
    def _log_thinking(self, step_type: str, content: str):
        """Log agent's reasoning steps"""
        self.thinking_steps.append({
            "type": step_type,
            "content": content,
            "timestamp": self.tools.get_current_time()["timestamp"]
        })
    
    def get_thinking_process(self) -> List[Dict[str, str]]:
        """Return the agent's thought process"""
        return self.thinking_steps


# ═══════════════════════════════════════════════════════════
# Simple wrapper for testing
# ═══════════════════════════════════════════════════════════
def create_agent(llm_function, vector_store=None) -> AIAgent:
    """
    Convenience function to create an agent
    
    Args:
        llm_function: Function that calls the LLM
        vector_store: Optional vector store for RAG
        
    Returns:
        Configured AIAgent instance
    """
    tools = AgentTools(vector_store=vector_store)
    agent = AIAgent(llm_function, tools, max_iterations=5)
    return agent
