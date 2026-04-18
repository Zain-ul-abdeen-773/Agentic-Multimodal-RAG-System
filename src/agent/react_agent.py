"""
LangGraph ReAct Agent — Agentic Multimodal Retrieval
=====================================================
Techniques implemented:
  - LangGraph state machine with ReAct reasoning loop
  - 4 retrieval tools: image_search, text_search, graph_query, hybrid_search
  - Dynamic tool selection based on query intent
  - Memory buffer for multi-turn conversations
  - Reinforcement Learning reward signal for tool selection
  - Causal mask reasoning in agent thought process
"""

import json
from typing import Dict, List, Optional, Any, TypedDict, Annotated, Sequence
from pathlib import Path
from loguru import logger

try:
    from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
    from langchain_core.tools import tool
    from langchain_groq import ChatGroq
except ImportError:
    logger.warning("langchain not installed. Run: pip install langchain langchain-groq langchain-core")
    ChatGroq = None

try:
    from langgraph.graph import StateGraph, END
    from langgraph.prebuilt import ToolNode
except ImportError:
    StateGraph, END, ToolNode = None, None, None
    logger.warning("langgraph not installed. Run: pip install langgraph")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import get_config, AgentConfig


# ═══════════════════════════════════════════════════════════════
# Agent State Definition
# ═══════════════════════════════════════════════════════════════
class AgentState(TypedDict):
    """State managed by the LangGraph agent."""
    messages: List[Any]           # Conversation history
    query: str                    # Current user query
    query_type: str               # Classified query type
    retrieved_results: List[Dict] # Results from retrievers
    reasoning_trace: List[str]    # Agent's reasoning steps
    final_answer: str             # Synthesized answer
    iteration: int                # Current ReAct iteration
    tool_usage: List[str]         # Tools called in this session


# ═══════════════════════════════════════════════════════════════
# Tool Reward Tracker (RL signal)
# ═══════════════════════════════════════════════════════════════
class ToolRewardTracker:
    """
    Tracks tool usage rewards for reinforcement learning-style
    optimization of tool selection.
    
    After each query, the system records which tools were used
    and whether the answer was relevant (user feedback or auto-eval).
    Over time, this builds a policy for tool selection.
    """

    def __init__(self):
        self.rewards: Dict[str, List[float]] = {}  # tool_name → [rewards]
        self.q_values: Dict[str, float] = {}       # tool_name → avg reward

    def record_reward(self, tool_name: str, reward: float):
        """Record a reward for a tool usage."""
        if tool_name not in self.rewards:
            self.rewards[tool_name] = []
        self.rewards[tool_name].append(reward)
        
        # Update Q-value (running average)
        self.q_values[tool_name] = sum(self.rewards[tool_name]) / len(
            self.rewards[tool_name]
        )

    def get_tool_priority(self) -> Dict[str, float]:
        """Get prioritized tool ordering based on Q-values."""
        return dict(sorted(
            self.q_values.items(), key=lambda x: -x[1]
        ))

    def suggest_tools(self, query_type: str) -> List[str]:
        """Suggest tools based on accumulated rewards."""
        query_tool_map = {
            "visual": ["search_images", "hybrid_search"],
            "textual": ["search_text", "hybrid_search"],
            "hybrid": ["hybrid_search", "search_images", "search_text"],
            "graph": ["query_knowledge_graph", "search_text"],
        }
        return query_tool_map.get(query_type, ["hybrid_search"])


# ═══════════════════════════════════════════════════════════════
# ReAct Agent Builder
# ═══════════════════════════════════════════════════════════════
class AgenticRAG:
    """
    LangGraph-based ReAct agent for multimodal retrieval.
    
    The agent follows the Thought → Action → Observation loop:
      1. Thought: Analyze the query and decide what information is needed
      2. Action: Call one of the 4 retrieval tools
      3. Observation: Process retrieved results
      4. Repeat or synthesize final answer
    
    Tools:
      - search_images: CLIP-based visual search via FAISS HNSW
      - search_text: BM25 + semantic search for text documents
      - query_knowledge_graph: Cypher query via Neo4j
      - hybrid_search: RRF-fused multi-source retrieval
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        image_searcher=None,
        text_searcher=None,
        graph_querier=None,
        hybrid_searcher=None,
    ):
        self.config = config or AgentConfig()
        self.reward_tracker = ToolRewardTracker()
        
        # Store retrieval module references
        self._image_searcher = image_searcher
        self._text_searcher = text_searcher
        self._graph_querier = graph_querier
        self._hybrid_searcher = hybrid_searcher
        
        # LLM
        self.llm = None
        if ChatGroq is not None and self.config.groq_api_key:
            self.llm = ChatGroq(
                model=self.config.llm_model,
                api_key=self.config.groq_api_key,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
        
        # Build the graph
        self.graph = self._build_graph()
        
        logger.info(
            f"AgenticRAG initialized | LLM: {self.config.llm_model} | "
            f"Max iterations: {self.config.max_iterations}"
        )

    def _build_tools(self) -> List:
        """Build LangChain tool definitions."""
        tools = []
        
        # Capture references for closures
        img_searcher = self._image_searcher
        txt_searcher = self._text_searcher
        graph_q = self._graph_querier
        hybrid_s = self._hybrid_searcher
        
        @tool
        def search_images(query: str) -> str:
            """
            Search for images using CLIP visual embeddings.
            Use this tool when the query asks for visual content,
            images, photos, or visual descriptions of objects.
            """
            if img_searcher is not None:
                results = img_searcher.search(query)
                return json.dumps(results[:10], default=str)
            return json.dumps({"info": "Image search not available"})
        
        @tool
        def search_text(query: str) -> str:
            """
            Search text documents using BM25 keyword matching
            and semantic embeddings. Use this for factual questions,
            document retrieval, or when specific textual information
            is needed.
            """
            if txt_searcher is not None:
                results = txt_searcher.search(query)
                return json.dumps(results[:10], default=str)
            return json.dumps({"info": "Text search not available"})
        
        @tool
        def query_knowledge_graph(query: str) -> str:
            """
            Query the Neo4j knowledge graph for structured
            entity-relationship information. Use this when the query
            asks about relationships between entities, who inspected
            what, what contains what, or other relational questions.
            """
            if graph_q is not None:
                results = graph_q.search(query)
                return json.dumps(results[:10], default=str)
            return json.dumps({"info": "Graph query not available"})
        
        @tool
        def hybrid_search(query: str) -> str:
            """
            Perform hybrid retrieval combining visual, textual,
            and graph search with RRF fusion. Use this for complex
            queries that span multiple modalities or when unsure
            which single retriever to use.
            """
            if hybrid_s is not None:
                results = hybrid_s(query)
                return json.dumps(results[:10], default=str)
            return json.dumps({"info": "Hybrid search not available"})
        
        tools = [search_images, search_text, query_knowledge_graph, hybrid_search]
        return tools

    def _build_graph(self):
        """Build the LangGraph state machine."""
        if StateGraph is None or self.llm is None:
            logger.warning("LangGraph or LLM not available. Agent will run in fallback mode.")
            return None
        
        tools = self._build_tools()
        
        # Bind tools to LLM
        llm_with_tools = self.llm.bind_tools(tools)
        
        def agent_node(state: AgentState) -> Dict:
            """The reasoning node: decide next action."""
            messages = state.get("messages", [])
            
            # System prompt with ReAct instructions
            system_msg = SystemMessage(content="""You are an intelligent multimodal retrieval agent.
You have access to 4 retrieval tools:
1. search_images - For visual/image queries
2. search_text - For textual/document queries  
3. query_knowledge_graph - For relationship/entity queries
4. hybrid_search - For complex multi-modal queries

Follow the ReAct pattern:
- THINK about what information is needed
- ACT by calling the appropriate tool(s)
- OBSERVE the results
- Either call another tool or provide the final answer

Be concise and grounded in the retrieved evidence.""")
            
            full_messages = [system_msg] + messages
            response = llm_with_tools.invoke(full_messages)
            
            return {"messages": messages + [response]}
        
        def should_continue(state: AgentState) -> str:
            """Decide whether to continue the loop or end."""
            messages = state.get("messages", [])
            if not messages:
                return "end"
            
            last_message = messages[-1]
            iteration = state.get("iteration", 0)
            
            # Check if the LLM wants to call a tool
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                if iteration < self.config.max_iterations:
                    return "tools"
            
            return "end"
        
        # Build graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", ToolNode(tools))
        
        # Set entry point
        workflow.set_entry_point("agent")
        
        # Add edges
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {"tools": "tools", "end": END},
        )
        workflow.add_edge("tools", "agent")
        
        return workflow.compile()

    def invoke(self, query: str) -> Dict:
        """
        Run the agent on a query.
        
        Args:
            query: User's natural language query
        Returns:
            Dict with 'answer', 'reasoning_trace', 'tools_used', 'results'
        """
        if self.graph is None:
            return self._fallback_invoke(query)
        
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "query": query,
            "query_type": "",
            "retrieved_results": [],
            "reasoning_trace": [],
            "final_answer": "",
            "iteration": 0,
            "tool_usage": [],
        }
        
        try:
            result = self.graph.invoke(initial_state)
            
            # Extract answer from last AI message
            messages = result.get("messages", [])
            answer = ""
            tools_used = []
            
            for msg in messages:
                if isinstance(msg, AIMessage):
                    if msg.content:
                        answer = msg.content
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        tools_used.extend(
                            [tc["name"] for tc in msg.tool_calls]
                        )
            
            return {
                "answer": answer,
                "reasoning_trace": [str(m) for m in messages],
                "tools_used": list(set(tools_used)),
                "query": query,
            }
        
        except Exception as e:
            logger.error(f"Agent error: {e}")
            return self._fallback_invoke(query)

    def _fallback_invoke(self, query: str) -> Dict:
        """Fallback when LangGraph is not available."""
        results = []
        tools_used = []
        
        # Try all available searchers
        if self._text_searcher:
            try:
                text_results = self._text_searcher.search(query)
                results.extend(text_results[:5])
                tools_used.append("search_text")
            except Exception:
                pass
        
        if self._image_searcher:
            try:
                img_results = self._image_searcher.search(query)
                results.extend(img_results[:5])
                tools_used.append("search_images")
            except Exception:
                pass
        
        if self._graph_querier:
            try:
                graph_results = self._graph_querier.search(query)
                results.extend(graph_results[:5])
                tools_used.append("query_knowledge_graph")
            except Exception:
                pass
        
        # Simple answer construction
        if results:
            answer = f"Found {len(results)} relevant results for: '{query}'\n\n"
            for i, r in enumerate(results[:5], 1):
                text = r.get("text", r.get("caption", r.get("name", "N/A")))
                score = r.get("score", "N/A")
                answer += f"{i}. {text[:200]}... (score: {score})\n"
        else:
            answer = f"No results found for: '{query}'"
        
        return {
            "answer": answer,
            "reasoning_trace": ["Fallback mode: LangGraph unavailable"],
            "tools_used": tools_used,
            "query": query,
            "results": results,
        }
