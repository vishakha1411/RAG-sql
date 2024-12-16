from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START
from langchain_ollama import ChatOllama
#from langchain.embeddings import NomicTextEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from agent_graph.tool_chinook_sqlagent import query_chinook_sqldb
from agent_graph.tool_travel_sqlagent import query_travel_sqldb
from agent_graph.tool_lookup_policy_rag import lookup_swiss_airline_policy
from agent_graph.tool_tavily_search import load_tavily_search_tool
from agent_graph.tool_stories_rag import lookup_stories
from agent_graph.load_tools_config import LoadToolsConfig
from agent_graph.agent_backend import State, BasicToolNode, route_tools, plot_agent_schema

TOOLS_CFG = LoadToolsConfig()

def build_graph():
    """
    Builds an agent decision-making graph by combining an LLM with various tools
    and defining the flow of interactions between them.

    This function sets up a state graph where a primary language model (LLM) interacts
    with several predefined tools (e.g., databases, search functions, policy lookup, etc.).
    """
    # Initialize Llama3.1 using Ollama
    primary_llm = ChatOllama(model="llama3.1",base_url="http://127.0.0.1:11434", temperature=TOOLS_CFG.primary_agent_llm_temperature)

    # Initialize Nomic-Text embeddings for any embedding-based functionality
    embedding=OllamaEmbeddings(model="nomic-embed-text")

    graph_builder = StateGraph(State)

    # Load tools with their proper configs
    search_tool = load_tavily_search_tool(TOOLS_CFG.tavily_search_max_results)
    tools = [
        search_tool,
        lookup_swiss_airline_policy,
        lookup_stories,
        query_travel_sqldb,
        query_chinook_sqldb,
    ]

    # Tell the LLM which tools it can call
    primary_llm_with_tools = primary_llm.bind_tools(tools)

    def chatbot(state: State):
        """Executes the primary language model with tools bound and returns the generated message."""
        return {"messages": [primary_llm_with_tools.invoke(state["messages"])]}

    graph_builder.add_node("chatbot", chatbot)

    tool_node = BasicToolNode(
        tools=[
            search_tool,
            lookup_swiss_airline_policy,
            lookup_stories,
            query_travel_sqldb,
            query_chinook_sqldb,
        ]
    )
    graph_builder.add_node("tools", tool_node)

    # The `tools_condition` function returns "tools" if the chatbot asks to use a tool, and "__end__" otherwise
    graph_builder.add_conditional_edges(
        "chatbot",
        route_tools,
        {"tools": "tools", "__end__": "__end__"},
    )

    # Any time a tool is called, we return to the chatbot to decide the next step
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge(START, "chatbot")

    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)
    plot_agent_schema(graph)

    return graph
