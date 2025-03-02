from typing import Optional, TypedDict
from langgraph.graph import StateGraph, START, END
from node import Node
from state import State
from dotenv import load_dotenv

load_dotenv()

class ConfigSchema(TypedDict):
  pending: Optional[str]


def is_documents_found(state: State):
  return len(state.get("documents")) > 0

def is_conversation_long(state: State):
  return len(state.get("message_histories")) > 5

class Graph:
  def __init__(self):
    graph_builder = StateGraph(State, ConfigSchema)
    node = Node()

    graph_builder.add_node("extract_recent_chat_history", node.extract_recent_chat_history)
    graph_builder.add_node("generate_chat_summary", node.generate_chat_summary)
    graph_builder.add_node("generate_search_query", node.generate_search_query)
    graph_builder.add_node("retrieve_documents", node.retrieve_documents)
    graph_builder.add_node("generate_response", node.generate_response)

    graph_builder.add_edge(START, "extract_recent_chat_history")
    graph_builder.add_conditional_edges(
      "extract_recent_chat_history",
      is_conversation_long,
      {
        True: "generate_chat_summary",
        False: "generate_search_query"
      }
    )
    graph_builder.add_edge("generate_chat_summary", "generate_search_query")
    graph_builder.add_edge("generate_search_query", "retrieve_documents")
    graph_builder.add_conditional_edges(
      "retrieve_documents",
      is_documents_found,
      {
        True: "generate_response",
        False: END,
      }
    )
    graph_builder.add_edge("generate_response", END)
    self.graph = graph_builder.compile()

def make_graph():
  return Graph().graph
