from typing import Optional, TypedDict
from langgraph.graph import StateGraph, START, END
from node import Node
from corrective_retrieval_node import CorrectiveRetrievalNode
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
    
    corrective_retrieval_node = CorrectiveRetrievalNode()
    graph_builder.add_node("retrieval_grader", corrective_retrieval_node.retrieval_grader)
    graph_builder.add_node("remove_erroneous_retrievals", corrective_retrieval_node.remove_erroneous_retrievals)
    graph_builder.add_node("define_web_search_query", corrective_retrieval_node.define_web_search_query)
    graph_builder.add_node("support_documents_with_web_search", corrective_retrieval_node.support_documents_with_web_search)

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
        True: "retrieval_grader",
        False: "generate_response",
      }
    )
    graph_builder.add_edge("retrieval_grader", "remove_erroneous_retrievals")
    graph_builder.add_edge("remove_erroneous_retrievals", "define_web_search_query")
    graph_builder.add_edge("define_web_search_query", "support_documents_with_web_search")
    graph_builder.add_edge("support_documents_with_web_search", "generate_response")
    graph_builder.add_edge("generate_response", END)
    self.graph = graph_builder.compile()

def make_graph():
  return Graph().graph
