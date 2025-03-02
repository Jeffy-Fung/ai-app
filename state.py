from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import add_messages

class State(TypedDict):
  message_histories: Annotated[list, add_messages]
  recent_chat_history: Annotated[list, add_messages]
  user_query: str
  summary: str
  search_query: str
  filtered_document_ids: list[int | str]
  documents: list
  web_search_results: list
  output: str
