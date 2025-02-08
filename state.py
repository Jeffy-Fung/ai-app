from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import add_messages

class State(TypedDict):
  message_histories: Annotated[list, add_messages]
  raw_input: str
  rephrased_input: str
  search_query: str
  filtered_document_ids: list[int | str]
  documents: list
  output: str
