from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import add_messages

class State(TypedDict):
  messages: Annotated[list, add_messages]
  documents: list
  filtered_document_ids: list[int | str]
