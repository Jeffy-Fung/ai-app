from langchain_qdrant import QdrantVectorStore
from qdrant_client import models
from embedding import get_embedding

from dotenv import load_dotenv
import os

load_dotenv()


def get_retriever(filtered_document_ids: list[int | str] | None = None) -> QdrantVectorStore:
  embedding = get_embedding()

  return QdrantVectorStore.from_existing_collection(
    collection_name="news",
    embedding=embedding,
  url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
  ).as_retriever(
    search_kwargs={
      "filter": models.Filter(
        should=[
          models.FieldCondition(
            key="metadata.id",
            match=models.MatchAny(any=filtered_document_ids)
          )
        ]
      ),
      "k": 5
    }
  )
