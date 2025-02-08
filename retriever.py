from langchain_qdrant import QdrantVectorStore
from qdrant_client import models
from embedding import get_embedding

from dotenv import load_dotenv
import os

load_dotenv()


def get_retriever() -> QdrantVectorStore:
  embedding = get_embedding()

  return QdrantVectorStore.from_existing_collection(
    collection_name="testing_on_wow",
    embedding=embedding,
  url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
  ).as_retriever(
    search_kwargs={
      "filter": models.Filter(
        must=[
          models.FieldCondition(
            key="metadata.title",
            match=models.MatchValue(value="World of Warcraft")
          )
        ]
      ),
      "k": 5
    }
  )
