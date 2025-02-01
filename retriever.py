# from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import models

from dotenv import load_dotenv
import os

load_dotenv()


def get_retriever() -> QdrantVectorStore:
  # embedding = FastEmbedEmbeddings(model_name="nomic-ai/nomic-embed-text-v1.5-Q")
  embedding = OpenAIEmbeddings(model="text-embedding-3-large")

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
