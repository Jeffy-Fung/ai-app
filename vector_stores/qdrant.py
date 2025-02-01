from langchain_community.document_loaders import WikipediaLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
import os
import uuid
import dotenv

from qdrant_client import QdrantClient

dotenv.load_dotenv()

def get_qdrant_client() -> QdrantClient:
  return QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
  )

def get_vector_db_from_wikipedia_pages(embedding, query: str, collection_name: str) -> QdrantVectorStore:
  qdrant_client = get_qdrant_client()

  is_collection_exist = qdrant_client.collection_exists(collection_name)

  if is_collection_exist:
    print(f"Vector store already exists. No need to initialize.")

    return QdrantVectorStore.from_existing_collection(
      collection_name=collection_name,
      embedding=embedding,
      url=os.getenv("QDRANT_URL"),
      api_key=os.getenv("QDRANT_API_KEY"),
    )

  docs = WikipediaLoader(query=query, load_max_docs=2).load()

  print(f"Found {len(docs)} pieces of documents about {query}.")

  print(f"Here is the details of the found documents:")
  for doc in docs:
    print(f"Title: {doc.metadata['title']}")
    print(f"Source: {doc.metadata['source']}")

  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
  splitted_documents = text_splitter.split_documents(docs)

  ids = [str(uuid.uuid4()) for _ in splitted_documents]

  qdrant = QdrantVectorStore.from_documents(
    splitted_documents,
    embedding=embedding,
    collection_name=collection_name,
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    ids=ids,
  )

  return qdrant