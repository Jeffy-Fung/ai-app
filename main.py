from graph import Graph
from fastapi import FastAPI
from llm import get_llm
from pydantic import BaseModel
from enum import Enum
from typing import Annotated
from fastapi import Query
app = FastAPI()

@app.get("/")
async def health_check():
  return "Success response"

@app.get("/invoke")
async def invoke():
  graph = Graph().graph
  result = graph.invoke({
    "messages": [("user", "Quiz on World of Warcraft")],
    "learning_objectives": "1. Understand the history of World of Warcraft\n2. Understand the lore of World of Warcraft\n3. Understand the characters of World of Warcraft"
  })

  return result

class Role(str, Enum):
  SYSTEM = "system"
  HUMAN = "human"

class ChatRequest(BaseModel):
  messages: list[tuple[Role, str]]

@app.post("/simple-chat")
async def chat(request: ChatRequest):
  print("request: ", request)
  llm = get_llm()
  return llm.invoke(request.messages)

@app.get("/news-details")
async def get_news(urls: Annotated[list[str], Query()]):
  from newspaper.mthreading import fetch_news

  articles = fetch_news(urls)

  return articles

class EmbedNewsRequest(BaseModel):
  articles: list[dict[str, str | int]]

@app.post("/embed-news")
async def embed_news(request: EmbedNewsRequest):
  from langchain_text_splitters import RecursiveCharacterTextSplitter
  from langchain_core.documents import Document
  from langchain_qdrant import QdrantVectorStore
  import os
  import uuid
  from embedding import get_embedding

  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

  docs = [
    Document(
      page_content=article["content"],
      metadata={
        "url": article["url"],
        "title": article["title"],
        "id": article["id"]
      }
    )
    for article in request.articles
  ]
  splitted_documents = text_splitter.split_documents(docs)
  
  embedding = get_embedding()
  ids = [str(uuid.uuid4()) for _ in splitted_documents]
  QdrantVectorStore.from_documents(
    splitted_documents,
    embedding=embedding,
    collection_name="news",
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    ids=ids,
  )

  return splitted_documents
