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
