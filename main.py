from graph import Graph
from fastapi import FastAPI
# from psycopg_pool import ConnectionPool
import os
# from langgraph.checkpoint.postgres import PostgresSaver
from dotenv import load_dotenv

load_dotenv()

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

# @app.get("/invoke_with_checkpoint")
# async def invoke_with_checkpoint():
#   connection_kwargs = {
#     "autocommit": True,
#     "prepare_threshold": 0,
#   }

#   with ConnectionPool(
#     conninfo=os.getenv("POSTGRES_DB_URI"),
#     max_size=20,
#     kwargs=connection_kwargs,
#   ) as pool:
#     checkpointer = PostgresSaver(pool)
#     checkpointer.setup()
#     graph = Graph(checkpointer).graph
#     result = graph.invoke({
#       "messages": [("user", "Quiz on World of Warcraft")],
#       "learning_objectives": "1. Understand the history of World of Warcraft\n2. Understand the lore of World of Warcraft\n3. Understand the characters of World of Warcraft"
#     },
#     config={
#       "configurable": {
#         "thread_id": "1"
#       }
#     })
#   return result
