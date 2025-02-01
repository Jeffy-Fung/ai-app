from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


def get_llm():
  return ChatOpenAI(model="gpt-3.5-turbo-16k")
