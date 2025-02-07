from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv


load_dotenv()

def get_embedding():
  return OpenAIEmbeddings(model="text-embedding-3-large")
