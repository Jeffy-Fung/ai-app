from vector_stores.qdrant import get_vector_db_from_wikipedia_pages
from langchain_openai import OpenAIEmbeddings


def main():
  embedding = OpenAIEmbeddings(model="text-embedding-3-large")
  vector_db = get_vector_db_from_wikipedia_pages(
    embedding=embedding,
    query="World of Warcraft",
    collection_name="testing_on_wow"
  )

  results = vector_db.similarity_search_with_score(
      query="Who is the creator of World of Warcraft?",
  )
  for doc, score in results:
      print(f"* [SIM={score:3f}]")
      print(f" Metadata: {doc.metadata}")
      print(f" Content: {doc.page_content}")

if __name__ == "__main__":
  main()