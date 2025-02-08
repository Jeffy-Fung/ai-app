from llm import get_llm
from state import State
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from retriever import get_retriever
from langchain_core.messages import AIMessage


class Node:
  def __init__(self):
    self.llm = get_llm()

  def generate_response(self, state: State) -> State:
    prompt = PromptTemplate.from_template(
      """
      You are a assistant, reporting on the economic news in the world, to assist user digesting hugh amount of news articles.
      Your job is to answer the user's question based on the news articles provided. 
      If you dont know the answer, just say "I don't know".
      \n\nDocuments: \n\n{documents}
      \n\nQuestion: {question}
      """
    )

    rag_chain = prompt | self.llm | StrOutputParser()

    return {
      "messages": [
        rag_chain.invoke({
          "documents": state["documents"],
          "question": state["messages"][-1].content
        })
      ]
    }
  
  def retrieve_documents(self, state: State) -> State:
    retrieved_documents = get_retriever(state["filtered_document_ids"]).invoke(state["messages"][-1].content)

    response_state = {
      "documents": retrieved_documents
    }

    if len(retrieved_documents) == 0:
      response_state["messages"] = [
        AIMessage(content="No documents found.")
      ]

    return response_state
