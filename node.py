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
      "output": AIMessage(
          rag_chain.invoke({
            "documents": state["documents"],
            "question": state["rephrased_input"]
          })
        )
    }

  def retrieve_documents(self, state: State) -> State:
    retriever = get_retriever(state["filtered_document_ids"])

    retrieved_documents = retriever.invoke(state["rephrased_input"])

    response_state = {
      "documents": retrieved_documents
    }

    if len(retrieved_documents) == 0:
      response_state["output"] = "No documents found."

    return response_state
  
  def rephrase_input_based_on_history(self, state: State) -> State:
    prompt = PromptTemplate.from_template(
      """
        Given a chat history and the latest user question which might reference context in the chat history, 
        please rewrite the question to be standalone question, which can be understood without the chat history.
        Remember, DO NOT answer the question.
        \n\nChat History: {chat_history}
        \n\nQuestion: {input}
      """
    )

    rephased_input = prompt | self.llm | StrOutputParser()

    return {
      "rephrased_input": rephased_input.invoke({
          "chat_history": state["message_histories"],
          "input": state["raw_input"]
        })
    }
