from llm import get_llm
from state import State
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from retriever import get_retriever
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever

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
    retriever = get_retriever(state["filtered_document_ids"])

    system_instruction = """Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is."""

    prompt = ChatPromptTemplate.from_messages([
      ("system", system_instruction),
      MessagesPlaceholder("chat_history"),
      ("human", "{input}")
    ])

    history_aware_retriever = create_history_aware_retriever(
      self.llm,
      retriever,
      prompt
    )
    
    retrieved_documents = history_aware_retriever.invoke({
      "input": state["messages"][-1].content,
      "chat_history": state["messages"][:-1]
    })

    response_state = {
      "documents": retrieved_documents
    }

    if len(retrieved_documents) == 0:
      response_state["messages"] = [
        AIMessage(content="No documents found.")
      ]

    return response_state
