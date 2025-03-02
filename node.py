from llm import get_llm
from state import State
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from retriever import get_retriever
from langchain_core.messages import AIMessage, trim_messages
from web_search_tool import get_web_search_tool


class Node:
  def __init__(self):
    self.llm = get_llm()
    
  def extract_recent_chat_history(self, state: State) -> State:
    return {
      "recent_chat_history": trim_messages(
          state["message_histories"],
          token_counter=len,
          strategy="last",
          include_system=False,
          max_tokens=5
        ),
    }

  def generate_response(self, state: State) -> State:
    prompt = PromptTemplate.from_template(
      """
      You are a assistant, reporting on the economic news in the world, to assist user digesting hugh amount of news articles.
      Your job is to answer the user's question based on the news articles provided. 
      You can use the latest 5 conversations and the summary of the chat history to help you understand the context of the user's question.
      If you dont know the answer, just say "I don't know".

      \n\nQuestion: {question}
      \n\nNews articles: \n\n{documents}
      \n\nSummary of previous conversations: {summary}
      \n\nThe latest 5 conversations: {recent_chat_history}
      """
    )

    rag_chain = prompt | self.llm | StrOutputParser()

    return {
      "output": AIMessage(
          rag_chain.invoke({
            "documents": state["documents"],
            "question": state["user_query"],
            "summary": state["summary"],
            "recent_chat_history": state["recent_chat_history"]
          })
        )
    }

  def retrieve_documents(self, state: State) -> State:
    retriever = get_retriever(state["filtered_document_ids"])

    retrieved_documents = retriever.invoke(state["search_query"])

    if len(retrieved_documents) == 0:
      return {
        "output": AIMessage(
          "No documents found."
        )
      }

    return {
      "documents": retrieved_documents
    }
  
  def generate_search_query(self, state: State) -> State:
    prompt = PromptTemplate.from_template(
      """
        You are an AI assistant specializing in Question-Answering (QA) tasks within a Retrieval-Augmented Generation (RAG) system. 
        Your primary mission is to answer questions based on the provided context or chat history.
        Ensure your response is concise and directly addresses the question without any additional narration.

        Now, given the latest 5 conversations and the summary of the chat history, generate a search query to look up to get information relevant to the conversation.

        \n\nQuestion: {user_query}
        \n\nLatest 5 conversations: {recent_chat_history}
        \n\nSummary of chat history: {summary}
      """
    )

    chain = prompt | self.llm | StrOutputParser()

    return {
      "search_query": chain.invoke({
          "user_query": state["user_query"],
          "recent_chat_history": state["recent_chat_history"],
          "summary": state["summary"]
        })
    }

  def generate_chat_summary(self, state: State) -> State:
    prompt = PromptTemplate.from_template(
      """
        Given a chat history and the latest user question which might reference context in the chat history, 
        please generate a summary of the chat history, in order to help the LLM understand the context of the user's question.
        Remember, DO NOT answer the question.

        \n\nUser Question: {user_query}
        \n\nChat History: {chat_history}
      """
    )

    summary_generator = prompt | self.llm | StrOutputParser()

    return {
      "summary": summary_generator.invoke({
          "chat_history": state["message_histories"],
          "user_query": state["user_query"]
        })
    }

  def web_search(self, state: State) -> State:
    web_search_tool = get_web_search_tool()
    return {
      "web_search_results": web_search_tool.invoke(state["search_query"])
    }
