from llm import get_llm
from state import State
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from retriever import get_retriever
from langchain_core.messages import AIMessage


class Node:
  def __init__(self):
    self.llm = get_llm()
    self.retriever = get_retriever()

  def generate_response(self, state: State) -> State:
    prompt = PromptTemplate.from_template(
      """
      You are a teacher creating a quiz of the given documents to secondary school students. 
      The quiz should consist of 5 questions, in the format of multiple choice, with 4 options, and the correct answer.
      The quiz should test the following learning objectives: {learning_objectives}
      \nDocuments: \n\n{documents}
      """
    )

    rag_chain = prompt | self.llm | StrOutputParser()

    return {
      "messages": [
        rag_chain.invoke({
          "documents": state["documents"],
          "learning_objectives": state["learning_objectives"]
        })
      ]
    }
  
  def retrieve_documents(self, state: State) -> State:
    retrieved_documents = self.retriever.invoke(state["learning_objectives"])

    response_state = {
      "documents": retrieved_documents
    }

    if len(retrieved_documents) == 0:
      response_state["messages"] = [
        AIMessage(content="No documents found.")
      ]

    return response_state
