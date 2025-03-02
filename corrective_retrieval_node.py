from state import State
from llm import get_llm
from web_search_tool import get_web_search_tool
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class CorrectiveRetrievalNode:
  def __init__(self):
    self.llm = get_llm()
  
  def web_search(self, state: State) -> State:
    web_search_tool = get_web_search_tool()
    return {
      "web_search_results": web_search_tool.invoke(state["search_query"])
    }

  def retrieval_grader(self, state: State) -> State:
    system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    For example: If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    The goal is to filter out erroneous retrievals and to supplement the missing information for the ambiguous retrieved document when answering according to the user question. \n
    Give a score 'yes', 'no', or 'maybe' to indicate whether the document is relevant, erroneous, or ambiguous to the question."""

    grade_prompt = ChatPromptTemplate.from_messages(
      [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
      ]
    )
    structured_llm_grader = self.llm.with_structured_output(GradeDocuments)
    
    retrieval_grader = grade_prompt | structured_llm_grader

    documents = state["documents"]
    documents_with_scores = []
    for document in documents:
      result = retrieval_grader.invoke({
        "document": document,
        "question": state["user_query"]
      })
      documents_with_scores.append({
        "document": document,
        "score": result.score
      })

    return {
      "documents_with_scores": documents_with_scores
    }
    
  def define_web_search_query(self, state: State) -> State:
    prompt = PromptTemplate.from_template(
      """
      You are a helpful assistant that defines a web search query, based on a user question and a retrieved document.
      Your goal is to define a web search query, that is going to be searched on the web,
      to supplement the missing information for the retrieved document when answering according to the user question.
      
      User question: {question}
      Retrieved document: {document}
      """
    )

    chain = prompt | self.llm | StrOutputParser()
    
    documents = state["documents"]
    documents_with_scores = []
    for document in documents:
      if document.score == "maybe":
        result = chain.invoke({
          "question": state["user_query"],
          "document": document
        })
        documents_with_scores.append({
          "document": document,
          "score": document.score,
          "web_search_query": result
        })
      else:
        documents_with_scores.append({
          "document": document,
          "score": document.score,
          "web_search_query": None
        })

    return {
      "documents_with_scores": documents_with_scores
    }

  def support_documents_with_web_search(self, state: State) -> State:
    pass

# Data model for the retrieval grader
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    score: str = Field(
        description="Documents are relevant to the question, 'yes', 'no', or 'maybe'"
    )
