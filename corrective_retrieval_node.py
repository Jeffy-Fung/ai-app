from state import State
from llm import get_llm
from web_search_tool import get_web_search_tool
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

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
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a score 'yes', 'no', or 'maybe' to indicate whether the document is relevant to the question."""

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

# Data model for the retrieval grader
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    score: str = Field(
        description="Documents are relevant to the question, 'yes', 'no', or 'maybe'"
    )
