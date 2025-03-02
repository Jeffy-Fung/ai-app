from state import State
from llm import get_llm
from web_search_tool import get_web_search_tool
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document

class CorrectiveRetrievalNode:
  def __init__(self):
    self.llm = get_llm()

  def retrieval_grader(self, state: State) -> State:
    system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    For example: If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    The goal is to filter out erroneous retrievals and to supplement the missing information for the ambiguous retrieved document when answering according to the user question. \n
    Give a score of 'relevant', 'irrelevant', or 'ambiguous' to indicate whether the document is relevant, erroneous, or ambiguous to the question."""

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
    
  def remove_erroneous_retrievals(self, state: State) -> State:
    documents_with_scores = state["documents_with_scores"]
    documents = [document for document in documents_with_scores if document["score"] != "irrelevant"]
    return {
      "documents_with_scores": documents
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
      if document["score"] == "ambiguous":
        result = chain.invoke({
          "question": state["user_query"],
          "document": document
        })
        documents_with_scores.append({
          "document": document,
          "score": document["score"],
          "web_search_query": result
        })
      else:
        documents_with_scores.append({
          "document": document,
          "score": document["score"],
          "web_search_query": None
        })

    return {
      "documents_with_scores": documents_with_scores
    }

  def support_documents_with_web_search(self, state: State) -> State:
    documents_with_scores = state["documents_with_scores"]
    
    web_search_results = []
    for document in documents_with_scores:
      if document.web_search_query is None:
        continue

      web_search_tool = get_web_search_tool()
      results = web_search_tool.invoke(document.web_search_query)
      overall_results = "\n".join([d["content"] for d in results])
      overall_results = Document(page_content=overall_results)
      web_search_results.append({
        "document": document,
        "web_search_results": overall_results
      })

    return {
      "web_search_results": web_search_results
    }

# Data model for the retrieval grader
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    score: str = Field(
        description="Documents are relevant to the question, 'relevant', 'irrelevant', or 'ambiguous'"
    )
