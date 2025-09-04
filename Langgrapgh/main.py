import os
import json
from typing import List, Annotated
from dotenv import load_dotenv
from typing_extensions import TypedDict

# LangChain Imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI 
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
# LangGraph Imports
from langgraph.graph import StateGraph, START, END



load_dotenv()

class GraphState(TypedDict):
    log_filepath: str
    retriever: object
    error_snippets: List[str]
    extracted_codes: List[int]
    final_json_output: str

# --- Pydantic Model for Structured LLM Output ---
class ErrorCodes(BaseModel):
    """A list of numeric error codes found in the log snippets."""
    codes: List[int] = Field(description="A list of unique integer error codes. Example: [1234, 500, 401]")

# --- Node Functions ---

def ingest_and_index(state: GraphState) -> GraphState:
    """
    Node 1: Reads the log file, creates document chunks, generates embeddings,
    and builds the RAG retriever.
    """
    print("--- 1. INGESTING AND INDEXING LOG FILE ---")
    filepath = state["log_filepath"]
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Log file not found at: {filepath}")
    
    with open(filepath, 'r') as f:
        log_content = f.read()
    lines = log_content.splitlines()
    documents = [Document(page_content=line) for line in lines]
    print(f"   > Ingested {len(documents)} lines from '{filepath}'")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001") 
    
    vectorstore = FAISS.from_documents(documents, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    print("   > FAISS vector store and retriever created with Google Gemini Embeddings.")
    return {"retriever": retriever}

def retrieve_errors(state: GraphState) -> GraphState:
    """
    Node 2: Uses the retriever to find log lines that likely contain errors.
    """
    print("--- 2. RETRIEVING ERROR SNIPPETS (RAG) ---")
    retriever = state["retriever"]
    query = "Find all logs containing errors, failures, exceptions, or fault codes"
    
    retrieved_docs = retriever.invoke(query)
    error_snippets = [doc.page_content for doc in retrieved_docs]
    
    print(f"   > Retrieved {len(error_snippets)} potential error snippets.")
    return {"error_snippets": error_snippets}

def extract_codes(state: GraphState) -> GraphState:
    """
    Node 3: Passes retrieved snippets to an LLM to extract numeric error codes.
    """
    print("--- 3. EXTRACTING NUMERIC CODES WITH LLM ---")
    error_snippets = state["error_snippets"]
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")
    structured_llm = llm.with_structured_output(ErrorCodes)
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", 
         "You are an expert log analysis assistant. Your task is to extract all unique "
         "numeric error codes from the provided log snippets. "
         "Handle various formats like 'ERR-1234' (extract 1234), 'code: 500' (extract 500), "
         "and hexadecimal like '0x1F4' (convert to decimal 500). "
         "Return only a list of integers."),
        ("human", "Here are the log snippets:\n\n---\n\n{snippets}")
    ])
    
    chain = prompt_template | structured_llm
    snippets_str = "\n".join(error_snippets)
    result = chain.invoke({"snippets": snippets_str})
    
    print(f"   > LLM extracted the following codes: {result.codes}")
    return {"extracted_codes": result.codes}

def format_output(state: GraphState) -> GraphState:
    """
    Node 4: Cleans the extracted codes to be unique and formats the final JSON output.
    """
    print("--- 4. FORMATTING FINAL OUTPUT ---")
    extracted_codes = state["extracted_codes"]
    
    unique_codes = sorted(list(set(extracted_codes)))
    output_dict = {"unique_error_codes": unique_codes}
    final_json = json.dumps(output_dict, indent=2)
    
    print("   > Final JSON created.")
    return {"final_json_output": final_json}

# --- Graph Assembly ---
graph_builder = StateGraph(GraphState)

graph_builder.add_node("ingest_and_index", ingest_and_index)
graph_builder.add_node("retrieve_errors", retrieve_errors)
graph_builder.add_node("extract_codes", extract_codes)
graph_builder.add_node("format_output", format_output)

graph_builder.add_edge(START, "ingest_and_index")
graph_builder.add_edge("ingest_and_index", "retrieve_errors")
graph_builder.add_edge("retrieve_errors", "extract_codes")
graph_builder.add_edge("extract_codes", "format_output")
graph_builder.add_edge("format_output", END)

app = graph_builder.compile()

if __name__ == "__main__":
    inputs = {"log_filepath": "log.txt"}
    final_state = app.invoke(inputs)
    
    print("\n" + "="*50)
    print("✅ Final Output:")
    print(final_state["final_json_output"])
    print("="*50)



    try:
        png_data = app.get_graph().draw_mermaid_png()
        with open("graph.png", "wb") as f:
            f.write(png_data)
        print("\nGraph visualization saved to 'graph.png'")
    except Exception as e:
        print(f"\nCould not generate graph visualization: {e}")
