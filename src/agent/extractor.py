import pdfplumber
import json
import re
import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from src.agent.prompts import DATA_EXTRACTION_SYSTEM_PROMPT

def extract_text(file_path: str) -> str:
    """Extract text from a PDF file."""
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        return f"Error extracting PDF: {str(e)}"
    return text




def parse_features(text: str) -> dict:
    """Use the LLM to parse raw text into structured features."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable not set.")
    
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    
    messages = [
        SystemMessage(content=DATA_EXTRACTION_SYSTEM_PROMPT),
        HumanMessage(content=f"Please extract the financial features from the following raw text. Return ONLY a valid JSON object. Text:\n\n{text}")
    ]
    
    response = llm.invoke(messages)
    
    # Parse the JSON response
    try:
        content = response.content.strip()
        # Find JSON payload dynamically if wrapped in markdown
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
        if match:
            content = match.group(1)
        return json.loads(content.strip())
    except json.JSONDecodeError:
        print("Failed to decode JSON:", response.content)
        return {}

def process_document(file_path: str) -> dict:
    """End-to-end document processing."""
    text = extract_text(file_path)
    if "Error extracting" in text:
        return {"error": text}
    
    return parse_features(text)
