import json
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from src.agent.tools import predict_default_risk, explain_with_shap
from src.agent.extractor import process_document
from src.agent.prompts import REPORT_GENERATION_PROMPT

class UnderwritingCopilot:
    def __init__(self):
        self.llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    
    def process_file(self, file_path: str):
        """Phase 1: Document Upload and Extraction"""
        return process_document(file_path)
    
    def analyze_risk(self, features_dict: dict) -> dict:
        """Phase 2: Prediction and Explainability"""
        features_json = json.dumps(features_dict)
        
        prediction_result = predict_default_risk.invoke({"features_json": features_json})
        explanation = explain_with_shap.invoke({"features_json": features_json})
        
        pred_data = json.loads(prediction_result)
        
        return {
            "prediction": pred_data,
            "explanation": explanation
        }
        
    def generate_report(self, features: dict, analysis: dict) -> str:
        """Phase 3: Final Report Generation"""
        messages = [
            SystemMessage(content=REPORT_GENERATION_PROMPT),
            HumanMessage(content=f"Features:\n{json.dumps(features, indent=2)}\n\nPrediction Data:\n{json.dumps(analysis['prediction'], indent=2)}\n\nSHAP Analysis:\n{analysis['explanation']}")
        ]
        
        response = self.llm.invoke(messages)
        return response.content

    def answer_question(self, user_query: str, context: dict) -> str:
        """Handles follow-up chat questions using the context"""
        sys_prompt = "You are an AI Underwriting Copilot. Use the context provided about the applicant to answer the user's questions. Remember not to use any emojis."
        messages = [
            SystemMessage(content=sys_prompt),
            HumanMessage(content=f"Context:\n{json.dumps(context, indent=2)}\n\nUser Question: {user_query}")
        ]
        response = self.llm.invoke(messages)
        return response.content
