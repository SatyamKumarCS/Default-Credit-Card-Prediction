"""
AI Underwriting Copilot Page - Streamlit Dashboard
No emojis as requested.
"""
import sys
import os
import json
import base64
from io import BytesIO

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, BASE_DIR)

import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from src.agent.copilot import UnderwritingCopilot
from src.agent.extractor import extract_text

# Load environment variables
load_dotenv()

# Must be the FIRST Streamlit call
st.set_page_config(
    page_title="AI Copilot",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Dark-theme CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

:root {
  --bg-base   : #0D1117;
  --bg-surface: #161B22;
  --bg-card   : #1C2333;
  --bg-input  : #1C2333;
  --border    : #30363D;
  --border-hi : #58A6FF;
  --text-hi   : #E6EDF3;
  --text-mid  : #8B949E;
  --text-low  : #484F58;
  --accent    : #58A6FF;
  --accent2   : #BC8CFF;
  --green     : #3FB950;
  --orange    : #F78166;
  --yellow    : #E3B341;
  --radius-lg : 16px;
  --radius-md : 12px;
  --radius-sm : 8px;
  --shadow    : 0 4px 24px rgba(0,0,0,0.5);
}

html, body, [class*="css"] {
  font-family: 'Inter', system-ui, sans-serif !important;
  background-color: var(--bg-base) !important;
  color: var(--text-hi) !important;
}
.block-container {
  padding: 20px 28px 32px !important;
  max-width: 1440px !important;
}

#MainMenu, footer, header { display: none !important; }

/* Upload Area */
[data-testid="stFileUploadDropzone"] {
    background: var(--bg-input) !important;
    border: 1px dashed var(--border-hi) !important;
    border-radius: var(--radius-md) !important;
}

/* Chat Bubbles */
[data-testid="stChatMessage"] {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    padding: 10px 15px !important;
}
</style>
""", unsafe_allow_html=True)

# Application Header
st.markdown("""
<div style="background:#161B22;border-radius:14px;padding:16px 24px;margin-bottom:20px;
            display:flex;align-items:center;justify-content:space-between;
            border:1px solid #30363D;box-shadow:0 4px 24px rgba(0,0,0,0.5);">
  <div style="display:flex;align-items:center;gap:14px;">
    <div style="width:44px;height:44px;border-radius:12px;
                background:linear-gradient(135deg,#58A6FF,#BC8CFF);
                display:flex;align-items:center;justify-content:center;
                font-size:18px;font-weight:800;color:#0D1117;">AI</div>
    <div>
      <div style="font-size:20px;font-weight:800;color:#E6EDF3;">
        Underwriting<span style="color:#58A6FF;">Copilot</span>
      </div>
      <div style="font-size:11px;color:#8B949E;">Automated Extraction and Risk Assessment</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

with st.expander("Instructions: What this Copilot does and how to use it", expanded=False):
    st.markdown("""
    **What this Copilot does:**
    This tool functions as an AI-powered underwriting assistant. It automatically extracts financial features from raw bank statements via an LLM and passes them into our predictive ML model. 
    It then returns a human-readable risk report based on SHAP (explainable AI) factors to help loan officers easily understand *why* the prediction was made.
    
    **How to use it:**
    1. **Upload a PDF Document** (or select a built-in demo profile under **Test with Demo Data**).
    2. Click **Process Document** (or **Use Selected Demo Profile**).
    3. The Copilot will autonomously:
       - Extract unstructured data directly into our 24 required ML features.
       - Run the Logistic Regression inference pipeline under the hood.
       - Explain the Risk Score using SHAP variables formatted in plain English.
       - Output a readable recommendation report indicating if a human review is required.
    4. You can then use the chat box to ask follow-up questions about the applicant's data!
    """)


if not os.environ.get("GROQ_API_KEY"):
    st.error("GROQ_API_KEY is not set in the environment. Please add it to your .env file or environment variables.")
    st.stop()


# Initialize state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "copilot" not in st.session_state:
    st.session_state.copilot = UnderwritingCopilot()
if "context" not in st.session_state:
    st.session_state.context = None
if "file_processed" not in st.session_state:
    st.session_state.file_processed = False


# Two-column layout
left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.markdown("### Document Upload")
    
    uploaded_file = st.file_uploader("Upload Bank Statement or Application (PDF)", type=["pdf"])
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Process Document", width="stretch") and uploaded_file is not None:
            with st.spinner("Extracting text and features..."):
                # Save uploaded file temporarily
                temp_path = os.path.join(BASE_DIR, "temp_upload.pdf")
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Use extractor tool
                features = st.session_state.copilot.process_file(temp_path)
                
                os.remove(temp_path)
                
                if "error" in features:
                    st.error(features["error"])
                else:
                    st.session_state.context = {"features": features}
                    st.session_state.file_processed = True
                    st.session_state.messages.append({"role": "assistant", "content": "Document processed successfully. Proceeding to Agent Risk Analysis..."})
                    st.rerun()
                    
    with col2:
        st.markdown("**Test with Demo Data**")
        sample_choice = st.selectbox(
            "Select Risk Profile", 
            ["Grey Zone (Medium Risk)", "Low Risk profile", "High Risk profile"],
            label_visibility="collapsed"
        )
        
        if st.button("Use Selected Demo Profile", width="stretch"):
            filename_map = {
                "Low Risk profile": "sample_low_risk.pdf",
                "Grey Zone (Medium Risk)": "sample_grey_zone.pdf",
                "High Risk profile": "sample_high_risk.pdf"
            }
            sample_path = os.path.join(BASE_DIR, "samples", filename_map[sample_choice])
            
            if os.path.exists(sample_path):
                with st.spinner("Processing demo document..."):
                    features = st.session_state.copilot.process_file(sample_path)
                    st.session_state.context = {"features": features}
                    st.session_state.file_processed = True
                    st.session_state.messages.append({"role": "assistant", "content": "Sample default document mapped and processed. Proceeding to Agent Risk Analysis..."})
                    st.rerun()
            else:
                st.warning("Sample statement not found. Please run the generate script first.")

with right_col:
    st.markdown("### Agent Interaction")
    
    chat_container = st.container(height=500)
    
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
        if st.session_state.file_processed and "analysis" not in st.session_state.context:
            with st.spinner("Agent running prediction and explainability model..."):
                features = st.session_state.context["features"]
                analysis = st.session_state.copilot.analyze_risk(features)
                st.session_state.context["analysis"] = analysis
                
                pred_pct = analysis["prediction"].get("prediction_percentage", 0)
                decision_zone = analysis["prediction"].get("decision_zone", "")
                
                status_msg = f"Prediction complete. Default Probability: {pred_pct}%."
                if decision_zone == "grey_zone":
                    status_msg += " This falls in the Grey Zone (30%-60%). HUMAN REVIEW REQUIRED."
                elif decision_zone == "low_risk":
                    status_msg += " This is Low Risk. Auto-approval recommended."
                else:
                    status_msg += " This is High Risk. Auto-decline recommended."
                    
                st.session_state.messages.append({"role": "assistant", "content": status_msg})
                
                # Generate Report
                with st.spinner("Agent compiling final report..."):
                    report = st.session_state.copilot.generate_report(features, analysis)
                    st.session_state.context["report"] = report
                    st.session_state.messages.append({"role": "assistant", "content": report})
                
                st.rerun()

    # Chat input for follow-up questions
    if st.session_state.context and "report" in st.session_state.context:
        if prompt := st.chat_input("Ask follow-up questions about this applicant..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = st.session_state.copilot.answer_question(prompt, st.session_state.context)
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})

# Risk Context Dashboard section
if st.session_state.context and "analysis" in st.session_state.context:
    st.markdown("---")
    st.markdown("### Extracted Features & Context")
    
    context = st.session_state.context
    
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.markdown("#### Raw Extracted Features")
        features_df = pd.DataFrame([context["features"]]).T
        features_df.columns = ["Value"]
        st.dataframe(features_df, width="stretch")
        
    with c2:
        st.markdown("#### Machine Learning Explainability (SHAP)")
        st.markdown(context["analysis"]["explanation"])

