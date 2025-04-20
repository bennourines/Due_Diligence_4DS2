# app.py - Streamlit frontend
import streamlit as st
import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from io import BytesIO

# Define API URL
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Set page config
st.set_page_config(
    page_title="Crypto Due Diligence Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem !important;
        font-weight: 600 !important;
        color: #1E3A8A !important;
    }
    .sub-header {
        font-size: 1.5rem !important;
        font-weight: 500 !important;
        color: #3B82F6 !important;
    }
    .risk-high {
        color: #DC2626 !important;
        font-weight: 600 !important;
    }
    .risk-medium {
        color: #F59E0B !important;
        font-weight: 600 !important;
    }
    .risk-low {
        color: #10B981 !important;
        font-weight: 600 !important;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #E5E7EB;
    }
    .bot-message {
        background-color: #app.mongodbEAFE;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'project_id' not in st.session_state:
    st.session_state.project_id = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'risk_report' not in st.session_state:
    st.session_state.risk_report = None
if 'user_id' not in st.session_state:
    st.session_state.user_id = f"user_{datetime.now().strftime('%Y%m%d%H%M%S')}"

# Helper functions
def upload_document(file):
    """Upload document to API"""
    files = {"file": (file.name, file.getvalue(), file.type)}
    response = requests.post(
        f"{API_URL}/upload/",
        files=files,
        params={"user_id": st.session_state.user_id}
    )
    
    if response.status_code == 201:
        return response.json()
    else:
        st.error(f"Error uploading document: {response.text}")
        return None

def query_documents(query):
    """Query documents via API"""
    payload = {
        "project_id": st.session_state.project_id,
        "user_id": st.session_state.user_id,
        "query": query
    }
    
    response = requests.post(f"{API_URL}/query/", json=payload)
    
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error querying documents: {response.text}")
        return None

def generate_risk_report():
    """Generate risk report via API"""
    payload = {
        "project_id": st.session_state.project_id,
        "user_id": st.session_state.user_id,
        "query": "Generate a comprehensive risk report"
    }
    
    response = requests.post(f"{API_URL}/analyze/", json=payload)
    
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error generating risk report: {response.text}")
        return None

def get_chat_history():
    """Get chat history from API"""
    response = requests.get(
        f"{API_URL}/history/{st.session_state.project_id}"
    )
    
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error retrieving chat history: {response.text}")
        return []

def display_chat_message(role, content):
    """Display chat message with proper formatting"""
    if role == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>You:</strong><br>{content}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message bot-message">
            <strong>Assistant:</strong><br>{content}
        </div>
        """, unsafe_allow_html=True)

def plot_risk_radar_chart(risk_report):
    """Generate radar chart from risk report"""
    categories = []
    scores = []
    
    for category, data in risk_report["categories"].items():
        categories.append(category.capitalize())
        scores.append(data["score"])
    
    # Create radar chart with Plotly
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=scores,
        theta=categories,
        fill='toself',
        name='Risk Score',
        line_color='#3B82F6'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=False
    )
    
    return fig

def generate_risk_heatmap(risk_report):
    """Generate risk heatmap from risk report"""
    categories = []
    scores = []
    risk_levels = []
    
    for category, data in risk_report["categories"].items():
        for answer in data["answers"]:
            categories.append(category.capitalize())
            scores.append(answer["score"])
            if answer["score"] < 40:
                risk_levels.append("High")
            elif answer["score"] < 70:
                risk_levels.append("Medium")
            else:
                risk_levels.append("Low")
    
    # Create DataFrame
    df = pd.DataFrame({
        "Category": categories,
        "Score": scores,
        "Risk Level": risk_levels
    })
    
    # Create heatmap with Plotly
    fig = px.treemap(
        df, 
        path=['Category', 'Risk Level'], 
        values='Score',
        color='Score',
        color_continuous_scale='RdYlGn',
        color_continuous_midpoint=50
    )
    
    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    return fig

# Main app interface
def main():
    # Header
    st.markdown("<h1 class='main-header'>Crypto Due Diligence Assistant</h1>", unsafe_allow_html=True)
    st.markdown("Upload cryptocurrency documentation for automated risk analysis and due diligence.")
    
    # Sidebar
    with st.sidebar:
        st.markdown("<h2 class='sub-header'>Project Controls</h2>", unsafe_allow_html=True)
        
        # Document upload section
        st.subheader("Document Upload")
        uploaded_file = st.file_uploader("Upload crypto documentation", 
                                        type=["pdf", "docx", "xlsx", "txt", "md"],
                                        help="Upload whitepapers, audit reports, tokenomics docs, etc.")
        
        if uploaded_file is not None:
            if st.button("Process Document"):
                with st.spinner("Processing document..."):
                    result = upload_document(uploaded_file)
                    if result:
                        st.session_state.project_id = result["project_id"]
                        st.success(f"Document processed successfully! Project ID: {st.session_state.project_id}")
                        
                        # Clear existing report if any
                        st.session_state.risk_report = None
        
        # Generate risk report
        if st.session_state.project_id:
            st.divider()
            st.subheader("Risk Analysis")
            if st.button("Generate Risk Report"):
                with st.spinner("Analyzing documents and generating risk report..."):
                    risk_report = generate_risk_report()
                    if risk_report:
                        st.session_state.risk_report = risk_report
                        st.success("Risk report generated successfully!")

    # Main content area
    if st.session_state.project_id:
        # Create tabs for different views
        tab1, tab2 = st.tabs(["Chat Interface", "Risk Report"])
        
        # Tab 1: Chat Interface
        with tab1:
            st.markdown("<h2 class='sub-header'>Chat with Your Documents</h2>", unsafe_allow_html=True)
            
            # Display chat history
            chat_container = st.container()
            with chat_container:
                for message in st.session_state.chat_history:
                    display_chat_message("user", message["query"])
                    display_chat_message("assistant", message["response"])
            
            # Chat input
            st.divider()
            with st.form(key="chat_form", clear_on_submit=True):
                user_input = st.text_area("Ask a question about the crypto asset:", height=100)
                submit_button = st.form_submit_button("Send")
                
                if submit_button and user_input:
                    with st.spinner("Generating response..."):
                        # Add user message to chat history
                        st.session_state.chat_history.append({"query": user_input, "response": ""})
                        
                        # Get response from API
                        response = query_documents(user_input)
                        if response:
                            # Update chat history
                            st.session_state.chat_history[-1]["response"] = response["response"]
                            
                            # Rerun to update UI
                            st.rerun()
        
        # Tab 2: Risk Report
        with tab2:
            st.markdown("<h2 class='sub-header'>Comprehensive Risk Analysis</h2>", unsafe_allow_html=True)
            
            if st.session_state.risk_report:
                report = st.session_state.risk_report
                
                # Overall Score and Recommendation
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    # Display score gauge
                    score = report["overall_score"]
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=score,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "#3B82F6"},
                            'steps': [
                                {'range': [0, 40], 'color': "#FECACA"},
                                {'range': [40, 70], 'color': "#FEF3C7"},
                                {'range': [70, 100], 'color': "#D1FAE5"}
                            ]
                        },
                        title={'text': "Overall Risk Score"}
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Display summary and recommendation
                    st.subheader("Executive Summary")
                    st.write(report["summary"])
                    
                    # Recommendation with color based on score
                    st.subheader("Recommendation")
                    if score >= 80:
                        st.markdown(f"<p class='risk-low'>{report['recommendation']}</p>", unsafe_allow_html=True)
                    elif score >= 60:
                        st.markdown(f"<p class='risk-medium'>{report['recommendation']}</p>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<p class='risk-high'>{report['recommendation']}</p>", unsafe_allow_html=True)
                
                # Key findings
                st.subheader("Key Findings")
                for finding in report["key_findings"]:
                    if "High Risk" in finding:
                        st.markdown(f"<p class='risk-high'>⚠️ {finding}</p>", unsafe_allow_html=True)
                    elif "Strength" in finding:
                        st.markdown(f"<p class='risk-low'>✅ {finding}</p>", unsafe_allow_html=True)
                    else:
                        st.write(finding)
                
                # Risk visualizations
                st.subheader("Risk Profile")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Radar chart
                    st.plotly_chart(plot_risk_radar_chart(report), use_container_width=True)
                
                with col2:
                    # Risk heatmap
                    st.plotly_chart(generate_risk_heatmap(report), use_container_width=True)
                
                # Detailed category breakdown
                st.subheader("Detailed Analysis by Category")
                
                for category, data in report["categories"].items():
                    with st.expander(f"{category.capitalize()} - Score: {data['score']:.1f}/100"):
                        # Display risk level
                        if data["risk_level"] == "High":
                            st.markdown(f"<p class='risk-high'>Risk Level: {data['risk_level']}</p>", unsafe_allow_html=True)
                        elif data["risk_level"] == "Medium":
                            st.markdown(f"<p class='risk-medium'>Risk Level: {data['risk_level']}</p>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<p class='risk-low'>Risk Level: {data['risk_level']}</p>", unsafe_allow_html=True)
                        
                        # Display individual question answers
                        for answer in data["answers"]:
                            st.write(f"**Question:** {answer['question']}")
                            st.write(f"**Answer:** {answer['answer']}")
                            
                            # Score with color
                            score = answer["score"]
                            if score < 40:
                                st.markdown(f"<p class='risk-high'>Score: {score:.1f}/100 (Weight: {answer['weight']}%)</p>", unsafe_allow_html=True)
                            elif score < 70:
                                st.markdown(f"<p class='risk-medium'>Score: {score:.1f}/100 (Weight: {answer['weight']}%)</p>", unsafe_allow_html=True)
                            else:
                                st.markdown(f"<p class='risk-low'>Score: {score:.1f}/100 (Weight: {answer['weight']}%)</p>", unsafe_allow_html=True)
                            
                            st.divider()
            else:
                st.info("No risk report available. Generate a risk report using the button in the sidebar.")
    else:
        # Welcome screen
        st.markdown("""
        ## Welcome to Crypto Due Diligence Assistant!
        
        This tool helps you analyze cryptocurrency projects by processing documents like:
        - Whitepapers
        - Tokenomics documentation
        - Smart contract audit reports
        - Team backgrounds
        - Regulatory filings
        
        ### How it works:
        1. **Upload Documents**: Start by uploading relevant documents using the sidebar
        2. **Chat with Documents**: Ask specific questions about the project
        3. **Generate Risk Report**: Get a comprehensive analysis of risks and opportunities
        
        ### Key Features:
        - **Hybrid Search**: Combines semantic and keyword search for accurate results
        - **Risk Analysis**: Evaluates projects across tokenomics, technical, team, regulatory, and market dimensions
        - **Visual Reports**: See easy-to-understand visualizations of risk factors
        
        Get started by uploading a document in the sidebar!
        """)

if __name__ == "__main__":
    main()