# streamlit sqlite3 dependency handling
# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from streamlit.components.v1 import html
import os
import re
import json
import random

from utils.retriever import Retriever
from utils.generator import HuggingfaceTimelineGenerator
from utils.document_loader import DocumentLoader
from utils.components import timeline_css, generate_timeline_html
from streamlit_timeline import timeline

# Set page config
st.set_page_config(
    page_title="ChronoEvents: Timeline Generator",
    page_icon=":hourglass_flowing_sand:",
    layout="wide"
)

# Initialize necessary components
if "db" not in st.session_state:
    docloader = DocumentLoader()
    st.session_state["docloader"] = docloader
    print("Database loaded successfully")

st.markdown("""
<style>
.highlight {
    background-color: rgba(0,80,142,0.15);
    padding: 0 0.4rem;
    border-radius: 0.25rem;
    font-weight: bold;
    line-height: 1.5;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: black;'>ChronoEvents: Timeline Generator</h1>", unsafe_allow_html=True)

st.markdown(
    """
    <div style="width: 60%; margin: 0 auto; text-align: center; font-size: 1.1rem;">
        <h6>
        An innovative LLM/RAG-powered tool that transforms news transcripts into structured, temporal insights.
        </h6>
    </div>
    """, 
    unsafe_allow_html=True
)

min_date, max_date = st.session_state["docloader"].get_database_dates()
st.markdown(f"<h6 style='text-align: center; color: grey;'>Data Source: CNN News Central transcripts ({min_date} - {max_date})</h6>", unsafe_allow_html=True)

st.markdown(
    """
    <div style="width: 60%; margin: 0 auto; text-align: center; font-size: 1rem;">
        <br>
        <p>
        Simply enter your topic in the input box below, and the tool will: <br>
        1. Retrieve relevant information from the <span class='highlight'> LangChain Chroma </span> database <br>
        2. Generate a summarization of all the retrieved events with <span class='highlight'> Mistral-7b </span> <br>
        3. Display a timeline visualization for each date and event description <br>
        </p>
    </div>
    """, 
    unsafe_allow_html=True
)
    
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# Helper function to clear text
def clear_text():
    st.session_state.user_input = ""

# Helper function to generate display sample
sample_inputs = ['tariff', 'plane crashes', 'trade war', 'Artificial Intelligence', 'New York', 'visa revoke']
def generate_random_sample():
    random_sample = random.choice(sample_inputs)
    st.session_state.user_input = random_sample

retriever = Retriever(st.session_state["docloader"].vector_store)
generator = HuggingfaceTimelineGenerator()

def st_normal():
    _, col, _ = st.columns([1, 3, 1])
    return col

# Input section
with st_normal():
    # Input area with placeholder text
    user_input = st.text_area(
        "Enter your topic or query:",
        placeholder="Describe the topic or events you'd like to create a timeline for...",
        height=70,
        key="user_input"
            )

    col1, col2, col3 = st.columns([4, 1, 1])

    with col1:
        # Generate Random Sample button
        st.button("Random topics", on_click=generate_random_sample, type="secondary")

    with col2:
        # Clear Input button
        st.button("Clear Input", on_click=clear_text, type="secondary")

    with col3:
        # Generate timeline button
        generate_button = st.button("Generate", type="primary")

    # Results section
    if generate_button and user_input:
        with st.spinner("Generating timeline..."):
            # Retrieve relevant documents
            retrieved_df = retriever.get_similarity_search(user_input)
            
            # Generate timeline
            summarized_list = generator.get_summary(retrieved_df, user_input=user_input)
            timeline_data = generator.get_timeline_data(summarized_list, user_input)
            

            char_sum = 0
            for event in timeline_data:
                _, description = event['title'], event['description']
                char_sum += len(description)
            
            html(generate_timeline_html(timeline_data), height=350*char_sum//500+300)  # Increased height for better scrolling 

    st.markdown("---")

def st_normal():
    _, col, _ = st.columns([1, 8, 1])
    return col

with st_normal():
    # Title
    st.markdown("""
        <h2 style='text-align: center; margin-bottom: 0.5rem;'>Project Overview</h2>
        """, unsafe_allow_html=True)

    # Motivation Section
    st.markdown("""
    <div class='section'>

    #### 💡 Motivation
    Navigating websites or documents with numerous timestamps can make it difficult to grasp their chronological flow. 
    While chatbots like ChatGPT excel at conversation, they struggle to extract and present structured insights from large volumes of text. 
    This project addresses that gap by generating meaningful timelines from content such as news transcripts and podcasts, helping users visualize how events evolve over time. 
    Currently, it features a timeline summarizer, with future plans to incorporate sentiment and trend analysis.
    </div>
    """, unsafe_allow_html=True)

    # Dataset Section
    st.markdown(f"""
    <div class='section'>

    #### 📚 Data Retrieval and Storage
    The CNN News Central transcripts are vectorized with **all-MiniLM-L6-v2** and stored in Chroma database. 
    The data pipeline can be configured to scrape transcripts daily using **GitHub Actions** (optional).
    Given the limitations of the free tier, I keep the time range of the transcripts within approximately a month.
    The current date range is within {min_date} - {max_date}. (Note that there are some gaps on days when the podcast wasn't aired.)
    </div>
    """, unsafe_allow_html=True)

    # System Design Section
    st.markdown("""
    <div class='section'>

    #### ⚙️ System Design
    <div class='tech-item'>
    <strong>Large Language Models:</strong> LangChain Chroma, Recursive Character Text Splitter, Huggingface Embeddings, Huggingface Mistral-7b
    </div>

    <div class='tech-item'>
    <strong>Frontend:</strong> Streamlit, HTML, CSS
    </div>

    <div class='tech-item'>
    <strong>Data retrieval:</strong> BeautifulSoup (GitHub Actions optional)
    </div>

    </div>
    """, unsafe_allow_html=True)

    # Disclaimer Section
    st.markdown("""
    ---
    <div class='section'>

    😉 Disclaimer: The results generated by this model and the news content displayed are for demonstration purposes only and do not reflect my personal opinions and should not be used as a sole source of information.
    </div>
    """, unsafe_allow_html=True)
