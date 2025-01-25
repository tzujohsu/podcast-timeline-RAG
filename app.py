import streamlit as st
import os
# from document_loader import *
# from retriever import *
# from generator import *
import re
import json

import streamlit as st
from components.sidebar import sidebar
from components.overview_analysis import overview_analysis
from components.single_file_analysis import single_file_analysis
from components.interactive_chatbot import interactive_chatbot

def main():
    st.set_page_config(layout="wide")
    st.title("Time-Aware Text Analysis and Retrieval")

    # Sidebar for global setup
    folder_path = sidebar()

    # Page navigation
    page = st.sidebar.radio("Choose a page", ["Overview Analysis", "Single File Analysis", "Interactive Chatbot"])

    if page == "Overview Analysis":
        overview_analysis()
    elif page == "Single File Analysis":
        single_file_analysis(folder_path)
    elif page == "Interactive Chatbot":
        interactive_chatbot()

if __name__ == "__main__":
    main()

