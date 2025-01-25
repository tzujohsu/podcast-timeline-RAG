import streamlit as st
from utils.document_loader import DocumentLoader
import os

def sidebar():
    st.sidebar.header("Global Setup")
    folder_path = st.sidebar.text_input("Enter the folder path:", "data/data-news")

    # if st.sidebar.button("Process Data"):
    #     if uploaded_files:
    #         documents = load_documents(uploaded_files)
    #         # Further processing and storing in RAG database
    #     else:
    #         st.sidebar.warning("Please upload at least one text file.")

    if folder_path:
        if not os.path.isdir(folder_path):
            st.error("The provided path is not a valid directory. Please enter a valid folder path.")
        else:
            if st.sidebar.button("🗂️ Process Data", use_container_width=True, type='primary'):
                if "db" not in st.session_state:
                    with st.spinner("Creating embeddings and loading documents into Chroma..."):
                        document_loader = DocumentLoader()
                        st.session_state["db"] = document_loader.load_documents_into_database(folder_path)
                    st.info("All set to analyze and retrieve!")
    else:
        st.warning("Please enter a folder path to load documents into the database.")
    return folder_path
