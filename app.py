import streamlit as st
import os
from document_loader import *
from retriever import *
from generator import *
from streamlit_timeline import timeline
import re
import json

# Constants
EMBEDDING_MODEL = "nomic-embed-text"
PATH = "data/data-news"

# Page configuration
st.set_page_config(layout="wide")

# Initialize session state for page navigation
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Generate Timeline"

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Generate Timeline", "View Text Files"])
st.session_state.current_page = page

if page == "Generate Timeline":
    st.title("Timeline summarization via RAG 💬")
    
    # Your existing timeline generation code
    st.sidebar.subheader('Models and parameters')
    folder_path = st.sidebar.text_input("Enter the folder path:", PATH)
    
    if folder_path:
        if not os.path.isdir(folder_path):
            st.error("The provided path is not a valid directory. Please enter a valid folder path.")
        else:
            if st.sidebar.button("🗂️ Index Documents", use_container_width=True, type='primary'):
                if "db" not in st.session_state:
                    with st.spinner("Creating embeddings and loading documents into Chroma..."):
                        st.session_state["db"] = load_documents_into_database(folder_path)
                    st.info("All set to retrieve!")
    else:
        st.warning("Please enter a folder path to load documents into the database.")

    # Rest of your timeline generation code
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if isinstance(message["content"], dict) and message["content"].get("type") == "timeline":
                timeline_data_json = json.dumps(message["content"]["data"])
                timeline(timeline_data_json, height=600)
            else:
                st.markdown(message["content"])

    if prompt := st.chat_input("Question"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            retrieved_df = get_similarity_search(st.session_state["db"], prompt)
            summarized_list = get_summary(retrieved_df, prompt)
            timeline_data = get_timeline_data(summarized_list, prompt)
            timeline_data_json = json.dumps(timeline_data)
            
            with st.container():
                st.write(f"Here is a timeline for {prompt}")
                timeline(timeline_data_json, height=600)
            
            # Store the timeline data in a structured format
            st.session_state.messages.append({
                "role": "assistant", 
                "content": {
                    "type": "timeline",
                    "data": timeline_data
                }
            })

elif page == "View Text Files":
    st.title("Text File Viewer")
    
    # File selection section
    file_path = st.sidebar.text_input("Enter the folder path for text files:", PATH)
    def parse_filename(filename):
        # Extract date and segment from filename like "2024-12-30-segment-06.txt"
        try:
            date_part = filename.split('-segment-')[0]
            segment = int(filename.split('-segment-')[1].replace('.txt', ''))
            year, month, day = map(int, date_part.split('-'))
            return (year, month, day, segment)
        except:
            return (0, 0, 0, 0)  # Return zeros for invalid filenames

    if file_path and os.path.isdir(file_path):
        text_files = [f for f in os.listdir(file_path) if f.endswith('.txt')]
        text_files.sort(key=parse_filename)
        if text_files:
            selected_file = st.sidebar.selectbox(
                "Select a file to view",
                text_files,
                key="file_selector"
            )
            
            try:
                with open(os.path.join(file_path, selected_file), 'r', encoding='utf-8') as file:
                    content = file.read()
                    
                # Updated pattern to split by punctuation followed by space and specific characters
                split_pattern = r"(?<=[.!?])\s(?=\[)|(?<=[.!?])\s(?=[A-Z]+:)|(?<=[.!?])\s(?=\()"
                
                # Split content into lines and format
                formatted_lines = []
                current_line = ""
                
                for line in content.split('\n'):
                    line = line.strip()
                    if not line:
                        continue
                        
                    # Split the line based on the pattern
                    parts = re.split(split_pattern, line)
                    
                    for part in parts:
                        part = part.strip()
                        if part:
                            if current_line:
                                formatted_lines.append(current_line)
                                formatted_lines.append('')  # Add empty line
                            current_line = part
                
                # Add the last line if not empty
                if current_line:
                    formatted_lines.append(current_line)
                
                formatted_content = '\n'.join(formatted_lines)
                
                # Create a container with scrollable text area
                with st.container():
                    st.markdown("### File Contents")
                    st.text_area(
                        "File Contents",  # Non-empty label
                        value=formatted_content,
                        height=500,
                        key="file_contents",
                        label_visibility="collapsed"  # Hide the label
                    )
                    
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
        else:
            st.warning("No text files found in the specified directory.")
    else:
        st.warning("Please enter a valid folder path containing text files.")

# Clear Chat History button (available on both pages)
if st.sidebar.button('🗑️ Clear Chat History', use_container_width=True):
    st.session_state.messages = []
    st.session_state["db"] = None
