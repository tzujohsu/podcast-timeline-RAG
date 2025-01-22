import streamlit as st
import os

from document_loader import *
from retriever import *
from generator import *
from streamlit_timeline import timeline


#%% CONSTANTS
EMBEDDING_MODEL = "nomic-embed-text"
PATH = "data/data-news"

#%% Streamlit app title
st.set_page_config(layout="wide")
st.title("Timeline summarization via RAG 💬")


#%% Streamlit sidebar

# Subheader
st.sidebar.subheader('Models and parameters')

# Folder selection
folder_path = st.sidebar.text_input("Enter the folder path:", PATH)

if folder_path:
    if not os.path.isdir(folder_path):
        st.error(
            "The provided path is not a valid directory. Please enter a valid folder path."
        )
    else:
        if st.sidebar.button("🗂️ Index Documents", use_container_width=True, type='primary'):
            if "db" not in st.session_state:
                with st.spinner(
                    "Creating embeddings and loading documents into Chroma..."
                ):
                    st.session_state["db"] = load_documents_into_database(folder_path)
                st.info("All set to retrieve!")
else:
    st.warning("Please enter a folder path to load documents into the database.")




# Clear Chat History
def clear_chat_history():
    st.session_state.messages = []
    st.session_state["db"] = None
st.sidebar.button('🗑️ Clear Chat History ', on_click=clear_chat_history, use_container_width = True)

#%% Main Content

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

with open(f'timeline.json', "r") as f:
    data = f.read()


if prompt := st.chat_input("Question"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        retrieved_df = get_similarity_search(st.session_state["db"], prompt)
        summarized_list = get_summary(retrieved_df, prompt)
        timeline_data = get_timeline_data(summarized_list, prompt)
        timeline_data = json.dumps(timeline_data)

        
        container =  st.container()
        with st.container():
            st.write(f"Here is a timeline for {prompt}")
            timeline(timeline_data, height=600)

        st.session_state.messages.append({"role": "assistant", "content": f"A timeline for the events related to your prompt '{prompt}'."})
