import streamlit as st
import json
from utils.retriever import Retriever
from utils.generator import TimelineGenerator
from streamlit_timeline import timeline


def interactive_chatbot():
    st.header("Timeline summarization via RAG 💬")
    # Placeholder for chatbot interaction
    
    # Clear Chat History button (available on both pages)
    if st.sidebar.button('🗑️ Clear Chat History', use_container_width=True):
        st.session_state.messages = []
        st.session_state["db"] = None

    EMBEDDING_MODEL = "nomic-embed-text"
    PATH = "data/data-news"
    retriever = Retriever(st.session_state["db"])

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
            
            retrieved_df = retriever.get_similarity_search(prompt)
            generator = TimelineGenerator()
            summarized_list = generator.get_summary(retrieved_df, prompt)
            timeline_data = generator.get_timeline_data(summarized_list, prompt)
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

