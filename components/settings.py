import streamlit as st

def settings():
    st.sidebar.header("Settings")
    openai_key = st.sidebar.text_input("OpenAI API Key", type="password")
    chunk_size = st.sidebar.number_input("Chunk Size", min_value=100, max_value=2000, step=100, value=500)
    embedding_model = st.sidebar.selectbox("Embedding Model", ["openai", "sentence-transformers"])
    if st.sidebar.button("Save Settings"):
        st.session_state['openai_key'] = openai_key
        st.session_state['chunk_size'] = chunk_size
        st.session_state['embedding_model'] = embedding_model
        st.sidebar.success("Settings saved.")