import os
os.environ["OPENAI_API_KEY"] = '' # enter your openai api key here

if os.getenv("OPENAI_API_KEY") is not None:
    print ("OPENAI_API_KEY is ready")
else:
    print ("OPENAI_API_KEY environment variable not found")

from langchain_openai import OpenAIEmbeddings
from typing import List
from langchain_core.documents import Document
from uuid import uuid4
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_chroma import Chroma

#%%

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
# base_path = 'podcast-RAG/'


vector_store = Chroma(
    collection_name="rag-podcast-poc",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)

# https://python.langchain.com/api_reference/text_splitters/character/langchain_text_splitters.character.RecursiveCharacterTextSplitter.html
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=150,
    length_function=len,
    is_separator_regex=False,
)

def load_documents(path: str) -> List[Document]:
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"The specified path does not exist: {path}")
    
    docs = []
    ii = 1
    for file in (os.listdir(path)):
        with open(path+'/'+file, 'r', encoding='latin-1') as f:
            content = f.read()
        contents = text_splitter.create_documents([content])
        for chunkid, content in enumerate(contents):
            docs.append(Document(
                page_content=f"{content}",
                metadata={"date": file[:10]
                        , "segment": file.split('-')[-1][:-4]
                        , "chunk": chunkid+1},
                id=ii,
            ))
            ii += 1
    return docs

def load_documents_into_database(folder_path = "data/data-news", persist_directory = './chroma_langchain_db'):
    if not os.path.exists(persist_directory) or len(os.listdir(persist_directory)) < 2:
        documents = load_documents(folder_path)
        uuids = [str(uuid4()) for _ in range(len(documents))]
        
        vector_store = Chroma(
            collection_name="rag-podcast-poc",
            embedding_function=embeddings,
            persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
        )
            
        vector_store.add_documents(documents=documents, ids=uuids)
    else:
        vector_store = Chroma(collection_name="rag-podcast-poc", persist_directory=persist_directory, embedding_function=embeddings)


    return vector_store
