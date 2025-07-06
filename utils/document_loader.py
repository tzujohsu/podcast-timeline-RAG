import os
from langchain_openai import OpenAIEmbeddings
from typing import List
from langchain_core.documents import Document
from uuid import uuid4
import pandas as pd
from bs4 import BeautifulSoup
import requests
from tqdm import tqdm
from datetime import datetime, timedelta
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def get_html_and_parse(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes

        soup = BeautifulSoup(response.content, "html.parser")
        return soup
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL: {e}")
        return None
    
# 
def get_transcripts(start_date, end_date=None, latest=True):
    """
    Scrape CNN transcripts for a given date range.

    Args:
        start_date (datetime): The starting date for scraping transcripts.
        end_date (datetime, optional): The ending date for scraping transcripts. Required if `latest` is False.
        latest (bool, optional): If True, only scrape transcripts for `start_date`. If False, scrape between `start_date` and `end_date`.

    Returns:
        list: A list of transcript contents.
    """
    base_url = 'https://transcripts.cnn.com'
    transcript_contents = []
    
    current_date = start_date
    while True:
        # Construct the URL for the current date
        transcript_list_url = f'{base_url}/show/cnc?start_fileid=cnc_{current_date.strftime("%Y-%m-%d")}'
        soup = get_html_and_parse(transcript_list_url)
        
        # Find all transcript links on the page
        transcript_links = []
        for link in soup.find_all('a'):
            if 'Did Not Air' not in str(link) and 'date/202' in str(link) and 'Next Page' not in str(link):
                
                transcript_date = link.attrs['href'].split('date')[1].split('/')[1]

                if latest and transcript_date != (current_date - timedelta(days=1)).strftime('%Y-%m-%d'):
                    continue
                if not latest and (transcript_date > start_date.strftime('%Y-%m-%d') or transcript_date < end_date.strftime('%Y-%m-%d')):
                    continue
                
                transcript_links.append(link)
        
        # Scrape the content for each transcript link
        for link in transcript_links:
            transcript_url = f"{base_url}{link.attrs['href']}"
            transcript_soup = get_html_and_parse(transcript_url)
            transcript_text = (transcript_soup.find_all('p')[-1]).text
            transcript_contents.append((transcript_url, transcript_text))
        
        # If `latest` is True, stop after scraping the first page
        if latest:
            break
        
        # Find the next page link
        next_page_link = soup.find_all('a')[-1].attrs['href']
        next_page_date = next_page_link.split('date')[0].split('_')[-2]
        next_page_date = datetime.strptime(next_page_date, '%Y-%m-%d')
        
        # Stop if the next page's date exceeds the `end_date`
        if next_page_date < end_date:
            break
        
        current_date = next_page_date
    print("\tstart date", start_date, "end date", current_date)
    
    return transcript_contents # [(url, content), ...]

class DocumentLoader:
    def __init__(self, 
                 base_path = 'data/data-news/',
                 model="all-MiniLM-L6-v2", 
                 collection_name="rag-podcast-poc", 
                 persist_directory="./chroma_langchain_db"):
        
        if model == 'text-embedding-3-small':
            self.embeddings = OpenAIEmbeddings(model=model, openai_api_key=os.getenv("OPENAI_API_KEY"))
        else:
            self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.base_path = base_path
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=150,
            length_function=len,
            is_separator_regex=False,
        )

    def load_documents_from_path(self, file_paths: list) -> List[Document]:
        docs = []
        ii = 1
        # for file in os.listdir(path):
        for file in file_paths:
            if  '.DS_Store' != file:
                with open(self.base_path + file, 'r', encoding='latin-1') as f:
                    content = f.read()
                contents = self.text_splitter.create_documents([content])
                for chunkid, content in enumerate(contents):
                    docs.append(Document(
                        page_content=f"{content}",
                        metadata={"date": int(file[:10].replace("-", "")),
                                    "segment": file.split('-')[-1][:-4],
                                    "chunk": chunkid+1},
                        id=ii,
                    ))
                    ii += 1
        self.get_database_dates()
        return docs
    
    def load_documents_from_scraping(self, start_date, end_date=None, latest=True) -> List[Document]:
        transcript_objects = get_transcripts(start_date, end_date, latest)
        print("\tget transcripts done")
        docs = []
        ii = 1
        for i, objects in enumerate(transcript_objects):
                url, content = objects[0], objects[1]
                date = int((url.split('date/')[1].split('/')[0]).replace("-", ""))
                segment = int(url.split('/')[-1])
                contents = self.text_splitter.create_documents([content])
                for chunkid, content in enumerate(contents):
                    docs.append(Document(
                        page_content=f"{content}",
                        metadata={"date": date,
                                    "segment": segment,
                                    "chunk": chunkid+1},
                        id=ii,
                    ))
                    ii += 1
        self.get_database_dates()
        return docs

    def load_documents_into_database(self, documents: list = None):
        if len(documents) != 0:
            uuids = [str(uuid4()) for _ in range(len(documents))]
            self.vector_store.add_documents(documents=documents, ids=uuids)
        else:
            return -1

        return self.vector_store

    def remove_documents(self, cutoff_threshold: int = 30):        
        cutoff_date = int((datetime.now() - timedelta(days=cutoff_threshold)).strftime('%Y%m%d'))
        print("before delete # documents: ", len(self.vector_store.get()['documents']))
        results = self.vector_store.get(where={"date": {"$lt": cutoff_date}})
        if results['ids']:
            self.vector_store.delete(ids=results['ids'])
        print("\tafter deletion # documents: ", len(self.vector_store.get()['documents']))
        self.get_database_dates()

    
    def get_database_dates(self):
        """
        return the range of dates of articles stored in the database
        """
        today_date = int((datetime.now()).strftime('%Y%m%d'))
        # https://python.langchain.com/docs/integrations/vectorstores/faiss/
        results = self.vector_store.get(where={"date": {"$lt": today_date}})['metadatas']
        dates = set()
        for result in results:
            dates.add(result['date'])
        try:
            if dates:
                print('\tdate range:', min(dates), ' - ', max(dates))
                min_date = datetime.strptime(str(min(dates)), "%Y%m%d").strftime("%m/%d/%Y")
                max_date = datetime.strptime(str(max(dates)), "%Y%m%d").strftime("%m/%d/%Y")
                return min_date, max_date
        except:
            pass
# if __name__ == "__main__":
#     start_date = datetime(2025, 2, 25)
#     end_date = datetime(2025, 2, 15)
#     transcripts = get_transcripts(start_date, end_date, latest=False)
#     print(len(transcripts))