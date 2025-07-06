import logging
from utils.document_loader import *
from utils.retriever import Retriever

#%% setup config
today_date = datetime.now()
#%%
# step 1
docloader = DocumentLoader()
print('step 1: load the vector database done')
print('\t# docs in the current database: ', len(docloader.vector_store.get()['documents']))

# step 2: 
documents = docloader.load_documents_from_scraping(start_date=today_date , latest=True)
# documents = docloader.load_documents_from_scraping(start_date=(datetime.now() - timedelta(days=2)), end_date= (datetime.now() - timedelta(days=32)), latest=False)
print('step 2: scrape the latest news done')

# step 3:
ret = docloader.load_documents_into_database(documents)
if ret != -1:
    print('step 3: load the latest documents into db done')
else:
    print('step 3: no new documents to load')
print('\t# docs in the current database: ', len(docloader.vector_store.get()['documents']))

# step 4
docloader.remove_documents(cutoff_threshold=32)
print('step 4: remove the oldest transcripts done')
print('\t# docs in the current database: ', len(docloader.vector_store.get()['documents']))