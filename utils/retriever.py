import pandas as pd


# def post_retrieval_processing(results):
#     data = [{
#         'date': res.metadata['date'],
#         'content': res.page_content[14:-1],
#         'segment': res.metadata['segment'],
#         'chunk': res.metadata['chunk'],
#         'relevancy_rank': index + 1 # Relevancy rank is based on the order in the list
#     } for index, res in enumerate(results)]
#     df = pd.DataFrame(data)
    
#     df = (df.sort_values(by='relevancy_rank', ascending=True)
#         .groupby('date')
#         .head(3)
#         .sort_values(by=['date','segment','chunk', 'relevancy_rank'])
#         .reset_index(drop=True))
    
#     return df

# def get_similarity_search(vector_store, input):
#     results = vector_store.similarity_search(
#         input,
#         k=30,
#     )
    
#     post_df = post_retrieval_processing(results)

#     return post_df


import pandas as pd

class Retriever:
    def __init__(self, vector_store):
        self.vector_store = vector_store

    def post_retrieval_processing(self, results):
        data = [{
            'date': res.metadata['date'],
            'content': res.page_content[14:-1],
            'segment': res.metadata['segment'],
            'chunk': res.metadata['chunk'],
            'relevancy_rank': index + 1
        } for index, res in enumerate(results)]
        df = pd.DataFrame(data)
        
        df = (df.sort_values(by='relevancy_rank', ascending=True)
            .groupby('date')
            .head(3)
            .sort_values(by=['date', 'segment', 'chunk', 'relevancy_rank'])
            .reset_index(drop=True))
        
        return df

    def get_similarity_search(self, input):
        results = self.vector_store.similarity_search(input, k=30)
        post_df = self.post_retrieval_processing(results)
        return post_df
