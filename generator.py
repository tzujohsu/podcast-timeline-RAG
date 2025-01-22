from datetime import datetime
from openai import OpenAI
import os
import json
from dotenv import load_dotenv
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) # Add your OpenAI API key here

system_message = ""

with open(f'timeline_template.json', "r") as f:
    timeline_template = f.read()
timeline_template = json.loads(timeline_template)


def get_timeline_data(summarized_list: list, user_prompt: str):
    timeline = timeline_template.copy()
    timeline['title']['text']['text'] = f"Timeline of events: {user_prompt}"
    timeline['events'] = []
    for item in summarized_list:
        date, text = item['date'], str(item['content'])
        timeline['events'].append(
            {
                # "media": {
                #     "url": "https://picsum.photos/200/300",
                #     "caption": "Photo insert"
                # },
                "start_date": {
                    "year":date.year,
                    "month":date.month,
                    "day":date.day, 
                    "minute": 0,
                    "second": 0,
                    "microsecond":  000000
                },
                "text": {
                    "headline": f"Events on {str(date).split(' ')[0]}",
                    "text": f" {text} "
                }
            }
        )
    return timeline
          

def get_summary(retrieved_df, user_prompt):
    """
        retrieved_df: pd.DataFrame
            date	    content	                                        segment	chunk	relevancy_rank
        0	2024-12-20	page_content='getting an early start to the ho...	02	10	32
        2	2024-12-20	page_content='would have been the time that th...	12	4	31
        3	2024-12-23	page_content='some delays. And in California, ...	02	8	28
        4	2024-12-23	page_content='This area, in particular, Tuesda...	02	9	
     """
    
    content_df = retrieved_df.groupby('date')['content'].apply(
            lambda x: ' ==== next segment ==== '.join(x)).reset_index()
    summarized_list = []
    query = f"""Please summarize the content that is directly related to '{user_prompt}' into at most 4 straightforward bullet points.
                If the content is not really related to '{user_prompt}', please return NA. These content are podcast news transcripts.
                Add <br> between each bullet point. Be concise, do not return irrelevant explanations"""
    
    for i, v in content_df.iterrows():
        date, context = v[0], v[1]
        prompt_messages = prepare_messages(query, context, [], f"summarize the content related to '{user_prompt}' into bullet points")
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=prompt_messages
        )
        model_response = completion.choices[0].message.content
        if model_response != 'NA':
            summarized_list.append({'date': datetime.strptime(date, '%Y-%m-%d'), 'content': model_response})


    return summarized_list

def prepare_messages(
        query: str, context: str, conversation: list[dict], system_message: str
    ) -> list[dict]:
        messages = [
            {
                "role": "system",
                "content": system_message,
            }
        ]
        if conversation:
            for message in conversation:
                messages.append({"role": message.type, "content": message.content})

        messages.append(
            {
                "role": "user",
                "content": f"Follow this instruction: '{query}' with this provided context: {context}",
            }
        )

        return messages