from datetime import datetime
from openai import OpenAI
import os
import json
from dotenv import load_dotenv
import requests
import streamlit as st
import re

# helper function
def insert_br_in_pattern(input_string):
    # Define the regex pattern to match ". 2." or ". 3."
    pattern = re.compile(r'(^|\n)\s*(?!1\.)(\d+)\.')
    
    # Replace the matched pattern with ". <br> 2." or ". <br> 3."
    result = pattern.sub(r'\1<br> <br> \2.', input_string)
    return result

class HuggingfaceTimelineGenerator:
    def __init__(self, model_name='mistralai/Mistral-7B-Instruct-v0.3'):
        self.system_message = ""
        with open('timeline_template.json', "r") as f:
            self.timeline_template = json.load(f)
        # 2025 07 04 update
        self.API_URL = f"https://router.huggingface.co/novita/v3/openai/chat/completions"
        self.headers = {"Authorization": 'Bearer ' + st.secrets["HUGGINGFACE_TOKEN3"]}
        
        self.prompt = """Summarize the following content that's DIRECTLY related to {0} into 3 full sentences ONLY and nothing else. 
                        Be concise and focus on the fact. Make sure the summarization is connected to the {0}.
                        Add html line break "<br>" between each sentence, Return as 1. <br> 2. <br> 3. <br> \n [Content] """

    def get_summary(self, retrieved_df, user_input):
        content_df = retrieved_df.groupby('date')['content'].apply(
            lambda x: ' ==== next segment ==== '.join(x)).reset_index()
        
        summarized_list = []
        for i, v in content_df.iterrows():
            date, context = v.iloc[0], v.iloc[1]
            
            payload = { "messages": [
                            {
                                "role": "user",
                                "content": self.prompt.format(user_input) + context + ' [Result] '
                            }
                        ],
                        "model": "mistralai/mistral-7b-instruct",}
            response = requests.post(self.API_URL, headers=self.headers, json=payload)

            if response.status_code != 200:
                st.error(response.json())
                return []
            summarized_result = response.json()['choices'][0]['message']['content'].strip()
            
            try:
                summarized_results = summarized_result.split('<br>')
                summarized_result = '<br> <br> '.join([result.strip() for result in summarized_results])
            except:
                pass

            if '<br>' not in summarized_result:
                summarized_result = insert_br_in_pattern(summarized_result)
            date = str(date)
            date = date[:4]+'-'+date[4:4+2]+'-'+date[6:]
            summarized_list.append({'date': datetime.strptime(date, '%Y-%m-%d'), 'content': summarized_result})

        return summarized_list

    def get_timeline_data(self, summarized_list, user_input):

        events = []
        for item in summarized_list:
            date, text = str(item['date']).split()[0], (str(item['content'])).strip()
            events.append({
                'title':date,
                'description':text
            })
        return events
        

class OpenAITimelineGenerator:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.system_message = ""
        with open('timeline_template.json', "r") as f:
            self.timeline_template = json.load(f)

    def get_timeline_data(self, summarized_list, user_input):
        timeline = self.timeline_template.copy()
        timeline['title']['text']['text'] = f"Timeline of events: {user_input}"
        timeline['events'] = []
        for item in summarized_list:
            date, text = item['date'], str(item['content'])
            timeline['events'].append({
                "start_date": {
                    "year": date.year,
                    "month": date.month,
                    "day": date.day,
                    "minute": 0,
                    "second": 0,
                    "microsecond": 0
                },
                "text": {
                    "headline": f"Events on {str(date).split(' ')[0]}",
                    "text": f" {text} "
                }
            })
        return timeline

    def get_summary(self, retrieved_df, user_input):
        content_df = retrieved_df.groupby('date')['content'].apply(
            lambda x: ' ==== next segment ==== '.join(x)).reset_index()
        summarized_list = []
        query = f"""Please summarize the content that is directly related to '{user_input}' into at most 4 straightforward bullet points.
                    If the content is not really related to '{user_input}', please return NA. These content are podcast news transcripts.
                    Add <br> between each bullet point. Be concise, do not return irrelevant explanations"""
        
        for i, v in content_df.iterrows():
            date, context = v[0], v[1]
            prompt_messages = self.prepare_messages(query, context, [], f"summarize the content related to '{user_input}' into bullet points")
            completion = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=prompt_messages
            )
            model_response = completion.choices[0].message.content
            if model_response != 'NA':
                summarized_list.append({'date': datetime.strptime(date, '%Y-%m-%d'), 'content': model_response})

        return summarized_list

    def prepare_messages(self, query, context, conversation, system_message):
        messages = [{"role": "system", "content": system_message}]
        if conversation:
            for message in conversation:
                messages.append({"role": message.type, "content": message.content})

        messages.append({
            "role": "user",
            "content": f"Follow this instruction: '{query}' with this provided context: {context}",
        })

        return messages
