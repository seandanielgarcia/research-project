import os
import pandas as pd
import google.generativeai as genai
from topicgpt import TopicGPT
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("gemini_key"))


def gemini_call(prompt):
    response = genai.GenerativeModel("gemini-2.0-flash").generate_content(prompt)
    return response.text

df = pd.read_csv("data/combined_posts.csv")
docs = df["Summary"].dropna().tolist()

tgpt = TopicGPT(
    model=gemini_call,
    embedding_model="sentence-transformers/all-MiniLM-L6-v2"
)

topics = tgpt.get_topics(docs)

print(topics)

import json
with open("results/clustering/topicgpt/topics.json", "w") as f:
    json.dump(topics, f, indent=2)
