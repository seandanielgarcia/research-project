import os
import re
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("gemini_key"))

input_path  = "chatgpt_hallucinated_posts.txt"
output_path = "improved_prompt_summary.txt"

with open(input_path, "r", encoding="utf-8") as f:
    content = f.read()

# Split on each "=== Post ID:" header
posts = re.findall(r'(=== Post ID:.*?)(?=(?:^=== Post ID:)|\Z)', content, flags=re.MULTILINE | re.DOTALL)
print(f"Found {len(posts)} posts to process.")

model = genai.GenerativeModel('gemini-2.0-flash')

with open(output_path, "w", encoding="utf-8") as out:
    for idx, post in enumerate(posts, start=1):
        print(f"Summarizing post {idx}/{len(posts)}…")
        prompt = f"""You are a data extraction specialist. Analyze this Reddit post and extract the following fields into the exact text format shown. If a field is missing, use "N/A". Dates must be ISO 8601 (YYYY-MM-DDTHH:MM:SSZ).

        Examples of harms that people are reporting:
        - Misinformation
        - Hallucination
        - Not outputting the correct answer
        -Jailbreak

This is very important: 
We are looking for reports! A report is an individual’s specific written account with a model where the model behavior was harmful, unfair or unusual.


We do not care about Users commentary on harms, flaws of LLMs or ways to improve LLMs. We only care about posts in which the model itself is the one that is harming.


An example of a report:

Is anyone else getting this? Every time I try to talk to ChatGPT it starts hallucinating messages that i never sent and answering the most random things. I have never sent anything to do with 'Dancing With Water', I have no idea what that is. Is this happening to anyone else or have I just broken my guy?

An example of something mentioning hallucination but is NOT a report:

I'm honestly worried about OpenAI's decision to make GPT-5 a unified model instead of separate specialized models. Right now, whenever I need accurate and fact-based responses that rely on real-time web searches I use o4-mini-high, because models like GPT-4o or 4.1 constantly hallucinate and confidently lie about simple facts. By merging all of these different functionalities into a single unified model, I'm afraid GPT-5 will keep defaulting to a model without proper logical reasoning.

For example, say I ask GPT-5 for recent, accurate information on a developing news story. If the unified model judges this query as "simple" or "straightforward," it might route my request to a weaker, more hallucination-prone component (like the current GPT-4o), giving me a half-assed web search that returns 4 weak sources, instead of the 15 or 20 robust references I'd actually need to ensure accuracy. I just don't trust the unified model's ability to consistently determine the best approach.


REDDIT POST:
{post}

OUTPUT FORMAT:
Post ID: …
Summary: …
Reporr: Yes/No
Report Type: …
Date UTC: …
Score: …
Comments: …
URL: …"""
        response = model.generate_content(prompt)
        text = response.text.strip()
        if not text:
            print(f" Empty response for post {idx}")
            text = "No output"
        out.write(text + "\n\n---\n\n")
        print(f"Wrote summary for post {idx}")

print(f"All done. Summaries written to {output_path}")

if __name__ == "__main__":
    pass
