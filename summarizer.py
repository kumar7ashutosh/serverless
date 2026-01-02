import os
from openai import OpenAI

PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
BASE_URL = "https://api.groq.com/openai/v1"
API_KEY = os.getenv("GROQ_API_KEY")
MODEL = "llama-3.1-8b-instant"
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

INPUT_FOLDER = "input_files"
OUTPUT_FOLDER = "output_summaries"

def summarize_text(text):
    response=client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role":"system","content":"you are helpful text summarizer"},
            {"role":"user","content":f"summarize text \n\n {text}"}
        ],
        temperature=0.5,
        max_tokens=300       
    )
    return response.choices[0].message.content

def summarize_text_from_file(file_path):
    with open(file_path,'r',encoding='utf-8') as f:
        text=f.read()
    return summarize_text(text)

def process_files():
    for filename in os.listdir(INPUT_FOLDER):
        if filename.endswith(".txt"):
            input_path = os.path.join(INPUT_FOLDER, filename)
            output_path = os.path.join(OUTPUT_FOLDER, f"summary_{filename}")

            summary = summarize_text_from_file(input_path)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(summary)

            print(f"Summarized: {filename} -> {output_path}")

if __name__ == "__main__":
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    process_files()