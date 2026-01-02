import os
import boto3
import logging
from openai import OpenAI

# Setup logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)  # For CloudWatch visibility

# Ensure handlers are added only once in Lambda environment
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

s3 = boto3.client("s3")
PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
BASE_URL = "https://api.groq.com/openai/v1"
API_KEY = os.getenv("GROQ_API_KEY")
MODEL = "llama-3.1-8b-instant"
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

BUCKET = os.getenv("BUCKET_NAME", "lambdagenaibucket2")
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

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

def lambda_handler(event, context):
    logger.info("Lambda triggered. Raw event: %s", event)

    try:
        record = event['Records'][0]
        bucket_name = record['s3']['bucket']['name']
        key = record['s3']['object']['key']

        logger.info("Received file: s3://%s/%s", bucket_name, key)

        if not key.endswith(".txt"):
            logger.warning("Skipping non-text file: %s", key)
            return {"status": "skipped_non_txt"}

        file_obj = s3.get_object(Bucket=bucket_name, Key=key)
        text = file_obj["Body"].read().decode("utf-8")
        logger.info("Downloaded file. Size: %d characters", len(text))

        summary = summarize_text(text)

        filename = key.split("/")[-1]
        output_key = f"output/summary_{filename}"
        s3.put_object(Body=summary.encode("utf-8"), Bucket=bucket_name, Key=output_key)

        logger.info("Successfully saved summary to: s3://%s/%s", bucket_name, output_key)
        return {"status": "complete"}

    except Exception as e:
        logger.error("Unhandled exception: %s", str(e), exc_info=True)
        raise