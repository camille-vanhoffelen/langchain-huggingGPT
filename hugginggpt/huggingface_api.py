import os

from dotenv import load_dotenv

load_dotenv()

HUGGINGFACE_INFERENCE_API_URL = "https://api-inference.huggingface.co/models/"
HUGGINGFACE_INFERENCE_API_STATUS_URL = f"https://api-inference.huggingface.co/status/"


def get_hf_headers():
    HUGGINGFACEHUB_API_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    return {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}
