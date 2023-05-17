import os
from dotenv import load_dotenv

load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
HUGGINGFACE_HEADERS = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}
HUGGINGFACE_INFERENCE_API_URL = "https://api-inference.huggingface.co/models/"
