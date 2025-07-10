from langchain_qdrant.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from dotenv import load_dotenv
import os

load_dotenv() # Load environment variables from .env file


client = QdrantClient(
    url="https://977135f1-f9a6-4273-a5f2-7e3768956e02.us-west-1-0.aws.cloud.qdrant.io",
    api_key=os.getenv("API_KEY"),
)


embedding = OllamaEmbeddings(model="mxbai-embed-large")

client.recreate_collection(
    collection_name="chatbot",  # Name of the collection
    vectors_config=VectorParams(size=1024, distance=Distance.COSINE),  # Vector size and distance metric
)

qdrant = QdrantVectorStore(
        client=client,
        collection_name="chatbot",
        embedding= HuggingFaceEmbeddings(
            model_name="mixedbread-ai/mxbai-embed-large-v1",
            model_kwargs={"token": "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"}
        )
    )