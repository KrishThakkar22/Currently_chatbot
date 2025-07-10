import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_qdrant.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# Load environment variables (e.g., API key)
load_dotenv()

# Step 1: Set up Qdrant client
client = QdrantClient(
    url="https://977135f1-f9a6-4273-a5f2-7e3768956e02.us-west-1-0.aws.cloud.qdrant.io",
    api_key=os.getenv("API_KEY"),
    timeout=60.0  # Increase timeout
)

# Step 2: Recreate collection (optional if already done)
client.recreate_collection(
    collection_name="chatbot",
    vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
)

# Step 3: Initialize embedding model
embedding = HuggingFaceEmbeddings(
    model_name="mixedbread-ai/mxbai-embed-large-v1",
    model_kwargs={"token": "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"}
)

# Step 4: Set up Qdrant Vector Store
qdrant = Qdrant(
    client=client,
    collection_name="chatbot",
    embeddings=embedding
)

# Step 5: Load PDFs
pdf_files = [file for file in os.listdir() if file.endswith(".pdf")]
print(f"Found PDF files: {pdf_files}")

all_documents = []
for file in pdf_files:
    loader = PyPDFLoader(file)
    pages = loader.load()
    for page in pages:
        page.metadata["source_file"] = file
    all_documents.extend(pages)

# Step 6: Split documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
documents = text_splitter.split_documents(all_documents)

# Step 7: Batch upload
def batch_upload(docs, batch_size=50):
    for i in range(0, len(docs), batch_size):
        print(f"Uploading batch {i // batch_size + 1}...")
        qdrant.add_documents(docs[i:i+batch_size])

batch_upload(documents)

print("✅ All documents uploaded successfully to Qdrant.")
