import os
from dotenv import load_dotenv
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from qdrant_client import QdrantClient
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
load_dotenv()

# Step 1: Configure Qdrant client
qdrant = QdrantClient(
    url="https://977135f1-f9a6-4273-a5f2-7e3768956e02.us-west-1-0.aws.cloud.qdrant.io",
    api_key=os.getenv("API_KEY"),
)


qdrant.delete(
    collection_name="chatbot",
    points_selector=Filter(
        must=[
            FieldCondition(
                key="source_file",
                match=MatchValue(value="How Do I Block or Report Someone.pdf")
            )
        ]
    )
)


# qdrant.create_payload_index(
#     collection_name="chatbot",
#     field_name="source_file",
#     field_schema="keyword"  # this tells Qdrant to treat it as a filterable string
# )

vectorStore = QdrantVectorStore(
        client=qdrant,
        collection_name="chatbot",
        embedding= HuggingFaceEmbeddings(
            model_name="mixedbread-ai/mxbai-embed-large-v1",
            model_kwargs={"token": "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"}
        )
    )


# metadata = vectorStore.client.scroll(
#     collection_name="chatbot",
#     limit=5,
#     with_payload=True
# )

# print(metadata)
# pdf_files = [file for file in os.listdir() if file.endswith(".pdf") and file.startswith("Guidelines and Rules")]
# print(f"Found PDF files: {pdf_files}")

# all_documents = []

# for file in pdf_files:
#     loader = PyPDFLoader(file)
#     pages = loader.load()
#     for page in pages:
#         page.metadata["source_file"] = file
#     all_documents.extend(pages)

# # Step 6: Split text into manageable chunks
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=800,
#     chunk_overlap=200,
# )
# documents = text_splitter.split_documents(all_documents)

# # Step 7: Store into Qdrant
# vectorStore.add_documents(documents)

print("✅ Data successfully Deleted from Qdrant collection 'chatbot'.")