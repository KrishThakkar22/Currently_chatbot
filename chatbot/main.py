from fastapi import FastAPI, Request, HTTPException
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.memory import ConversationSummaryMemory
from langchain_qdrant import QdrantVectorStore
from langchain_openai import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from collections import defaultdict
from datetime import datetime
from typing import Dict
import asyncio
import httpx
import os
import uvicorn
import time

# Load environment variables
load_dotenv()

my_admin_id = 8467307
EXPIRE_SECONDS = 1800
message_buffers = defaultdict(list)
last_message_times = defaultdict(float)
locks = defaultdict(asyncio.Lock)
pending_tasks = {}  # For debounce control
buffer_wait_seconds = 5
REPLIED_MESSAGE_IDS = {}

system_template = """
 You are **Shivi**, a helpful and polite junior customer support assistant at Currently — a social app for sharing real-time moments.

 🎯 **Your job:** Use ONLY the provided context snippets to answer the user's question. You may summarize, synthesize, or combine information across snippets — but do NOT invent facts or use outside knowledge.

 ---

 ## ✅ What you MUST do:

 - If the context contains relevant information, answer clearly, briefly, and naturally.
 - If asked “why” or “how”, look across ALL snippets for reasons or steps — even partial ones.
 - If the user's message is a **greeting** or casual message (e.g., "hi", "hello", "good morning", "can you help me?"), respond **warmly and casually** — like a real assistant would.
 - If the answer is **not** found in the context and it's a real question, say:
   **"I'm not sure about that right now, but I'll connect you with a senior team member. Please stay connected."**

 ---

 ## ❌ What you MUST AVOID:

 - DO NOT say: "contact customer support", "raise a ticket", "reach out to the team", or anything similar.
 - DO NOT mention context, documents, PDFs, or that you are a bot or AI.
 - DO NOT invent information or guess.
 - DO NOT ask anykind follow-up questions in replies.
 - DO NOT trigger the fallback message for simple greetings.

 ---

 If the user asks personal questions (e.g., 'why is my account banned?'), but you don't have access to specific account data, reply with a general reason.
Always assume the user may be referring to their own experience even if the phrasing is vague.
 ## 💬 Examples:

 **Q:** Why was my moment rejected?  
 **Context 1:** “Moments are reviewed for safety and policy compliance.”  
 **Context 2:** “Moments may be rejected if they contain offensive content or violate terms.”  
 **A:** Moments might be rejected if they contain offensive content or go against our safety policies.

 **Q:** Thanks  
 **A:** You're very welcome! 😊 Let me know if I can help with anything else.

 **Q:** Hi  
 **A:** Hi there! 👋 How can I help you today?

 **Q:** Good morning  
 **A:** Good morning! ☀️ Hope you're having a great day. What can I help you with?

 **Q:** I have a question  
 **A:** Sure! 😊 Feel free to ask — I’m here to help.

 **Q:** Why did I lose coins? (no info in context)  
 **A:** I'm not sure about that right now, but I'll connect you with a senior team member. Please stay connected.

 **Q:** Okay 
 **A:** Feel Free to ask again.

 **Q:** Thank You (Similar text)  
 **A:** Your Welcome, Feel free to ask again.

 ---

 ✅ Be friendly and natural. Only use the fallback if there's a **real question with no info** in context.
"""

human_template = """
Here is the context:

{context}

Customer's question: "{question}"

How would Max respond — using ONLY the context above, and without saying anything like 'please contact support' or 'reach out to the team'?
"""

app = FastAPI()

user_memory_store: dict[str, ConversationSummaryMemory] = {}

qdrant_client = QdrantClient(
    url="https://977135f1-f9a6-4273-a5f2-7e3768956e02.us-west-1-0.aws.cloud.qdrant.io",
    api_key=os.getenv("API_KEY")
)

vectorStore = QdrantVectorStore(
    client=qdrant_client,
    collection_name="chatbot",
    embedding=HuggingFaceEmbeddings(
        model_name="mixedbread-ai/mxbai-embed-large-v1",
        model_kwargs={"token": os.getenv("HF_TOKEN")}
    )
)

retriever = vectorStore.as_retriever(search_kwargs={"k": 20})

model = ChatOpenAI(
    model="gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY")
)

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template(human_template)
])

def get_memory_for_user(user_id: str) -> ConversationSummaryMemory:
    if user_id not in user_memory_store:
        user_memory_store[user_id] = ConversationSummaryMemory(
            llm=ChatOpenAI(model="gpt-4", api_key=os.getenv("OPENAI_API_KEY")),
            memory_key="chat_history",
            input_key="question",
            return_messages=True
        )
    return user_memory_store[user_id]

def get_lock_for_conversation(conv_id: str) -> asyncio.Lock:
    return locks[conv_id]

def get_chain_for_user(user_id: str) -> ConversationalRetrievalChain:
    memory = get_memory_for_user(user_id)
    return ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt}
    )

async def reply_to_intercom(conversation_id: str, message: str, msg_id: str, retries: int = 3):
    url = f"https://api.intercom.io/conversations/{conversation_id}/reply"
    headers = {
        "Authorization": f"Bearer dG9rOmMzNzFhOTk5X2I3NmNfNDhiN19hYjhhX2Q0YjRlMWVmN2FiYjoxOjA=",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    payload = {
        "message_type": "comment",
        "type": "admin",
        "admin_id": my_admin_id,
        "body": message
    }

    for attempt in range(retries):
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                print("✅ Sent reply to Intercom")
                REPLIED_MESSAGE_IDS[msg_id] = time.time()
                return
        except httpx.RequestError as exc:
            print(f"❌ Request error (attempt {attempt+1}): {exc}")
        except httpx.HTTPStatusError as exc:
            print(f"❌ HTTP error (attempt {attempt+1}): {exc.response.status_code} - {exc.response.text}")
    print("🚨 Failed to send reply to Intercom after retries.")


async def close_intercom_conversation(conversation_id: str):
    url = f"https://api.intercom.io/conversations/{conversation_id}/reply"
    headers = {
        "Authorization": "Bearer dG9rOmMzNzFhOTk5X2I3NmNfNDhiN19hYjhhX2Q0YjRlMWVmN2FiYjoxOjA=",  # Replace with your token
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    payload = {
        "message_type": "close",
        "type": "admin",
        "admin_id": my_admin_id  # Your admin ID
    }
    del message_buffers[conversation_id]
    del last_message_times[conversation_id]
    del locks[conversation_id]
    del user_memory_store[conversation_id]
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        print("✅ Conversation closed successfully")


async def cleanup_replied_ids():
    while True:
        now = time.time()
        expired = [mid for mid, ts in REPLIED_MESSAGE_IDS.items() if now - ts > EXPIRE_SECONDS]
        for mid in expired:
            del REPLIED_MESSAGE_IDS[mid]
        await asyncio.sleep(600)  # Run cleanup every 10 minutes


# ASSIGNED_CONVERSATIONS = set()

async def assign_if_new_conversation(conversation_id: str, admin_id: int):
    # if conversation_id in ASSIGNED_CONVERSATIONS:
    #     print(f"🔁 Already assigned conversation {conversation_id}, skipping.")
    #     return

    headers = {
        "Authorization": "Bearer dG9rOmMzNzFhOTk5X2I3NmNfNDhiN19hYjhhX2Q0YjRlMWVmN2FiYjoxOjA=",
        "Accept": "application/json",
        "Content-Type": "application/json"
    }

    assign_url = f"https://api.intercom.io/conversations/{conversation_id}/reply"
    assign_payload = {
        "message_type": "assignment",
        "type": "admin",
        "admin_id": admin_id,
        "assignee_id": admin_id
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.post(assign_url, headers=headers, json=assign_payload)
            response.raise_for_status()
            print(f"✅ Assigned new conversation {conversation_id} to admin {admin_id}")
        except httpx.HTTPStatusError as e:
            print(f"❌ HTTP error while assigning: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            print(f"❌ Unexpected error during assignment: {e}")


async def unassign_conversation(conversation_id: str):
    url = f"https://api.intercom.io/conversations/{conversation_id}/reply"
    headers = {
        "Authorization": f"Bearer dG9rOmMzNzFhOTk5X2I3NmNfNDhiN19hYjhhX2Q0YjRlMWVmN2FiYjoxOjA=",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    payload = {
        "message_type": "assignment",
        "type": "admin",
        "admin_id": my_admin_id,
        "assignee_id": 8032673
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        print("✅ Conversation unassigned successfully")

    # del message_buffer[conversation_id]

def schedule_chain_invoke(conv_id, question, msg_id):
    message_buffers[conv_id].append(question)
    last_message_times[conv_id] = time.time()

    if conv_id in pending_tasks:
        pending_tasks[conv_id].cancel()

    pending_tasks[conv_id] = asyncio.create_task(_delayed_invoke(conv_id, msg_id))

async def _delayed_invoke(conv_id, msg_id, retries=3):
    await asyncio.sleep(buffer_wait_seconds)

    lock = get_lock_for_conversation(conv_id)
    async with lock:
        combined_message = " ".join(message_buffers[conv_id])
        message_buffers[conv_id] = []

        chain = get_chain_for_user(conv_id)

        for attempt in range(retries):
            try:
                result = await chain.ainvoke({"question": combined_message})
                answer = result.get("answer", "Sorry, no answer returned.")
                break
            except Exception as e:
                print(f"❌ Error during chain invocation (attempt {attempt+1}):", str(e))
                if attempt == retries - 1:
                    answer = "An error occurred while processing your request. Please try again later."

        if "connect you with a senior" in answer.lower():
            await reply_to_intercom(conversation_id=conv_id, message=answer, msg_id=msg_id)
            await unassign_conversation(conversation_id=conv_id)
        elif ("welcome" in answer.lower() or "ask again" in answer.lower()) :
            await reply_to_intercom(conversation_id=conv_id, message=answer, msg_id=msg_id)
            await close_intercom_conversation(conversation_id=conv_id)
        else:
            await reply_to_intercom(conversation_id=conv_id, message=answer, msg_id=msg_id)


def parse_intercom_datetime(iso_str: str) -> int:
    """Convert ISO time to UNIX timestamp (in seconds)."""
    dt = datetime.strptime(iso_str, "%Y-%m-%dT%H:%M:%S.%fZ")
    return int(dt.timestamp())

def should_bot_take_over_reopened_chat(stats: dict) -> bool:
    try:
        last_close = parse_intercom_datetime(stats.get("last_close_at", "1970-01-01T00:00:00.000Z"))
        last_contact = parse_intercom_datetime(stats.get("last_contact_reply_at", "1970-01-01T00:00:00.000Z"))
        last_admin = parse_intercom_datetime(stats.get("last_admin_reply_at", "1970-01-01T00:00:00.000Z"))

        return last_contact > last_close and last_admin < last_contact
    except:
        return False


@app.post("/query")
async def chat_endpoint(request: Request):
    payload = await request.json()
    
    # print(payload)
    assignee_id = payload.get("data", {}).get("item", {}).get("admin_assignee_id")
    msg_id = payload["data"]["item"]["conversation_parts"]["conversation_parts"][-1]["id"]
    if msg_id in REPLIED_MESSAGE_IDS:
        print("Already Replied")
        return {"status": "already_replied"}
    try:
        conv_id = payload["data"]["item"]["id"]
    except KeyError:
        raise HTTPException(status_code=400, detail="Invalid payload structure")

    parts = payload.get("data", {}).get("item", {}).get("conversation_parts", {}).get("conversation_parts", [])
    
    html = (
        parts[-1].get("body") or
        payload.get("data", {}).get("item", {}).get("source", {}).get("body")
    )

    html = html or ""
    question = BeautifulSoup(html, "html.parser").get_text().strip()
    print(question)
    if assignee_id is None or assignee_id == "":
        await assign_if_new_conversation(conversation_id=conv_id, admin_id=my_admin_id)
        schedule_chain_invoke(conv_id, question, msg_id)

    elif assignee_id == my_admin_id:
        print(payload)
        schedule_chain_invoke(conv_id, question, msg_id)

    else:
        if (should_bot_take_over_reopened_chat(payload["data"]["item"]["statistics"])):
            print("👀 Reopened conversation — reassigning to bot")
            await assign_if_new_conversation(conversation_id=conv_id, admin_id=my_admin_id)
            schedule_chain_invoke(conv_id, question, msg_id)
        else:
            print("🔒 Active conversation with human — skipping")

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(cleanup_replied_ids())

if __name__ == "__main__":
    # asyncio.run(cleanup_replied_ids())
    uvicorn.run("main:app", host="0.0.0.0", port=3000, reload=True)