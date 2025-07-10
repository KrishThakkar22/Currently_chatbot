<h2 align="center">🤖 AI Chatbot Project Report — CurrentlyBot</h2>

<hr>

<h3>📌 Project Overview</h3>
<p>
The goal of this project was to develop an intelligent, context-aware chatbot integrated with <strong>Intercom</strong>, powered by <strong>GPT-4</strong>, to handle real-time user queries efficiently. The bot is capable of:
<ul>
  <li>Autonomous decision-making</li>
  <li>Message understanding</li>
  <li>Dynamic assignment</li>
  <li>Fallback escalation</li>
</ul>
It maintains conversational memory and context through <strong>Langchain</strong>.
</p>

<h3>✨ Features Implemented</h3>
<ul>
  <li><strong>🕔 Debouncing Mechanism (5-second Buffer):</strong> Batches rapid messages to maintain coherent conversations.</li>
  <li><strong>✅ Auto-Closing Conversations:</strong> Closes inactive or resolved chats automatically.</li>
  <li><strong>⚠️ Intelligent Escalation:</strong> Transfers to human agent when GPT-4 lacks confidence.</li>
  <li><strong>📥 First-Time Assignment:</strong> Auto-assigns unassigned conversations on first user message.</li>
  <li><strong>🧠 Context Management:</strong> Uses Langchain memory summaries to preserve conversation context.</li>
  <li><strong>⚡ Async Multi-Conversation Handling:</strong> Uses FastAPI and asyncio to handle chats concurrently.</li>
</ul>

<h3>💡 Strengths of GPT-4 Integration</h3>
<ul>
  <li><strong>🌐 Multilingual Support:</strong> Understands and responds in Hindi, English, Gujarati, and more — no extra translation API needed.</li>
  <li>✅ Already in place: Native multilingual capabilities of GPT-4</li>
  <li>🔜 Future improvement: Dynamically adjust prompt tone based on detected language</li>
  <li><strong>🧠 Better Understanding:</strong> GPT-4’s large context window allows deeper follow-ups and intelligent responses.</li>
</ul>

<h3>🚧 Known Limitations</h3>
<ul>
  <li><strong>🔄 Reopened Chat Detection:</strong> Currently unreliable when users reopen old chats.</li>
  <li><strong>🙁 Occasional Lack of Personalization:</strong> GPT-4 replies can feel generic without profile-based customization.</li>
</ul>

<h3>🚀 Future Scope & Improvements</h3>
<ul>
  <li>✅ Improve reopened chat handling via last reply timestamp</li>
  <li>✅ Add personalization by pulling user data from Intercom</li>
  <li>✅ Add analytics dashboard for monitoring chatbot activity</li>
</ul>

<h3>🛠️ Tech Stack Summary</h3>
<ul>
  <li><strong>FastAPI</strong> – Webhook server for Intercom integration</li>
  <li><strong>OpenAI GPT-4</strong> – Language understanding & response generation</li>
  <li><strong>Langchain</strong> – Memory and summarization module</li>
  <li><strong>Intercom API</strong> – For conversation and message management</li>
  <li><strong>Qdrant</strong> – (Optional) Vector DB for long-term memory</li>
  <li><strong>AsyncIO / httpx</strong> – Asynchronous message handling</li>
</ul>

<h3>📦 Deliverables</h3>
<ul>
  <li>✅ FastAPI backend integrated with Intercom</li>
  <li>✅ Langchain + GPT-4 integration with memory summarization</li>
  <li>✅ Debounce logic for batching messages</li>
  <li>✅ Auto-assign, fallback routing, and auto-close workflow</li>
</ul>

<hr>

<p align="center"><strong>Made with ❤️ by the CurrentlyBot Team</strong></p>
