## 🎧 CSU Multi-Agent Support Bot
This project implements a comprehensive, multi-agent support system using LangGraph and Retrieval-Augmented Generation (RAG). The bot classifies user queries into IT or HR issues, uses dedicated knowledge bases (PDFs) to find answers, handles small talk, and automatically escalates unanswered or uncategorized queries by creating a support ticket.

The system is split into two files:

agent_backend.py: The core backend logic, agents, and LangGraph workflow.

agent_frontend.py: A front-end interface built with Streamlit.

✨ Features
Intent Routing: A Supervisor Agent classifies user input into SMALL_TALK, IT_ISSUE, HR_ISSUE, or OTHER.

RAG Agents: Dedicated it_agent and hr_agent use separate ChromaDB vector stores (Knowledge Bases) for accurate, context-aware answers.

Conversation Memory: State is persisted across sessions and turns using a SQLite checkpointing system powered by LangGraph.

Ticket Escalation: If a RAG agent fails to find an answer, or if the intent is OTHER, the system offers to create a ticket. Upon confirmation, it logs the issue to a CSV file.

User Interface: A simple, persistent chat interface provided by Streamlit.

🛠️ Prerequisites

Before starting, ensure you have the following installed and available:

Python 3.9+

An OpenAI API Key

Knowledge Base Files: Two PDF files to serve as the IT and HR documentation.

🚀 Setup and Installation
1. Clone the Repository (or setup files)
Ensure you have the following files in your project directory:

agent_backend.py (Backend logic)
agent_frontend.py (Streamlit frontend)
requirements.txt (Provided in a previous turn)
Your two PDF knowledge base files (e.g., CSU_GLB_IT_Support.pdf, CSU_GLB_HR_Policy.pdf).

2. Install Dependencies
Install all necessary Python packages using the provided requirements.txt:
Bash
pip install -r requirements.txt

3. Configure Environment Variables
Create a file named .env in the root directory of your project and populate it with your OpenAI API Key.
.env
Ini, TOML
OPENAI_API_KEY="your_secret_openai_api_key_here"

## Optional: You can specify models if desired
## OPENAI_EMBEDDING_MODEL="text-embedding-ada-002"
## OPENAI_CHAT_MODEL="gpt-4o-mini"

4. Update Hardcoded File Paths in agent_backend.py
The core backend file uses absolute paths that must be updated to match your local file structure. This is mandatory.
Open agent_backend.py and modify the paths in the CONFIGURATION and KNOWLEDGE BASE SETUP sections.
Python

## In agent_backend.py, modify these paths:
## --------------------------------------------------------------------------
## CONFIGURATION SECTION (Line ~28)
## Replace with the actual absolute path where you want the ticket log to be saved.
TICKET_LOG_FILE = r""

# --------------------------------------------------------------------------
## KNOWLEDGE BASE SETUP SECTION (Lines ~60-61)
## Replace with the actual absolute paths to your IT and HR PDF files.
IT_KB = _load_or_build_kb(r"", "./chroma_it_store")
HR_KB = _load_or_build_kb(r"",  "./chroma_hr_store")
## Note: The folders "./chroma_it_store" and "./chroma_hr_store" will be created automatically.
▶️ Running the Application
Once setup is complete, run the Streamlit application from your terminal:
Bash
streamlit run agent_frontend.py
This will open the web application in your browser, where you can begin interacting with the support bot.
The first time you run it, the system will read your PDFs and build the Chroma vector stores in the designated folders (./chroma_it_store and ./chroma_hr_store). This may take a minute or two.
Conversation history and state are saved in chatbot.db.
Escalated issues are logged to the support_tickets.csv file.