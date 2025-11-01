"""
CSU Multi-Agent Support Bot - Backend
======================================
A LangGraph-based multi-agent system for IT and HR support with ticket escalation.

The system uses a Supervisor agent for intent routing, dedicated RAG agents for IT/HR,
and LangGraph's SqliteSaver for conversation persistence (memory).

Dependencies:
- langchain, langgraph, python-dotenv, chromadb, pypdf, openai
"""
import os
import csv
import uuid
import sqlite3
from datetime import datetime
from typing import Dict, Optional, List, Literal, Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langgraph.graph import StateGraph, END, add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
# Load environment variables (e.g., OPENAI_API_KEY) from .env file
load_dotenv()
# Database and file paths
DB_PATH         = "chatbot.db" # SQLite file for LangGraph checkpointer
# Absolute path for the support ticket log CSV file
TICKET_LOG_FILE = r"C:\DS_Material\CSU\AI_Course_CSU\P1\D1\support_tickets.csv"
EMB_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
LLM_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
# ═══════════════════════════════════════════════════════════════════════════
# INITIALIZE OBJECTS
# ═══════════════════════════════════════════════════════════════════════════
# OpenAI embeddings for vector similarity search (used by Chroma)
embeddings = OpenAIEmbeddings(model=EMB_MODEL)
# OpenAI chat model (temperature=0 is used for deterministic routing/classification)
llm        = ChatOpenAI(model=LLM_MODEL, temperature=0)
# SQLite connection for conversation persistence checkpointer
conn       = sqlite3.connect(DB_PATH, check_same_thread=False)
# LangGraph checkpointer for maintaining conversation state per user thread
memory     = SqliteSaver(conn)

# ═══════════════════════════════════════════════════════════════════════════
# KNOWLEDGE BASE SETUP
# ═══════════════════════════════════════════════════════════════════════════

def _load_or_build_kb(pdf: str, folder: str) -> Optional[Chroma]:
    """
    Load an existing ChromaDB vector store from disk or build a new one from a PDF.
    
    Args:
        pdf: Path to the source PDF file.
        folder: Directory path to store/load the ChromaDB vector store.
        
    Returns:
        Chroma vector store instance or None if the PDF doesn't exist.
    """
    # If vector store already exists, load it
    if os.path.exists(folder):
        return Chroma(persist_directory=folder, embedding_function=embeddings)
    # If the source PDF is not found, return None (KB is unavailable)
    if not os.path.exists(pdf):
        return None
    # Load PDF and split into chunks using recursive splitting
    docs = PyPDFLoader(pdf).load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)
    # Create and persist the new vector store from the chunks
    return Chroma.from_documents(chunks, embeddings, persist_directory=folder)
# Load or build knowledge bases for IT and HR
IT_KB = _load_or_build_kb(r"C:\DS_Material\CSU\AI_Course_CSU\P1\D1\CSU_GLB_IT_Support.pdf", "./chroma_it_store")
HR_KB = _load_or_build_kb(r"C:\DS_Material\CSU\AI_Course_CSU\P1\D1\CSU_GLB_HR_Policy.pdf",  "./chroma_hr_store")

# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def _write_ticket(tid: str, query: str, typ: str, uid: str):
    """
    Appends a new support ticket record to the CSV log file.
    Ensures a header is written if the file is created for the first time.
    """
    file_exists = os.path.exists(TICKET_LOG_FILE)
    # Open file in append mode ('a') with universal newline handling
    with open(TICKET_LOG_FILE, "a", newline="", encoding="utf-8") as f: 
        writer = csv.DictWriter(f, fieldnames=["Ticket ID","Timestamp","Issue Type","User Query","User ID"])
        # Write header only if the file is newly created
        if not file_exists:
            writer.writeheader()
        # Write the new ticket data row
        writer.writerow({"Ticket ID":tid,"Timestamp":datetime.now().isoformat(),
                         "Issue Type":typ,"User Query":query,"User ID":uid})

def _full_history(state: 'AgentState') -> str:
    """Helper to format the conversation history for use in LLM prompts."""
    return "\n".join([f"{type(m).__name__.replace('Message', '')}: {m.content}" for m in state["messages"]])


# ---------- state ----------
class AgentState(TypedDict):
    """
    The state dictionary passed between LangGraph nodes.
    'messages' uses the `add_messages` key for automatic history updates.
    """
    messages: Annotated[List[BaseMessage], add_messages]
    next_agent: str                   # Name of the next node to execute
    user_id: str                      # Unique ID for the user/thread
    current_query: str                # The user's last message
    pending_ticket: Optional[Dict[str,str]] # Data if a ticket offer is pending confirmation
    final_response: Optional[str]     # The final response for the user interface

# ---------- persona ----------
PERSONA = """You are "CSU-Bot", a friendly, human-like support assistant.
- Use the whole conversation history; never repeat questions.
- If the user told you their name, use it naturally.
- Keep answers short, warm, conversational (max 2-3 sentences unless steps needed).
- Use bullet emojis (•) for steps.
- If you cannot help, say "I'll connect you with a human agent."
"""


# ---------- nodes ----------
def supervisor_agent(state: AgentState) -> AgentState:
    """
    The entry point and primary routing node.
    It checks for pending ticket confirmation first, then classifies the intent.
    """
    messages = state["messages"]
    current_query = messages[-1].content if messages else ""
    
    # --- 1. Handle Pending Ticket Confirmation ---
    if state.get("pending_ticket"):
        user_response = current_query.lower().strip()
        # Check for positive confirmation
        if any(word in user_response for word in ["yes", "y", "ok", "please", "proceed", "create", "sure"]):
            state["next_agent"] = "other_agent" # Route to other_agent to create the ticket
            state["current_query"] = current_query
            return state
        # User declined confirmation
        else:
            response = "✅ No problem! No ticket was created. How else can I help you?"
            state["messages"] = [AIMessage(content=response)]
            state["pending_ticket"] = None
            state["final_response"] = response
            state["next_agent"] = "END"
            return state
    
    # --- 2. Intent Classification ---
    # Prompt instructing the LLM to classify the user's current query
    classification_prompt = f"""You are an expert intent-router for a corporate support assistant.
Output ONLY the uppercase category label.

Categories
-----------
SMALL_TALK
  Use this for ANY message whose primary purpose is social, personal, mnemonic, or casual.
  Includes but is not limited to:

  1. Greetings & farewells
    hi, hello, hey, good morning, bye, see you, take care, cheers, ttyl

  2. Thanks & politeness
    thanks, thank you, ty, appreciated, you're awesome, 🙏

  3. Personal questions aimed at the bot
    - how are you / how's your day / are you a robot / are you human
    - what is your name / who created you / what model are you

  4. Memory-check or identity questions about the USER
    - what is my name / do you remember my name
    - who am i / tell me who i am
    - do you remember me / do you know me
    - remind me what i told you

  5. Meta-chat about the conversation itself
    - can we change topic / let's talk about something else
    - forget everything / clear history (treat as SMALL_TALK, do NOT clear)
    - repeat what you just said / summarise our chat

  6. Casual pleasantries & small social talk
    - how's the weather / hope you're having a good day
    - tell me a joke / give me fun fact / entertain me
    - i'm bored / i am lonely / let's chat

  7. Emotional or status utterances
    - i'm happy / sad / excited / tired / stressed
    - wish me luck / i got promoted / it's my birthday

  8. Requests that explicitly ask for a human-like reaction
    - act like my friend / pretend you're my pal
    - answer like a human / be more casual

  9. ANY follow-up message that is ONLY:
    "yes", "yeah", "yep", "no", "nope", "ok", "okay", "sure", "cool", "great", "awesome", "thanks", "👍", "❤️", "😂" etc.
    (These are social acknowledgements, not IT/HR content.)

IT_ISSUE
  Hardware, software, network, passwords, VPN, email, printers, O365, MFA, error codes, slowness, crashes, installation, updates, git, server access, AWS, Azure, laptop issues, blue screen, Wi-Fi, cable, mouse, keyboard, monitor, audio, webcam, Teams, Zoom, Slack, Confluence, Jira, Service-Now, SAP, Salesforce, SQL, code debugging, API failures, disk space, backup, antivirus, firewall, SSL certificate, domain, DNS, DHCP, IP address, licensing, Photoshop, Adobe, Excel formulas, PowerBI, Tableau, etc.

HR_ISSUE
  Policies, employee handbook, vacation / PTO, sick leave, parental leave, holidays, payroll, salary, bonus, W-2, 1099, expense reimbursement, benefits enrolment, health insurance, dental, vision, 401k, pension, retirement, stock options, ESPP, tuition assistance, relocation, immigration, visa, green card, job postings, internal transfer, performance review, promotion, resignation, termination, exit interview, code of conduct, harassment, diversity & inclusion, training, learning budget, wellness program, gym subsidy, commuter benefits, birthday leave, bereavement leave, COBRA, FMLA, ADA, workers comp, etc.

OTHER
  Anything that is clearly none of the above AND has no casual / social / memory flavour.
  Example: "I need catering for an off-site event", "Where is the cafeteria menu", "Book a meeting room", "Order business cards", "Translate this document to Spanish".

Decision rules
--------------
1. If the user explicitly asks about THEIR OWN name or whether you remember them → SMALL_TALK.
2. If the sentence is ambiguous but contains personal pronouns ("I", "my", "me") plus "name" or "remember" → SMALL_TALK.
3. Single-word acknowledgements ("yes", "no", "ok", "thanks", "👍") → SMALL_TALK.
4. When in doubt between SMALL_TALK and OTHER, choose SMALL_TALK to avoid unnecessary tickets.

User: "{current_query}"
Label:"""
    
    # Invoke the LLM for classification and clean the output
    classification = llm.invoke([HumanMessage(content=classification_prompt)]).content.strip().upper()
    # Map classification to the next agent node, default to 'other_agent'
    state["next_agent"] = {"SMALL_TALK":"small_talk_agent","IT_ISSUE":"it_agent","HR_ISSUE":"hr_agent"}.get(classification,"other_agent")
    state["current_query"] = current_query
    return state

def small_talk_agent(state: AgentState) -> AgentState:
    """Handles small talk intents using the general LLM and the persona."""
    history = _full_history(state)
    # Prompt the LLM with persona and full history
    prompt = f"{PERSONA}\n\nFull conversation:\n{history}\n\nUser now says: \"{state['current_query']}\""
    response = llm.invoke([HumanMessage(content=prompt)]).content
    state["messages"] = [AIMessage(content=response)]
    state["final_response"] = response
    state["next_agent"] = "END" # End the turn
    return state

def it_agent(state: AgentState) -> AgentState:
    """IT RAG Agent: Searches IT KB, provides a solution, or escalates."""
    query = state["current_query"]
    # Check if the IT knowledge base is available
    if not IT_KB:
        state["next_agent"] = "other_agent"
        state["pending_ticket"] = {"query": query, "source": "IT_KB_UNAVAILABLE"}
        return state

    try:
        # Perform RAG retrieval (MMR for diverse results)
        docs = IT_KB.as_retriever(search_type="mmr", search_kwargs={'k': 3, 'fetch_k': 15}).invoke(query)
        # If no relevant documents are found, escalate
        if not docs:
            state["next_agent"] = "other_agent"
            state["pending_ticket"] = {"query": query, "source": "IT"}
            return state

        context = "\n\n".join([d.page_content for d in docs])
        history = _full_history(state)
        
        # Prompt the LLM with context for RAG response generation
        prompt = f"""{PERSONA}

Full conversation:
{history}

IT docs:
{context}

User IT issue: {query}

Instructions:
- Provide short friendly step-by-step fix
- If unsure reply ONLY: ESCALATE
- Otherwise end your answer with exactly:

🎫  If these steps don't help, reply **yes** and I'll open a support ticket.
"""
        response = llm.invoke([HumanMessage(content=prompt)]).content.strip()

        low_resp = response.lower()
        # Check if the LLM decided to explicitly escalate or gave a short, poor answer
        if "escalate" in low_resp or len(response) < 30:
            state["next_agent"] = "other_agent"
            state["pending_ticket"] = {"query": query, "source": "IT"}
            return state

        final_response = f"**IT Support:** {response}"
        state["messages"] = [AIMessage(content=final_response)]
        state["final_response"] = final_response
        # Set pending_ticket to allow the user to confirm ticket creation in the next turn
        state["pending_ticket"] = {"query": query, "source": "IT"}
        state["next_agent"] = "END"
        return state
    except Exception as e:
        print("❌ IT agent error:", e) # Log RAG or LLM failure
        state["next_agent"] = "other_agent"
        state["pending_ticket"] = {"query": query, "source": "IT_ERROR"}
        return state

def hr_agent(state: AgentState) -> AgentState:
    """HR RAG Agent: Searches HR KB, provides an answer, or escalates."""
    query = state["current_query"]
    # Check if the HR knowledge base is available
    if not HR_KB:
        state["next_agent"] = "other_agent"
        state["pending_ticket"] = {"query":query,"source":"HR_KB_UNAVAILABLE"}
        return state
    
    try:
        # Perform RAG retrieval (MMR for diverse results)
        docs = HR_KB.as_retriever(search_type="mmr", search_kwargs={'k':3,'fetch_k':15}).invoke(query)
        # If no relevant documents are found, escalate
        if not docs:
            state["next_agent"] = "other_agent"
            state["pending_ticket"] = {"query":query,"source":"HR"}
            return state
        
        context = "\n\n".join([d.page_content for d in docs])
        history = _full_history(state)
        
        # Prompt the LLM with context for RAG response generation
        prompt = f"""{PERSONA}

Full conversation:
{history}

HR policy:
{context}

User HR question: {query}

Instructions:
- Provide concise friendly answer
- If unsure reply ONLY: ESCALATE
- Otherwise end your answer with exactly:

🎫  If this doesn't fully answer your question, reply **yes** and I'll open a support ticket.
"""
        response = llm.invoke([HumanMessage(content=prompt)]).content.strip()
        
        low_resp = response.lower()
        # Check if the LLM decided to explicitly escalate or gave a short, poor answer
        if "escalate" in low_resp or len(response)<20:
            state["next_agent"] = "other_agent"
            state["pending_ticket"] = {"query":query,"source":"HR"}
            return state
        
        final_response = f"**HR Policy:** {response}"
        state["messages"] = [AIMessage(content=final_response)]
        state["final_response"] = final_response
        # Set pending_ticket to allow the user to confirm ticket creation in the next turn
        state["pending_ticket"] = {"query": query, "source": "HR"}
        state["next_agent"] = "END"
        return state
    except Exception as e:
        print("❌ HR agent error:", e) # Log RAG or LLM failure
        state["next_agent"] = "other_agent"
        state["pending_ticket"] = {"query":query,"source":"HR_ERROR"}
        return state


def other_agent(state: AgentState) -> AgentState:
    """
    Handles two cases:
    1. Creates a ticket if confirmed by the user (from supervisor_agent route).
    2. Offers a ticket for unclassified intents or agent escalations.
    """
    query = state["current_query"]

    confirmation_words = {"yes", "y", "yeah", "yep", "sure", "ok", "okay", "please", "create", "proceed"}
    user_response = query.lower().strip()
    # Check if the user's current query confirms a pending ticket
    user_confirmed = state.get("pending_ticket") and any(w in user_response for w in confirmation_words)

    if user_confirmed:
        # --- TICKET CREATION LOGIC ---
        ticket_info = state["pending_ticket"]
        tid = f"ESC-{uuid.uuid4().hex[:8].upper()}" # Generate unique Ticket ID

        # Debugging and file operation attempt
        print("🔍 ENTERED other_agent", "pending?", bool(state.get("pending_ticket")), "query=", repr(query), flush=True)
        print("🎫 TICKET WRITE:", os.getcwd(), os.path.abspath(TICKET_LOG_FILE), "exists?", os.path.exists(TICKET_LOG_FILE), flush=True) 
        try:
            # Write the ticket to the CSV log file
            _write_ticket(tid, ticket_info["query"], ticket_info["source"], state["user_id"])
            print("✅ CSV row written", flush=True)
        except Exception as e:
            # Handle file write error
            response = (
                f"⚠️  Could not save ticket: {e}\n"
                f"Dir: {os.getcwd()}\n"
                f"File: {os.path.abspath(TICKET_LOG_FILE)}"
            )
            state["messages"] = [AIMessage(content=response)]
            state["final_response"] = response
            state["pending_ticket"] = None
            state["next_agent"] = "END"
            return state

        # Success message
        response = (
            f"✅ **Ticket Created!**\n\n"
            f"Ticket ID: **{tid}**\n"
            f"Issue: '{ticket_info['query']}'\n\n"
            f"A human agent will contact you shortly."
        )
        state["messages"] = [AIMessage(content=response)]
        state["final_response"] = response
        state["pending_ticket"] = None # Clear pending ticket state
        state["next_agent"] = "END"
        return state

    # --- TICKET OFFER LOGIC (Initial Escalation/Routing to 'Other') ---
    ticket_info = state.get("pending_ticket") or {}
    query = ticket_info.get("query", state["current_query"])
    source = ticket_info.get("source", "OTHER")
    # Determine the reason for escalation/ticket offer
    intro = {
        "OTHER": "I can't classify your request into IT or HR.",
        "IT_KB_UNAVAILABLE": "I can't access the **IT documentation** right now.",
        "HR_KB_UNAVAILABLE": "I can't access the **HR policy documentation** right now.",
        "IT": "I couldn't find a definitive solution in the **IT documentation**.",
        "HR": "I couldn't find a definitive solution in the **HR documentation**.",
    }.get(source, "I need a human agent to help you further.")
    
    confirmation_msg = f"{intro}\n\n🎫 Would you like me to create a support ticket for:\n\n**\"{query}\"**\n\nPlease reply **Yes** or **No**."
    
    state["messages"] = [AIMessage(content=confirmation_msg)]
    state["final_response"] = confirmation_msg
    # Set pending_ticket to be checked by the supervisor on the next turn
    state["pending_ticket"] = {"query": query, "source": source}
    state["next_agent"] = "END"
    return state

def router(state: AgentState) -> Literal["small_talk_agent", "it_agent", "hr_agent", "other_agent", "END"]:
    """Conditional edge function: routes the workflow based on the 'next_agent' state value."""
    return state.get("next_agent", "END")

# ---------- build & compile ----------
def _build_graph():
    """Defines the LangGraph workflow structure and compiles it."""
    w = StateGraph(AgentState)
    nodes = {"supervisor": supervisor_agent, "small_talk_agent": small_talk_agent,
             "it_agent": it_agent, "hr_agent": hr_agent, "other_agent": other_agent}
    
    # Add all agent functions as nodes
    for name, func in nodes.items():
        w.add_node(name, func)
        
    w.set_entry_point("supervisor") # Set the starting node
    
    # Add conditional edges from the supervisor using the router function
    w.add_conditional_edges("supervisor", router, {**{k: k for k in nodes if k != "supervisor"}, "END": END})
    
    # All other nodes lead directly to the END of the current turn
    for n in ["small_talk_agent", "it_agent", "hr_agent", "other_agent"]:
        w.add_edge(n, END)
        
    # Compile the graph with persistence enabled
    return w.compile(checkpointer=memory)

graph = _build_graph()
# Debug print to confirm paths
print("🗂️  Working dir:", os.getcwd(), "CSV path:", os.path.abspath(TICKET_LOG_FILE))

# ==================  PUBLIC API  ==================
def send_message(user_id: str, text: str) -> str:
    """Send a message and get response, maintaining conversation history."""
    config = {"configurable": {"thread_id": user_id}}
    
    try:
        # Retrieve the current state to check for pending ticket confirmation
        current_state = graph.get_state(config)
        pending = current_state.values.get("pending_ticket") if current_state.values else None
    except Exception:
        pending = None
    
    # Define the initial state for the graph run
    input_state = {
        "messages": [HumanMessage(content=text)], # Add new user message
        "user_id": user_id,
        "current_query": text,
        "next_agent": "",
        "pending_ticket": pending, # Pass pending status to the new run
        "final_response": None
    }
    
    # Stream the graph execution and yield the final response chunk
    for chunk in graph.stream(input_state, config, stream_mode="values"):
        if chunk.get("final_response"):
            yield chunk["final_response"]

def get_history(user_id: str) -> List[BaseMessage]:
    """Get full conversation history for a user from the checkpointer."""
    config = {"configurable": {"thread_id": user_id}}
    try:
        state = graph.get_state(config)
        return state.values.get("messages", []) if state.values else []
    except Exception:
        return []

def list_threads() -> List[str]:
    """List all unique conversation threads (user IDs) from the SQLite checkpointer."""
    try:
        cur = conn.cursor()
        # Query distinct thread IDs sorted by most recent checkpoint
        cur.execute("SELECT DISTINCT thread_id FROM checkpoints ORDER BY checkpoint_id DESC")
        return [row[0] for row in cur.fetchall()]
    except sqlite3.OperationalError:
        return []

def get_agent_state(user_id: str) -> dict:
    """Get the current agent state dictionary for a user thread."""
    config = {"configurable": {"thread_id": user_id}}
    try:
        state = graph.get_state(config)
        return state.values if state.values else {}
    except Exception:
        return {}