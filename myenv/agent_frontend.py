"""
Streamlit Frontend for CSU Multi-Agent Support Bot
"""
import streamlit as st
import uuid
import agent_backend
import time
from agent_backend import get_history,send_message,list_threads,get_agent_state,send_message
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage


# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def get_thread_title(thread_id: str) -> str:
    """
    Generate a human-readable title for a conversation thread.
    Uses the last user message (truncated to 10 chars) as the title.
    
    Args:
        thread_id: Unique identifier for the conversation thread
        
    Returns:
        str: First 10 characters of last user message, or "New conversation"
    """
    history = agent_backend.get_history(thread_id)
    
    # Reverse the history to get the last user message first
    for msg in reversed(history):
        if isinstance(msg, HumanMessage):
            content = msg.content[:10]
            return content + ("…" if len(msg.content) > 10 else "")
    
    # If no user messages found, check for any message
    for msg in reversed(history):
        if hasattr(msg, 'content'):
            content = msg.content[:10]
            return content + ("…" if len(msg.content) > 10 else "")
    
    return "New conversation"


def switch_thread(thread_id: str):
    """
    Switch to a different conversation thread.
    Updates session state and triggers a rerun.
    
    Args:
        thread_id: ID of the thread to switch to
    """
    st.session_state["thread"] = thread_id
    st.session_state["initialized"] = True
    st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="CSU Support Bot",
    layout="centered",
    initial_sidebar_state="expanded"
)


# ═══════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════

# Ensure a thread ID exists at the top level of the script
# This prevents thread from being recreated on every rerun
if "thread" not in st.session_state:
    st.session_state["thread"] = str(uuid.uuid4())
    st.session_state["initialized"] = False


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR - CONVERSATION MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("Conversations")
    
    # New chat button - creates a fresh conversation thread
    if st.button("➕ New chat", use_container_width=True):
        st.session_state["thread"] = str(uuid.uuid4())
        st.session_state["initialized"] = False
        st.rerun()
    
    st.divider()
    st.subheader("My Conversation")
    
    # Fetch all conversation threads from database
    threads = agent_backend.list_threads()
    
    # Display threads in reverse order (most recent first)
    if threads:
        for thread_id in reversed(threads):
            # Generate a readable title for the thread
            label = get_thread_title(thread_id)
            
            # Highlight the currently active thread
            if thread_id == st.session_state["thread"]:
                label = f"▶️ {label}"
            
            # Button to switch to this thread
            if st.button(label, key=f"btn_{thread_id}", use_container_width=True):
                switch_thread(thread_id)
    else:
        st.caption("No previous conversations")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN CHAT AREA
# ═══════════════════════════════════════════════════════════════════════════

# Add title to the main chat area
st.title("🤖 CSU Support Bot")
st.markdown("### How can I help you today?")

# Get the current active thread
thread = st.session_state["thread"]

# Fetch conversation history for the current thread
history = agent_backend.get_history(thread)

# Display all messages in the conversation history
for msg in history:
    if isinstance(msg, HumanMessage):
        # Display ALL user messages including "Hi", "Hello", etc.
        with st.chat_message("user"):
            st.markdown(msg.content)
            
    elif isinstance(msg, AIMessage):
        # Display ALL assistant messages
        with st.chat_message("assistant"):
            st.markdown(msg.content)
            
    elif isinstance(msg, ToolMessage):
        # Display ALL tool/function call messages
        with st.chat_message("assistant"):
            st.caption(f"🔧 *Tool: {msg.name}*")


# ═══════════════════════════════════════════════════════════════════════════
# TICKET CREATION BANNER
# ═══════════════════════════════════════════════════════════════════════════

# Display a success banner if a ticket was just created
final_state = agent_backend.get_agent_state(thread)
final_text = final_state.get("final_response", "")

if final_text and "✅ **Ticket Created!**" in final_text:
    st.success(final_text)


# ═══════════════════════════════════════════════════════════════════════════
# CHAT INPUT WITH STREAMING RESPONSE
# ═══════════════════════════════════════════════════════════════════════════

# Chat input box - always displayed at the bottom
if prompt := st.chat_input("How can we help?"):
    
    # -----------------------------------------------------------------------
    # Step 1: Display the user's message immediately
    # -----------------------------------------------------------------------
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # -----------------------------------------------------------------------
    # Step 2: Send message to backend and get response
    # -----------------------------------------------------------------------
    answer = agent_backend.send_message(thread, prompt)
    
    # -----------------------------------------------------------------------
    # Step 3: Display assistant response with streaming effect
    # -----------------------------------------------------------------------
    with st.chat_message("assistant"):
        # Create a placeholder for dynamic content updates
        message_placeholder = st.empty()
        full_response = ""
        
        # Simulate streaming by displaying character by character
        for char in answer:
            full_response += char
            # Display with a typing cursor effect (▌)
            message_placeholder.markdown(full_response + "▌")
            # Adjust sleep time to control typing speed
            # 0.01 = 10ms per character (fast)
            # 0.03 = 30ms per character (medium)
            # 0.05 = 50ms per character (slow)
            time.sleep(0.01)
        
        # Display final response without the cursor
        message_placeholder.markdown(full_response)
    
    # -----------------------------------------------------------------------
    # Step 4: Rerun to refresh the page and show updated history
    # -----------------------------------------------------------------------
    st.rerun()