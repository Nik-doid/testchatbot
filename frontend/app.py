import streamlit as st
import requests
import uuid
from datetime import datetime

API_URL = "http://localhost:8000/chat"

st.set_page_config(page_title="ClassyBot", layout="centered")
st.title("ClassyBot")
st.markdown("Ask anything about Classic Tech 📡")

# Initialize session state variables
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "ask_triggered" not in st.session_state:
    st.session_state.ask_triggered = False
if "current_input" not in st.session_state:
    st.session_state.current_input = ""

# --- Function to format and style messages ---
def render_message(role, content, timestamp):
    time_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
    if role == "user":
      
        st.markdown(f"""
        <div style="
            background-color:#DCF8C6;
            color:black; 
            padding:10px; 
            border-radius:10px; 
            margin-bottom:5px; 
            max-width:70%;
            text-align:right;
            float:right;
            clear:both;">
            <b>You</b> <small style="color:gray;">{time_str}</small><br>
            {content}
        </div>
        <div style="clear:both;"></div>
        """, unsafe_allow_html=True)
    else:
        # Bot message bubble style
        st.markdown(f"""
        <div style="
            background-color:#F1F0F0; 
            color:black;
            padding:10px; 
            border-radius:10px; 
            margin-bottom:5px; 
            max-width:70%;
            text-align:left;
            float:left;
            clear:both;">
            <b>Bot</b> <small style="color:gray;">{time_str}</small><br>
            {content}
        </div>
        <div style="clear:both;"></div>
        """, unsafe_allow_html=True)



# Input text field
user_input = st.text_input("Ask here:", key="input_text", placeholder="Type your question...", value=st.session_state.current_input)

# Ask button
if st.button("Ask") and user_input.strip():
    st.session_state.ask_triggered = True
    st.session_state.current_input = user_input.strip()

# Process question after rerun
if st.session_state.ask_triggered:
    question = st.session_state.current_input
    # Append user message with timestamp
    st.session_state.messages.append({
        "role": "user",
        "content": question,
        "timestamp": datetime.now()
    })

    try:
        res = requests.post(API_URL, json={
            "query": question,
            "session_id": st.session_state.session_id
        })
        if res.status_code == 200:
            bot_reply = res.json()["response"]
        else:
            bot_reply = f"Error: {res.json().get('detail', 'Unknown error')}"
    except Exception as e:
        bot_reply = f"Request failed: {e}"

    # Append bot reply with timestamp
    st.session_state.messages.append({
        "role": "bot",
        "content": bot_reply,
        "timestamp": datetime.now()
    })

    # Reset input and ask_triggered
    st.session_state.current_input = ""
    st.session_state.ask_triggered = False

    # Rerun to update UI immediately

for msg in st.session_state.messages:
    render_message(msg["role"], msg["content"], msg["timestamp"])


# Clear button
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.session_state.current_input = ""
    st.session_state.ask_triggered = False
  
