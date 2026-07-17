import sys
import os
import torch
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

# Page Config for beautiful aesthetics
st.set_page_config(
    page_title="AI Chat Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for premium look and feel
st.markdown("""
<style>
    /* Dark theme enhancements */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    .css-1d391kg {
        background-color: #161b22;
    }
    h1 {
        color: #58a6ff !important;
        font-family: 'Outfit', 'Inter', sans-serif;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    /* Style st.chat_input to match design */
    .stChatInput {
        border-color: #30363d;
    }
</style>
""", unsafe_allow_html=True)

# Title & Subtitle
st.title("🤖 Local Modern Chat Assistant")
st.markdown("An interactive conversational assistant powered by **Qwen2.5-0.5B-Instruct**, running completely offline on your machine.")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Model details info card
    st.info(
        "**Model**: Qwen2.5-0.5B-Instruct\n\n"
        "**Size**: ~900MB\n\n"
        "This model runs locally. The first launch will download the model weights (~900MB) if not cached."
    )
    
    # Advanced parameters
    st.subheader("Generation Tuning")
    temperature = st.slider(
        "Temperature", 
        min_value=0.1, 
        max_value=1.5, 
        value=0.7, 
        step=0.1,
        help="Higher values yield more creative/random responses, while lower values are more precise and deterministic."
    )
    top_p = st.slider(
        "Top-p (Nucleus)", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.9, 
        step=0.05,
        help="Only tokens with cumulative probability above this threshold are selected."
    )
    max_tokens = st.slider(
        "Max New Tokens", 
        min_value=64, 
        max_value=1024, 
        value=512, 
        step=64,
        help="The maximum number of tokens the model will generate in response."
    )
    
    st.markdown("---")
    
    # Clear conversation state
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = [
            {"role": "system", "content": "You are a helpful, respectful, and honest assistant. Answer in clear, modern English."}
        ]
        st.rerun()

# Cache the tokenizer and model loading so it only runs once per app lifecycle
@st.cache_resource(show_spinner="Loading conversational model (Qwen/Qwen2.5-0.5B-Instruct)...")
def load_model():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    return tokenizer, model

# Load the model
try:
    tokenizer, model = load_model()
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.warning("Please make sure you have internet access (if running for the first time) and that Hugging Face libraries are installed correctly.")
    st.stop()

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful, respectful, and honest assistant. Answer in clear, modern English."}
    ]

# Display all messages in session history (excluding system prompt)
for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# User Chat Input
if user_input := st.chat_input("Type your message here..."):
    # Render user prompt in screen
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Append user prompt to history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Stream the assistant's reply
    with st.chat_message("assistant"):
        # Apply the conversation chat template
        text = tokenizer.apply_chat_template(
            st.session_state.messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize prompt and move to device
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        # Set up thread-safe TextIteratorStreamer for live streaming
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        generation_kwargs = dict(
            **model_inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            streamer=streamer
        )
        
        # Start model generation in background thread
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # Stream output to Streamlit UI in real-time
        response = st.write_stream(streamer)
        
    # Append final assistant reply to history
    st.session_state.messages.append({"role": "assistant", "content": response})
