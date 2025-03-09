import streamlit as st
import os
import json
import faiss
import torch
import numpy as np
from typing import List, Dict, Any
from datetime import datetime
from transformers import AutoTokenizer, AutoModel, pipeline

class MagicalAIAgent:
    # Your existing MagicalAIAgent class here (same as in magical_ai.py)
    # ... [copy the entire class implementation from the original code]
    def __init__(self, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2", 
                 llm_model_name: str = "google/flan-t5-base",
                 storage_dir: str = "magical_scrolls"):
        """
        Initialize the Magical AI Agent with its crystal ball and spellbooks.
        
        Args:
            embedding_model_name: Model used for creating vector embeddings
            llm_model_name: Model used for generating responses
            storage_dir: Directory to store indexed knowledge
        """
        with st.spinner("🧙 Initializing your magical AI agent..."):
            # Set up storage directories
            self.storage_dir = storage_dir
            self.vector_dir = os.path.join(storage_dir, "vectors")
            self.content_dir = os.path.join(storage_dir, "content")
            
            for directory in [self.storage_dir, self.vector_dir, self.content_dir]:
                os.makedirs(directory, exist_ok=True)
            
            # Load the embedding model (the vector spell)
            self.embedding_model_name = embedding_model_name
            self.tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
            self.model = AutoModel.from_pretrained(embedding_model_name)
            
            # Load the language model (the magical cauldron)
            self.llm = pipeline("text2text-generation", model=llm_model_name, max_length=512)
            
            # Initialize the knowledge crystal ball
            self.initialize_crystal_ball()
            
            # Keep track of all scrolls
            self.scroll_registry = {}
            self.load_scroll_registry()

    # Rest of your MagicalAIAgent methods go here

# Streamlit app
st.set_page_config(page_title="Magical AI Agent", page_icon="🧙")
st.title("🧙 Magical AI Agent")
st.write("Upload JSON files to teach me new knowledge, then ask me questions!")

# Initialize session state
if 'agent' not in st.session_state:
    with st.spinner("Initializing your magical assistant..."):
        st.session_state.agent = MagicalAIAgent()
        st.session_state.message_history = []

# File uploader for JSON files
uploaded_file = st.file_uploader("Upload a JSON scroll", type=["json"])
if uploaded_file is not None:
    # Save the uploaded file temporarily
    temp_file_path = os.path.join("temp_uploads", uploaded_file.name)
    os.makedirs("temp_uploads", exist_ok=True)
    
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    with st.spinner("Learning from your magical scroll..."):
        st.session_state.agent.learn_from_scroll(temp_file_path)
    
    st.success(f"Successfully learned from {uploaded_file.name}!")

# Chat interface
st.subheader("Ask the Magical AI")
user_question = st.text_input("Your question:")

if user_question:
    with st.spinner("The magical AI is thinking..."):
        answer = st.session_state.agent.answer_question(user_question)
    
    # Add to message history
    st.session_state.message_history.append({"question": user_question, "answer": answer})
    
    # Clear the input box
    st.experimental_rerun()

# Display message history
for message in st.session_state.message_history:
    st.write(f"**You:** {message['question']}")
    st.write(f"**🧙‍♂️ Magical AI:** {message['answer']}")
    st.markdown("---")

# Show statistics
with st.sidebar:
    st.subheader("Knowledge Stats")
    st.write(f"Documents in memory: {len(st.session_state.agent.documents)}")
    st.write(f"Scrolls processed: {len(st.session_state.agent.scroll_registry)}")
    
    with st.expander("View scroll registry"):
        st.json(st.session_state.agent.scroll_registry)