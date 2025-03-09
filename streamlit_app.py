import streamlit as st
import os
import json
import numpy as np
import tempfile
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

class SimpleMagicalAIAgent:
    def __init__(self):
        """Initialize a simplified version of the Magical AI Agent for cloud deployment"""
        with st.spinner("🧙 Initializing your magical AI agent..."):
            # Use a smaller embedding model suitable for cloud deployment
            self.embedding_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
            
            # Use a small LLM
            self.llm = pipeline(
                "text2text-generation", 
                model="google/flan-t5-small", 
                max_length=512
            )
            
            # Initialize empty knowledge base
            self.documents = []
            self.embeddings = []
            self.scroll_registry = {}
    
    def embed_text(self, text):
        """Create an embedding for the text"""
        return self.embedding_model.encode(text)
    
    def learn_from_scroll(self, scroll_path):
        """Learn from a JSON file"""
        st.info(f"Reading magical scroll: {os.path.basename(scroll_path)}")
        
        try:
            with open(scroll_path, 'r', encoding='utf-8') as f:
                scroll_data = json.load(f)
            
            # Process each document
            if isinstance(scroll_data, list):
                documents = scroll_data
            else:
                documents = [scroll_data]
            
            # Add timestamp and source to each document
            for doc in documents:
                if "source" not in doc:
                    doc["source"] = os.path.basename(scroll_path)
                doc["indexed_at"] = datetime.now().isoformat()
                
                # Create embedding
                if "content" in doc:
                    text_to_embed = doc["content"]
                    if "title" in doc:
                        text_to_embed = f"{doc['title']}: {text_to_embed}"
                elif "title" in doc:
                    text_to_embed = doc["title"]
                else:
                    text_to_embed = json.dumps(doc)
                
                # Store the document and its embedding
                embedding = self.embed_text(text_to_embed)
                self.documents.append(doc)
                self.embeddings.append(embedding)
            
            # Update the scroll registry
            self.scroll_registry[os.path.basename(scroll_path)] = {
                "doc_count": len(documents),
                "timestamp": datetime.now().isoformat()
            }
            
            st.success(f"✨ Successfully learned {len(documents)} new pieces of knowledge")
            
        except Exception as e:
            st.error(f"Failed to read the magical scroll: {str(e)}")
    
    def answer_question(self, question, top_k=3):
        """Answer a question using the knowledge base"""
        if len(self.documents) == 0:
            return "I haven't learned from any magical scrolls yet. Please provide me with some knowledge first!"
        
        # Create embedding for the question
        question_embedding = self.embed_text(question)
        
        # Calculate similarities
        similarities = cosine_similarity(
            [question_embedding], 
            self.embeddings
        )[0]
        
        # Get top_k most similar documents
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Retrieve the most relevant documents
        relevant_docs = [self.documents[i] for i in top_indices]
        
        # Format context for the LLM
        context = ""
        for i, doc in enumerate(relevant_docs, 1):
            if "title" in doc and "content" in doc:
                context += f"Document {i} - {doc['title']}:\n{doc['content']}\n\n"
            elif "content" in doc:
                context += f"Document {i}:\n{doc['content']}\n\n"
            else:
                context += f"Document {i}:\n{json.dumps(doc, indent=2)}\n\n"
        
        # Construct prompt for the LLM
        prompt = f"""Based on the following information, please answer the question.
        
Information:
{context}

Question: {question}

Answer:"""
        
        # Generate response
        response = self.llm(prompt)[0]['generated_text']
        
        return response

# Set up the Streamlit page
st.set_page_config(page_title="Magical AI Agent", page_icon="🧙")
st.title("🧙 Magical AI Agent")
st.write("Upload JSON files to teach me new knowledge, then ask me questions!")

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = SimpleMagicalAIAgent()
    st.session_state.message_history = []

# File uploader for JSON files
uploaded_file = st.file_uploader("Upload a JSON scroll", type=["json"])
if uploaded_file is not None:
    # Create a temporary file
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, uploaded_file.name)
    
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Learn from the temporary file
    with st.spinner("Learning from your magical scroll..."):
        st.session_state.agent.learn_from_scroll(temp_path)

# Chat interface
st.subheader("Ask the Magical AI")
user_question = st.text_input("Your question:")

if user_question:
    with st.spinner("The magical AI is thinking..."):
        answer = st.session_state.agent.answer_question(user_question)
    
    # Add to message history
    st.session_state.message_history.append({"question": user_question, "answer": answer})
    
    # Reset the input field
    st.text_input("Your question:", value="", key=f"question_{len(st.session_state.message_history)}")

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