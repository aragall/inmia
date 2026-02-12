import streamlit as st
import io
from PIL import Image
import os
import google.generativeai as genai
from typing import Optional
import fitz  # PyMuPDF

# RAG Dependencies
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Configure page settings
st.set_page_config(
    page_title="Real Estate AI Expert",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #FF4B4B;
        color: white;
    }
    .chat-message {
        padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
    }
    .chat-message.user {
        background-color: #2b313e;
        color: #ffffff;
    }
    .chat-message.bot {
        background-color: #f0f2f6;
        color: #000000;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "estimated_price" not in st.session_state:
    st.session_state.estimated_price = None
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# --- Gemini Configuration ---
def initialize_gemini(api_key: str):
    """Initialize the Gemini model."""
    try:
        genai.configure(api_key=api_key)
        # Use a model that supports vision
        model = genai.GenerativeModel('gemini-3-flash-preview') 
        return model
    except Exception as e:
        st.error(f"Failed to initialize Gemini: {str(e)}")
        return None

def get_gemini_response(model, prompt_parts):
    """Get response from Gemini."""
    try:
        response = model.generate_content(prompt_parts)
        return response.text
    except Exception as e:
        return f"Error generating response: {str(e)}"

# --- RAG Logic ---
def process_pdf_rag(uploaded_file, hf_token):
    """
    Extracts text and images from PDF.
    Creates a Vector Store (FAISS) with Hugging Face Embeddings for the text.
    Returns: extracted_text (full), largest_image, vector_store
    """
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text_content = ""
    largest_image = None
    max_area = 0

    # 1. Extract Content (Text + Image)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text_content += page.get_text()
        
        # Extract images
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            try:
                pil_image = Image.open(io.BytesIO(image_bytes))
                width, height = pil_image.size
                area = width * height
                if area > max_area:
                    max_area = area
                    largest_image = pil_image
            except Exception:
                continue

    # 2. Create Vector Store (RAG)
    vector_store = None
    if text_content and hf_token:
        try:
            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = text_splitter.split_text(text_content)
            
            if chunks:
                # Initialize HF Embeddings
                # Using a standard lightweight model
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': False},
                    # Ensure token is used if required by specific closed models, 
                    # but MiniLM is public. We pass it just in case or for private repos.
                    # langchain_huggingface handles token via env or param if supported.
                )
                
                # Create FAISS Vector Store
                vector_store = FAISS.from_texts(chunks, embedding=embeddings)
        except Exception as e:
            st.error(f"Failed to create Vector Store: {str(e)}")

    return text_content, largest_image, vector_store

def retrieve_context(vector_store, query):
    """Retrieves relevant text chunks for a query."""
    if not vector_store:
        return ""
    docs = vector_store.similarity_search(query, k=3)
    return "\n".join([doc.page_content for doc in docs])

# --- Sidebar Configuration ---
with st.sidebar:
    st.title("🏠 Property Details")
    
    # API Keys
    api_key = st.text_input("Gemini API Key", type="password", help="Enter your Google Gemini API Key")
    hf_token = st.text_input("Hugging Face Token", type="password", help="Enter your Hugging Face Token for RAG embeddings")

    if not api_key:
        api_key = os.getenv("GOOGLE_API_KEY") 

    st.divider()
    
    # Property Inputs
    zone = st.text_input("📍 Zone / Neighborhood", placeholder="e.g., Downtown, Suburbs...")
    age = st.number_input("🏗️ Age of Property (Years)", min_value=0, step=1, value=10)
    surface_area = st.number_input("qm Surface Area (m²)", min_value=10, step=1, value=100)
    condition = st.selectbox("✨ Condition", ["New", "Excellent", "Good", "Needs Renovation", "Poor"])
    additional_details = st.text_area("📝 Additional Details", placeholder="e.g., Near park, quiet street...")
    
    st.divider()
    
    # Image/PDF Upload
    uploaded_file = st.file_uploader("📂 Upload Cadastral Plan (Image or PDF)", type=["jpg", "jpeg", "png", "pdf"])

# --- Main Content ---
st.title("🤖 Real Estate AI Appraiser (RAG Enabled)")
st.markdown("Upload a cadastral plan (Image or PDF) to get an AI-powered price estimation.")

if api_key and uploaded_file:
    # Process the file
    file_type = uploaded_file.type
    
    if "pdf" in file_type:
        if not hf_token:
             st.warning("⚠️ Please provide a Hugging Face Token to enable RAG (Text Analysis). Processing image only for now.")
        
        # Only process if we haven't already or if it's a new file (simplified check)
        # In a real app we'd check file_id. For now, button press triggers analysis.
        pass
    else:
        # It's an image
        image = Image.open(uploaded_file)
        st.session_state.uploaded_image = image
        st.session_state.vector_store = None # No text context for image upload

    # --- UI Logic ---
    if st.button("💰 Estimate Price & Analyze"):
        with st.spinner("Processing & Analyzing..."):
            
            # 1. Handle PDF Processing (RAG) if needed
            if "pdf" in file_type and not st.session_state.uploaded_image:
                 text, img, v_store = process_pdf_rag(uploaded_file, hf_token)
                 if img:
                     st.session_state.uploaded_image = img
                     st.session_state.vector_store = v_store
                     st.success("✅ Extracted Cadastral Plan & Text from PDF.")
                 else:
                     st.error("❌ Could not find an image in the PDF.")
                     st.stop()
            elif "image" in file_type:
                # Ensure image is loaded
                 st.session_state.uploaded_image = Image.open(uploaded_file)
            
            # Display Image
            st.image(st.session_state.uploaded_image, caption="Analyzed Plan", use_column_width=True)

            # 2. Retrieve Context (RAG)
            rag_context = ""
            if st.session_state.vector_store:
                # Query the vector store for relevant property details
                query = f"property details surface area year built location price zoning {zone}"
                rag_context = retrieve_context(st.session_state.vector_store, query)
                with st.expander("� RAG Context (Retrieved from PDF)"):
                    st.text(rag_context)

            # 3. Gemini Analysis
            model = initialize_gemini(api_key)
            if model:
                # Construct the prompt
                prompt = [
                    f"""
                    You are an expert Real Estate Appraiser and Architect.
                    Analyze the uploaded cadastral plan image and the provided details to estimate the property price.
                    
                    Property Details:
                    - Zone: {zone}
                    - Age: {age} years
                    - Surface Area: {surface_area} m²
                    - Condition: {condition}
                    - Additional User Notes: {additional_details}
                    
                    Document Context (Retrieved via RAG):
                    {rag_context}
                    
                    Please provide:
                    1. A realistic price estimation range based on the visual layout and details.
                    2. A detailed analysis of the layout (strengths/weaknesses).
                    3. Suggestions for increasing value (renovations, etc.).
                    4. Any potential red flags visible in the plan.
                    
                    Format the output nicely in Markdown.
                    """,
                    st.session_state.uploaded_image
                ]
                
                response_text = get_gemini_response(model, prompt)
                
                # Store text in session state for chat context
                st.session_state.estimated_price = response_text
                st.session_state.messages.append({"role": "assistant", "content": response_text})

    # --- Result Display ---
    if st.session_state.estimated_price:
        st.markdown(st.session_state.estimated_price)
        st.divider()

    # --- Chat Interface ---
    st.subheader("💬 Chat with your AI Agent")

    # Display Chat History
    for message in st.session_state.messages:
        if message["content"] != st.session_state.estimated_price: 
             with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Chat Input
    if prompt := st.chat_input("Ask more about this property..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Thinking..."):
            model = initialize_gemini(api_key)
            if model and st.session_state.uploaded_image:
                
                # Retrieve fresh RAG context for the specific question
                current_rag_context = ""
                if st.session_state.vector_store:
                    current_rag_context = retrieve_context(st.session_state.vector_store, prompt)

                history_prompt = [
                    "You are a helpful Real Estate Assistant. Answer using the chat history, image, and retrieved context.",
                    f"Initial Analysis: {st.session_state.estimated_price}",
                    f"Relevant Document Context: {current_rag_context}"
                ]
                
                # Add recent history
                for msg in st.session_state.messages[-5:]:
                    history_prompt.append(f"{msg['role'].upper()}: {msg['content']}")
                
                history_prompt.append(f"USER: {prompt}")
                history_prompt.append("ASSISTANT:")
                
                final_input = [ "\n".join(history_prompt), st.session_state.uploaded_image ]
                
                response = get_gemini_response(model, final_input)
                
                with st.chat_message("assistant"):
                    st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

else:
    if not api_key:
        st.info("👋 Please enter your Gemini API Key in the sidebar.")
    if not uploaded_file:
        st.info("👋 Please upload a cadastral plan (Image or PDF).")
