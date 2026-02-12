import streamlit as st
import io
from PIL import Image
import os
import google.generativeai as genai
from typing import Optional

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
if "chat_session" not in st.session_state:
    st.session_state.chat_session = None
if "estimated_price" not in st.session_state:
    st.session_state.estimated_price = None

def initialize_gemini(api_key: str):
    """Initialize the Gemini model."""
    try:
        genai.configure(api_key=api_key)
        # Use a model that supports vision, e.g., gemini-1.5-flash or gemini-1.5-pro
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

# --- Sidebar Configuration ---
with st.sidebar:
    st.title("🏠 Property Details")
    
    # API Key Input
    api_key = st.text_input("Gemini API Key", type="password", help="Enter your Google Gemini API Key")
    if not api_key:
        api_key = os.getenv("GOOGLE_API_KEY") 
        if not api_key:
            st.warning("Please enter your API Key to proceed.")
            
    st.divider()
    
    # Property Inputs
    zone = st.text_input("📍 Zone / Neighborhood", placeholder="e.g., Downtown, Suburbs...")
    age = st.number_input("🏗️ Age of Property (Years)", min_value=0, step=1, value=10)
    surface_area = st.number_input("qm Surface Area (m²)", min_value=10, step=1, value=100)
    condition = st.selectbox("✨ Condition", ["New", "Excellent", "Good", "Needs Renovation", "Poor"])
    additional_details = st.text_area("📝 Additional Details", placeholder="e.g., Near park, quiet street...")
    
    st.divider()
    
    # Image Upload
    plan_file = st.file_uploader("📂 Upload Cadastral Plan", type=["jpg", "jpeg", "png"])

# --- Main Content ---
st.title("🤖 Real Estate AI Appraiser")
st.markdown("Upload a cadastral plan and provide details to get an AI-powered price estimation and analysis.")

if api_key and plan_file:
    # Display the uploaded image
    image = Image.open(plan_file)
    st.image(image, caption="Uploaded Cadastral Plan", use_column_width=True)
    
    # Configuration for analysis
    if st.button("💰 Estimate Price & Analyze"):
        with st.spinner("Analyzing plan and market data..."):
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
                    - Additional Notes: {additional_details}
                    
                    Please provide:
                    1. A realistic price estimation range based on the visual layout and details.
                    2. A detailed analysis of the layout (strengths/weaknesses).
                    3. Suggestions for increasing value (renovations, etc.).
                    4. Any potential red flags visible in the plan.
                    
                    Format the output nicely in Markdown.
                    """,
                    image
                ]
                
                response_text = get_gemini_response(model, prompt)
                
                # Store text in session state for chat context
                st.session_state.estimated_price = response_text
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                
                # Start a chat session with history if needed, for now we just append to messages
                # For a true chat with history including the image, we'd need to maintain the history list for Gemini.
                # Simplified: we just feed the last context in new prompts or trust the user refers to it.
                # To keep it simple and stateless for the MVP, we will just use the chat interface below.

    # --- Result Display ---
    if st.session_state.estimated_price:
        st.markdown("### 📊 Analysis Result")
        st.markdown(st.session_state.estimated_price)
        st.divider()

    # --- Chat Interface ---
    st.subheader("💬 Chat with your AI Agent")

    # Display chat messages
    for message in st.session_state.messages:
        # Skip the initial analysis if it's already displayed above? 
        # Actually standard practice is to show it in chat or above. Let's show chat history.
        if message["content"] != st.session_state.estimated_price: # visual cleanup, don't duplicate if we just showed it
             with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask more about this property..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.spinner("Thinking..."):
            model = initialize_gemini(api_key)
            if model:
                # We need to send history. 
                # Gemini support for history + images can be tricky in one-shot, but we can try sending text history.
                # Best approach for "Chat with Image": send image + history every time or use start_chat if supported with images.
                # Simple approach: Text-only follow up context + Image.
                
                history_prompt = [
                    "You are a helpful Real Estate Assistant. Here is the context of our conversation so far:",
                    f"Initial Analysis: {st.session_state.estimated_price}",
                ]
                
                # Add recent history (last 5 messages)
                for msg in st.session_state.messages[-5:]:
                    history_prompt.append(f"{msg['role'].upper()}: {msg['content']}")
                
                history_prompt.append(f"USER: {prompt}")
                history_prompt.append("ASSISTANT:")
                # We add the image again to ensure context is kept 
                # (Gemini 1.5 Flash is cheap and fast, so re-sending image is okay for this demo)
                final_input = [ "\n".join(history_prompt), image ]
                
                response = get_gemini_response(model, final_input)
                
                with st.chat_message("assistant"):
                    st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

else:
    if not api_key:
        st.info("👋 Please enter your Gemini API Key in the sidebar to start.")
    if not plan_file:
        st.info("👋 Please upload a cadastral plan image in the sidebar.")
