import io
import os
import pypdf 
import streamlit as st
from agent import initialize_app
from langchain_groq.chat_models import ChatGroq


st.title("Testcase Generation Agent")

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []
    
with st.sidebar:
    st.header("Configuration")
    
    uploaded_file = st.file_uploader("Upload Requirements Document", type=["txt", "pdf", "docx"])
    if "uploaded_file" not in st.session_state or uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file

    model_options = [
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "llama3-70b-8192",
        "llama3-8b-8192",
        "mixtral-8x7b-32768",
        "gemma2-9b-it"
    ]
    
    # Initialize session state for the model if it doesn't exist
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "llama-3.1-8b-instant"
            
    selected_model = st.selectbox("Select Model", model_options, key="selected_model", index=model_options.index(st.session_state.selected_model))
    
    # Update the model in session state when changed
    if selected_model != st.session_state.selected_model:
        st.session_state.selected_model = selected_model
        # Create a new LLM instance when model changes
        st.session_state.llm = ChatGroq(model=selected_model, temperature=0.0)
    
    # Initialize LLM if it doesn't exist
    if "llm" not in st.session_state:
        st.session_state.llm = ChatGroq(model=selected_model, temperature=0.0)
            
    # Add options for test case generation settings
    st.subheader("Test Generation Settings")
    
    include_edge_cases = st.checkbox("Include Edge Cases", value=True)
    if "include_edge_cases" not in st.session_state or include_edge_cases != st.session_state.include_edge_cases:
        st.session_state.include_edge_cases = include_edge_cases
    
    enhancement_level = st.slider("Test Case Detail Level", min_value=1, max_value=5, value=3)
    if "enhancement_level" not in st.session_state or enhancement_level != st.session_state.enhancement_level:
        st.session_state.enhancement_level = enhancement_level
    
    use_industry_standards = st.checkbox("Apply Industry Best Practices", value=True)
    if "use_industry_standards" not in st.session_state or use_industry_standards != st.session_state.use_industry_standards:
        st.session_state.use_industry_standards = use_industry_standards
    
    test_formats = ["Auto-detect", "Gherkin", "Selenium"]
    test_format = st.selectbox("Default Test Format", test_formats, index=0)
    if "test_format" not in st.session_state or test_format != st.session_state.test_format:
        st.session_state.test_format = test_format
    
    reset_button = st.button("🔄 Reset Conversation", key="reset_button")
    if reset_button:
        st.session_state.messages = []   
        st.rerun()

# Initialize the app
app = initialize_app(model_name=st.session_state.selected_model)

# Add an expander for examples and help
with st.expander("👋 How to use this app"):
    st.markdown("""
    ### Test Case Generation Agent
    
    This app helps you generate test cases from requirements documents. Here's how to use it:
    
    1. **Upload a requirements document** using the sidebar
    2. **Configure settings** for test case generation
    3. **Ask questions** like:
        - "Generate Gherkin test cases for the login feature"
        - "Create Selenium tests for the checkout process"
        - "Generate test cases with edge cases for payment processing"
    
    The agent will analyze your requirements and generate appropriate test cases.
    """)

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
# Get requirements document content
requirements_docs_content = ""
if "uploaded_file" in st.session_state and st.session_state.uploaded_file is not None:
    if st.session_state.uploaded_file.type == "text/plain":
        requirements_docs_content = st.session_state.uploaded_file.getvalue().decode("utf-8")
    elif st.session_state.uploaded_file.type == "application/pdf":
        pdf_reader = pypdf.PdfReader(io.BytesIO(st.session_state.uploaded_file.getvalue()))
        for page in pdf_reader.pages:
            requirements_docs_content += page.extract_text()
elif os.path.exists("./input.txt"):
    try:
        with open("./input.txt", "r", encoding='utf-8') as f:
            requirements_docs_content = f.read()
    except Exception as e:
        st.error(f"Error reading default file: {e}")

st.markdown("---")
st.write("\n")
st.markdown("<div style='text-align: center; color: gray; font-size: 0.95em; margin-bottom:6px;'>Developed by <b>Mayur Sapre</b></div>", unsafe_allow_html=True)
user_request = st.chat_input("Enter your request (e.g., 'Generate Gherkin test cases for the login feature'):")

if user_request:
    # Add user request to chat history
    st.session_state.messages.append({"role": "user", "content": user_request})
    with st.chat_message("user"):
        st.markdown(user_request)

    # Process with AI and get response
    with st.chat_message("assistant"):
        with st.spinner("Generating test cases..."):
            # Include settings in the inputs
            inputs = {
                "user_request": user_request, 
                "requirements_docs_content": requirements_docs_content,
                "include_edge_cases": st.session_state.include_edge_cases,
                "enhancement_level": st.session_state.enhancement_level,
                "use_industry_standards": st.session_state.use_industry_standards,
                "test_format": st.session_state.test_format
            }
            
            # Stream the results
            response_placeholder = st.empty()
            total_answer = ""
            
            progress_bar = st.progress(0)
            progress_text = st.empty()
            
            # Track stages of processing
            stages = ["Analyzing Requirements", "Researching Best Practices", 
                     "Generating Test Cases", "Enhancing with Edge Cases"]
            current_stage = 0
            
            for output in app.stream(inputs):
                for node_name, state in output.items():
                    # Update the progress based on the current node
                    if node_name == "summary_node":
                        current_stage = 0
                    elif node_name == "best_practices_node":
                        current_stage = 1
                    elif node_name in ["gherkin_node", "selenium_node"]:
                        current_stage = 2
                    elif node_name == "enhance_node":
                        current_stage = 3
                    
                    progress_value = (current_stage + 1) / len(stages)
                    progress_bar.progress(progress_value)
                    progress_text.text(f"Step {current_stage+1}/{len(stages)}: {stages[current_stage]}")
                    
                    if 'answer' in state:
                        total_answer = state['answer']
                        response_placeholder.markdown(total_answer)

            # Clear progress indicators when done
            progress_bar.empty()
            progress_text.empty()
            
            # Add final response to chat history
            st.session_state.messages.append({"role": "assistant", "content": total_answer})
