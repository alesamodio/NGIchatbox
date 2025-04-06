import streamlit as st
from Agent import iterative_rag_agent

st.set_page_config(page_title="NGI RAG Chatbot", page_icon="🤖", layout="wide")
st.markdown("""
<style>
    html, body, .stApp, .block-container {
        background-color: #ffffff !important;
    }
    .divider-line {
        border-top: 2px solid #ccc;
        width: 80%;
        margin-top: 20px;
        margin-bottom: 20px;
        margin-left: 85px;
        margin-right: 85px;
    }
    .bottom-divider {
        border-top: 2px solid #ccc;
        margin: 20px 0;
    }
    .stChatInputContainer {
        max-width: 80% !important;
        margin-left: 85px !important;
    }
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 10], gap="small")
with col1:
    st.image("ngi-logo.jpg", width=150)

col1, col2 = st.columns([1, 10], gap="small")
with col2:
    st.markdown("""
<h1 style='color:#8B0000; font-family:Calibri, sans-serif; display: flex; align-items: center; height: 70px; margin-left: 85px;'>RAG Chatbot</h1>
<div class='divider-line'></div>
""",
        unsafe_allow_html=True
    )

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        styled_message = f"""
        <div style='background-color:#f5f5f5; padding:15px; border-radius:10px; color:#333333;'>
        {msg["content"]}
        </div>
        """
        st.markdown(styled_message, unsafe_allow_html=True)

# Input box for new user query
user_input = st.chat_input("Ask a question about your documents...")
if user_input:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        styled_user_input = f"""
        <div style='background-color:#f5f5f5; padding:15px; border-radius:10px; color:#333333;'>
        {user_input}
        </div>
        """
        st.markdown(styled_user_input, unsafe_allow_html=True)

    # Run RAG agent
    with st.spinner("Retrieving and thinking..."):
        result = iterative_rag_agent(user_input)

    # Display response
    response = result["answer"]
    with st.chat_message("assistant"):
        styled_response = f"""
        <div style='background-color:#f5f5f5; padding:15px; border-radius:10px; color:#333333;'>
        {response}
        </div>
        """
        st.markdown(styled_response, unsafe_allow_html=True)

        # Optionally show sources
        if result.get("references"):
            with st.expander("📚 References"):
                for ref in result["references"]:
                    st.markdown(f"- {ref}")

    # Save response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Show refinement suggestion if needed
    if result.get("needs_refinement"):
        st.info(f"💡 Suggested refinement: `{result['suggested_query']}`")
