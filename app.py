"""Main Streamlit application for Document Intelligence."""

import streamlit as st
from typing import Optional

from src.doc_intelligence.app import DocumentIntelligenceApp

# Page configuration
st.set_page_config(
    page_title="Document Intelligence Chat",
    page_icon="🟧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Databricks professional theme
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&display=swap');
    
    /* Global font and size */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        font-size: 14px;
    }
    
    /* Databricks color palette */
    :root {
        --databricks-primary: #FF6B35;
        --databricks-secondary: #00A9FF;
        --databricks-dark: #1e1e1e;
        --databricks-light: #f8f9fa;
        --databricks-grey: #6c757d;
    }
    
    /* Sidebar styling */
    .stSidebar > div:first-child {
        padding-top: 16px;
    }
    
    /* Ensure sidebar controls are visible */
    button[title="Close sidebar"], button[title="Open sidebar"] {
        visibility: visible !important;
        opacity: 1 !important;
        position: relative !important;
        z-index: 999999 !important;
    }
    
    /* Sidebar title styling */
    .sidebar-title {
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        gap: 8px;
        margin-bottom: 20px;
    }
    
    /* Databricks logo styling */
    .sidebar-title svg {
        width: 160px;
        max-height: 96px;
        max-width: 100%;
        object-fit: contain;
    }
    
    /* Sidebar sections */
    .sidebar-section {
        margin-bottom: 16px;
    }
    
    /* Compact button spacing */
    .stSidebar .stButton {
        margin-bottom: 6px;
    }
    
    /* Welcome message font */
    .stSidebar .stMarkdown p {
        font-size: 10px;
        color: var(--databricks-grey);
    }
    
    /* Conversation history tiles - smaller */
    .stSidebar .stButton > button:not([kind="primary"]) {
        font-size: 10px;
        padding: 4px 8px;
        min-height: 28px;
        line-height: 1.1;
        text-align: left;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    
    /* Delete buttons - even smaller */
    .stSidebar .stColumns .stButton > button {
        font-size: 10px;
        padding: 3px 6px;
        min-height: 24px;
    }
    
    /* Hide footer but keep dev tools visible */
    footer {visibility: hidden;}
    #MainMenu, [data-testid="stDecoration"], [data-testid="collapsedControl"], stApp > header, stActionButton {
        visibility: visible !important;
        opacity: 1 !important;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def get_app():
    """Initialize and cache the Document Intelligence app."""
    return DocumentIntelligenceApp()


def main():
    """Main application entry point."""

    # Initialize app
    app = get_app()

    # Initialize session state
    if "app_initialized" not in st.session_state:
        st.session_state.current_conversation_id = None
        st.session_state.current_document_list = []
        st.session_state.chat_messages = []
        st.session_state.app_initialized = True

    # Render sidebar
    render_sidebar(app)

    # Render main chat interface
    render_chat_interface(app)


def render_sidebar(app: DocumentIntelligenceApp):
    """Render the sidebar with navigation and controls."""

    with st.sidebar:
        # Databricks logo and title
        logo_path = "fixtures/stacked-lockup-full-color-rgb.svg"
        with open(logo_path, "r") as f:
            logo_svg = f.read()

        st.image(logo_path, use_container_width=True)
        st.title("Document Intelligence")

        # Welcome message
        current_user = app.get_current_user()
        st.text(f"Welcome, {current_user}")

        # Main Services Status
        with st.expander("Service Status", expanded=False):
            status = app.get_system_status()
            for service, details in status["services"].items():
                service_name = service.replace("_", " ").title()
                if details["available"]:
                    st.success(f"✅ {service_name}")
                else:
                    message = details.get("message", "Not available")
                    st.warning(f"⚠️ {service_name}: {message}")

        # New conversation button
        if st.button("🆕 New Conversation", type="primary", use_container_width=True):
            start_new_conversation(app)

        # Conversation history
        render_conversation_history(app)


def render_conversation_history(app: DocumentIntelligenceApp):
    """Render conversation history in sidebar."""

    st.markdown("### Recent Conversations")

    try:
        conversations = app.get_user_conversations()

        if not conversations:
            st.markdown("*No previous conversations*")
            return

        # Show up to 15 conversations
        for conv in conversations[:15]:
            conv_id = conv["conversation_id"]
            title = conv["title"]

            # Truncate title to fit
            display_title = title[:22] + "..." if len(title) > 22 else title

            col1, col2 = st.columns([4, 1])

            with col1:
                if st.button(display_title, key=f"conv_{conv_id}"):
                    load_conversation(app, conv_id)

            with col2:
                if st.button("🗑️", key=f"del_{conv_id}"):
                    if app.delete_conversation(conv_id):
                        st.rerun()

    except Exception as e:
        st.error(f"Error loading conversations: {str(e)}")


def render_chat_interface(app: DocumentIntelligenceApp):
    """Render the main chat interface."""

    # Document upload section
    with st.expander(
        "📎 Upload Document", expanded=len(st.session_state.current_document_list) == 0
    ):
        uploaded_file = st.file_uploader(
            "Choose a document to analyze",
            type=["pdf", "txt", "docx", "md"],
            help="Upload a document to start chatting about its content",
        )

        if uploaded_file is not None:
            if st.button("Process Document", type="primary"):
                handle_document_upload(app, uploaded_file)

    # Current context display
    if st.session_state.current_document_list:
        with st.container():
            st.markdown("**Current Context:**")
            for doc_hash in st.session_state.current_document_list:
                doc_info = app.get_document_info(doc_hash)
                if doc_info:
                    st.markdown(f"📄 {doc_info['filename']} ({doc_info['status']})")

    # Chat interface
    chat_container = st.container()

    with chat_container:
        # Display chat messages
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input(
            "Ask a question about your documents or chat generally..."
        ):
            # Add user message to chat
            st.session_state.chat_messages.append({"role": "user", "content": prompt})

            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = generate_response(app, prompt)
                    st.markdown(response)

            # Add assistant response to chat
            st.session_state.chat_messages.append(
                {"role": "assistant", "content": response}
            )


def start_new_conversation(app: DocumentIntelligenceApp):
    """Start a new conversation."""
    st.session_state.current_conversation_id = None
    st.session_state.current_document_list = []
    st.session_state.chat_messages = []
    st.rerun()


def load_conversation(app: DocumentIntelligenceApp, conversation_id: str):
    """Load an existing conversation."""
    try:
        # Get conversation history
        messages = app.get_conversation_history(conversation_id)

        # Get conversation details to find associated documents
        conversations = app.get_user_conversations()
        current_conv = None
        for conv in conversations:
            if conv["conversation_id"] == conversation_id:
                current_conv = conv
                break

        if current_conv:
            st.session_state.current_conversation_id = conversation_id
            st.session_state.current_document_list = current_conv.get(
                "document_hashes", []
            )
            st.session_state.chat_messages = [
                {"role": msg["role"], "content": msg["content"]} for msg in messages
            ]
            st.rerun()

    except Exception as e:
        st.error(f"Error loading conversation: {str(e)}")


def handle_document_upload(app: DocumentIntelligenceApp, uploaded_file):
    """Handle document upload and processing."""
    try:
        # Read file content
        file_content = uploaded_file.read()
        filename = uploaded_file.name

        with st.spinner(f"Processing {filename}..."):
            # Process document
            result = app.upload_and_process_document(
                file_content=file_content, filename=filename
            )

            if result["success"]:
                doc_hash = result["doc_hash"]

                # Check if we're in an existing conversation
                if st.session_state.current_conversation_id:
                    # Add to existing conversation
                    if doc_hash not in st.session_state.current_document_list:
                        st.session_state.current_document_list.append(doc_hash)
                        app.add_documents_to_conversation(
                            st.session_state.current_conversation_id, [doc_hash]
                        )
                else:
                    # Start new conversation with this document
                    conv_result = app.start_new_conversation(
                        document_hashes=[doc_hash], title=f"Chat with {filename}"
                    )

                    if conv_result["success"]:
                        st.session_state.current_conversation_id = conv_result[
                            "conversation_id"
                        ]
                        st.session_state.current_document_list = [doc_hash]
                        st.session_state.chat_messages = []

                st.success(f"✅ {filename} processed successfully!")
                st.rerun()
            else:
                st.error(
                    f"❌ Failed to process document: {result.get('error', 'Unknown error')}"
                )

    except Exception as e:
        st.error(f"❌ Error processing document: {str(e)}")


def generate_response(app: DocumentIntelligenceApp, user_message: str) -> str:
    """Generate AI response to user message."""
    try:
        # Ensure we have a conversation
        if not st.session_state.current_conversation_id:
            # Start new conversation
            conv_result = app.start_new_conversation(
                document_hashes=st.session_state.current_document_list, title="New Chat"
            )

            if conv_result["success"]:
                st.session_state.current_conversation_id = conv_result[
                    "conversation_id"
                ]
            else:
                return "Sorry, I couldn't start a new conversation. Please try again."

        # Send message
        result = app.send_chat_message(
            conversation_id=st.session_state.current_conversation_id,
            user_message=user_message,
        )

        if result["success"]:
            return result["response"]
        else:
            return (
                f"Sorry, I encountered an error: {result.get('error', 'Unknown error')}"
            )

    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}"


if __name__ == "__main__":
    main()
