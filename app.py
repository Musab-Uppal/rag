import streamlit as st
import os
import tempfile
from rag_pipeline import SimpleRAGPipeline
from dotenv import load_dotenv

# Load environment variables
load_dotenv()



# Initialize session state
if 'rag_pipeline' not in st.session_state:
    try:
        st.session_state.rag_pipeline = SimpleRAGPipeline()
        st.session_state.documents_loaded = False
        st.session_state.chunk_count = 3
    except Exception as e:
        st.error(f"Failed to initialize RAG pipeline: {str(e)}")
        st.stop()

# Page configuration
st.set_page_config(
    page_title="Simple RAG Assistant",
    page_icon="📚",
    layout="wide"
)

# Title
st.title("📚 Simple Document Q&A Assistant")
st.markdown("Upload documents and ask questions using Groq LLM")

# Sidebar
with st.sidebar:
    st.header("📂 Document Upload")
    
    uploaded_files = st.file_uploader(
        "Choose files",
        type=['pdf', 'txt', 'docx'],
        accept_multiple_files=True,
        help="Upload PDF, TXT, or DOCX files"
    )
    
    if uploaded_files:
        st.info(f"{len(uploaded_files)} file(s) selected")
    
    if st.button("🚀 Process Documents", type="primary") and uploaded_files:
        with st.spinner("Processing documents..."):
            all_documents = []
            
            for uploaded_file in uploaded_files:
                # Save to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                try:
                    # Load document
                    documents = st.session_state.rag_pipeline.load_document(tmp_path)
                    all_documents.extend(documents)
                    st.success(f"✓ {uploaded_file.name}")
                except Exception as e:
                    st.error(f"✗ {uploaded_file.name}: {str(e)}")
                finally:
                    # Clean up
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
            
            if all_documents:
                try:
                    # Create vector store
                    st.session_state.rag_pipeline.create_vector_store(all_documents)
                    st.session_state.documents_loaded = True
                    st.success(f"✅ Processed {len(all_documents)} document(s) successfully!")
                    st.balloons()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    st.divider()
    
    # Settings
    st.header("⚙️ Settings")
    st.session_state.chunk_count = st.slider(
        "Number of document chunks to use",
        min_value=1,
        max_value=10,
        value=3,
        help="More chunks = more context but potentially more noise"
    )
    
    if st.button("🗑️ Clear All Documents"):
        st.session_state.documents_loaded = False
        st.session_state.rag_pipeline = SimpleRAGPipeline()
        st.success("Documents cleared!")
        st.rerun()

# Main content area
if not st.session_state.documents_loaded:
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.info("👈 Start by uploading documents in the sidebar")
        st.markdown("""
        ### How to use:
        1. Upload PDF, TXT, or DOCX files using the sidebar
        2. Click "Process Documents"
        3. Ask questions about your documents
        
        ### Supported formats:
        - **PDF**: Research papers, reports, articles
        - **TXT**: Plain text files
        - **DOCX**: Microsoft Word documents
        
        ### Example questions you can ask:
        - "What is this document about?"
        - "Summarize the key points"
        - "Find information about [topic]"
        """)
else:
    st.success(f"✅ Documents are loaded and ready!")
    
    # Chat interface
    st.divider()
    st.subheader("💬 Ask a Question")
    
    # Pre-defined questions
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📋 What is this document about?"):
            st.session_state.last_question = "What is this document about? Provide a brief overview."
    with col2:
        if st.button("🎯 Summarize key points"):
            st.session_state.last_question = "Summarize the key points or main takeaways from these documents."
    
    # Chat input
    query = st.chat_input("Type your question here...")
    
    # Use pre-defined question if set
    if 'last_question' in st.session_state:
        query = st.session_state.last_question
        del st.session_state.last_question
    
    if query:
        # Display user question
        with st.chat_message("user"):
            st.markdown(f"**Question:** {query}")
        
        # Get and display answer
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching documents..."):
                try:
                    response = st.session_state.rag_pipeline.query_documents(
                        query, 
                        k=st.session_state.chunk_count
                    )
                    
                    # Display answer
                    st.markdown("**Answer:**")
                    st.write(response["answer"])
                    
                    # Display sources
                    if response["sources"]:
                        with st.expander(f"📚 View sources ({len(response['sources'])} chunks)"):
                            for i, source in enumerate(response["sources"]):
                                st.markdown(f"**Chunk {i+1}** (Similarity: {source['similarity']})")
                                st.caption(f"Source: {source['metadata']['source']} | Page: {source['metadata']['page']}")
                                st.text(source['content'])
                                st.divider()
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.info("Try processing the documents again or use simpler questions.")

