import streamlit as st
import tempfile
import os
from io import BytesIO
from datetime import datetime

# LangChain imports
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import Document

# ReportLab imports
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT, TA_CENTER

def load_document(uploaded_file):
    """Load document from uploaded file"""
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Load based on file type
        if uploaded_file.name.endswith('.pdf'):
            loader = PyPDFLoader(tmp_file_path)
        elif uploaded_file.name.endswith(('.txt', '.md')):
            loader = TextLoader(tmp_file_path)
        else:
            st.error("Unsupported file type. Please upload PDF or TXT files.")
            return None
        
        documents = loader.load()
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return documents
    
    except Exception as e:
        st.error(f"Error loading document: {str(e)}")
        return None

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """Split documents into smaller chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(documents)

def create_summary_chain(api_key, model_name="gpt-3.5-turbo"):
    """Create the summarization chain"""
    llm = ChatOpenAI(
        api_key=api_key,
        model_name=model_name,
        temperature=0.3
    )
    
    # Create a prompt template for summarization with bullet points
    prompt_template = ChatPromptTemplate.from_template("""
    Please provide a comprehensive summary of the following document in bullet point format. 
    Each bullet point should capture key information, main ideas, or important details.
    Make the bullet points concise but informative.
    
    Document content:
    {context}
    
    Summary in bullet points:
    """)
    
    # Create the stuff documents chain
    chain = create_stuff_documents_chain(llm, prompt_template)
    
    return chain

def parse_bullet_points(summary_text):
    """Parse the summary text to extract bullet points"""
    lines = summary_text.strip().split('\n')
    bullet_points = []
    
    for line in lines:
        line = line.strip()
        if line:
            # Remove common bullet point markers
            if line.startswith(('•', '-', '*', '◦')):
                line = line[1:].strip()
            elif line.startswith(tuple(f"{i}." for i in range(1, 100))):
                # Remove numbered list markers
                line = line.split('.', 1)[1].strip() if '.' in line else line
            
            if line:  # Only add non-empty lines
                bullet_points.append(line)
    
    return bullet_points

def create_pdf_summary(bullet_points, original_filename, summary_text):
    """Create PDF with bullet point summary using ReportLab"""
    buffer = BytesIO()
    
    # Create PDF document
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=18
    )
    
    # Get styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor='darkblue'
    )
    
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Heading2'],
        fontSize=12,
        spaceAfter=20,
    )
    
    bullet_style = ParagraphStyle(
        'BulletPoint',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=12,
        leftIndent=20,
        bulletIndent=10,
        bulletFontName='Symbol',
        bulletText='•'
    )
    
    # Build content
    content = []
    
    # Title
    content.append(Paragraph("Document Summary Report", title_style))
    content.append(Spacer(1, 12))
    
    # Document info
    content.append(Paragraph(f"<b>Original Document:</b> {original_filename}", styles['Normal']))
    content.append(Paragraph(f"<b>Summary Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    content.append(Spacer(1, 20))
    
    # Summary section
    content.append(Paragraph("Summary", subtitle_style))
    
    # Add bullet points
    if bullet_points:
        for point in bullet_points:
            if point.strip():
                content.append(Paragraph(f"• {point}", bullet_style))
    else:
        # Fallback: display raw summary if bullet parsing fails
        content.append(Paragraph("Raw Summary:", styles['Heading3']))
        content.append(Spacer(1, 12))
        content.append(Paragraph(summary_text, styles['Normal']))
    
    # Build PDF
    doc.build(content)
    buffer.seek(0)
    
    return buffer

def main():
    st.set_page_config(
        page_title="Document Summarizer",
        page_icon="📄",
        layout="wide"
    )
    
    st.title("📄 Document Summarizer with PDF Export")
    st.markdown("Upload a document to get an AI-generated summary in bullet points, then download as PDF!")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # API Key input
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key"
        )
        
        # Model selection
        model_options = ["gpt-3.5-turbo", "gpt-4o", "gpt-4-turbo-preview"]
        selected_model = st.selectbox("Select Model", model_options)
        
        # Chunk size configuration
        chunk_size = st.slider("Chunk Size", 500, 2000, 1000)
        chunk_overlap = st.slider("Chunk Overlap", 50, 500, 200)
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Upload Document")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['pdf', 'txt', 'md'],
            help="Upload PDF, TXT, or Markdown files"
        )
        
        if uploaded_file is not None:
            st.success(f"File uploaded: {uploaded_file.name}")
            
            # Display file info
            st.info(f"File size: {len(uploaded_file.getvalue())} bytes")
    
    with col2:
        st.header("Summary Options")
        
        if st.button("Generate Summary", type="primary", disabled=not (uploaded_file and api_key)):
            if not api_key:
                st.error("Please enter your OpenAI API key in the sidebar.")
            elif not uploaded_file:
                st.error("Please upload a document first.")
            else:
                with st.spinner("Processing document and generating summary..."):
                    try:
                        # Load and process document
                        documents = load_document(uploaded_file)
                        if documents is None:
                            st.stop()
                        
                        # Split documents
                        doc_chunks = split_documents(documents, chunk_size, chunk_overlap)
                        st.info(f"Document split into {len(doc_chunks)} chunks")
                        
                        # Create summary chain
                        chain = create_summary_chain(api_key, selected_model)
                        
                        # Generate summary
                        summary = chain.invoke({"context": doc_chunks})
                        
                        # Store in session state
                        st.session_state.summary = summary
                        st.session_state.filename = uploaded_file.name
                        st.session_state.bullet_points = parse_bullet_points(summary)
                        
                        st.success("Summary generated successfully!")
                        
                    except Exception as e:
                        st.error(f"Error generating summary: {str(e)}")
    
    # Display summary
    if hasattr(st.session_state, 'summary') and st.session_state.summary:
        st.header("Generated Summary")
        
        # Display bullet points
        if st.session_state.bullet_points:
            st.subheader("Key Points:")
            for point in st.session_state.bullet_points:
                st.markdown(f"• {point}")
        else:
            st.subheader("Summary:")
            st.write(st.session_state.summary)
        
        # PDF Export section
        st.header("Export to PDF")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("Generate PDF", type="secondary"):
                with st.spinner("Creating PDF..."):
                    try:
                        pdf_buffer = create_pdf_summary(
                            st.session_state.bullet_points,
                            st.session_state.filename,
                            st.session_state.summary
                        )
                        st.session_state.pdf_buffer = pdf_buffer
                        st.success("PDF generated successfully!")
                    except Exception as e:
                        st.error(f"Error creating PDF: {str(e)}")
        
        with col2:
            if hasattr(st.session_state, 'pdf_buffer'):
                st.download_button(
                    label="Download PDF Summary",
                    data=st.session_state.pdf_buffer,
                    file_name=f"summary_{st.session_state.filename.split('.')[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )
    
    # Footer
    st.markdown("---")
    st.markdown("Built with Streamlit, LangChain, and ReportLab")

if __name__ == "__main__":
    main()
