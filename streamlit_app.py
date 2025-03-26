import os
import re
import langdetect
from datetime import datetime
from io import BytesIO

import streamlit as st
import PyPDF2
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def create_timestamped_filename(output_folder, base_file_name):
    """Create a timestamped filename for the output PDF."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_name = f"{base_file_name}_{timestamp}.pdf"
    full_path = os.path.join(output_folder, file_name)
    return full_path

def extract_text_from_pdf(uploaded_file):
    """Extract text from a PDF file."""
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ''
    for page in pdf_reader.pages:
        content = page.extract_text()
        if content:
            text += content + "\n"
    return text

def generate_summary_prompt():
    """Create a comprehensive summarization prompt."""
    return PromptTemplate(
        input_variables=["text"],
        template="""
**Consolidated Overview Summary**

**Document Name:** {document_name}

**Key Pointers:**

Generate a comprehensive summary focusing on the most critical aspects of the document. Ensure the summary:
- Uses bullet points
- Highlights key governance, strategic, and operational insights
- Captures the core purpose and significant guidelines
- Avoids direct quotes from the original text
- Provides a maximum of 15 key pointers

{text}
"""
    )

def summarize_pdf_documents(uploaded_files, api_key):
    """
    Summarize uploaded PDF documents with a structured format.
    
    Args:
        uploaded_files (list): List of uploaded PDF files
        api_key (str): OpenAI API key
    
    Returns:
        str: Consolidated summary text
    """
    # Validate API key
    if not api_key.startswith("sk-"):
        st.error("Invalid API key format. OpenAI API keys should start with 'sk-'.")
        return None

    try:
        # Initialize LLM
        llm = ChatOpenAI(
            openai_api_key=api_key,
            model_name="gpt-4o-2024-08-06",
            temperature=0.2,
            top_p=0.2
        )

        # Prepare consolidated summary
        consolidated_summary = ""

        # Process each uploaded PDF file
        for uploaded_file in uploaded_files:
            # Extract file name without extension
            document_name = os.path.splitext(uploaded_file.name)[0]
            
            # Extract text from PDF
            text = extract_text_from_pdf(uploaded_file)

            # Chunk the text
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=8000,
                chunk_overlap=500,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            text_chunks = text_splitter.split_text(text)
            docs = [Document(page_content=chunk) for chunk in text_chunks]

            # Create summary prompt with document name
            summary_prompt = PromptTemplate(
                input_variables=["text"],
                template=f"""
**Consolidated Overview Summary**

**Document Name: {document_name}**

**Key Pointers:**

Provide a comprehensive summary focusing on the most critical aspects of the document. Ensure the summary:
- Uses bullet points
- Highlights key governance, strategic, and operational insights
- Captures the core purpose and significant guidelines
- Avoids direct quotes from the original text
- Provides a maximum of 15 key pointers

{{text}}
"""
            )

            # Summarize using map-reduce chain
            map_reduce_chain = load_summarize_chain(
                llm,
                chain_type="map_reduce",
                map_prompt=summary_prompt,
                combine_prompt=summary_prompt
            )
            
            # Generate summary
            output_summary = map_reduce_chain.invoke(docs)
            summary = output_summary['output_text']

            # Append to consolidated summary
            consolidated_summary += summary + "\n\n"

        return consolidated_summary

    except Exception as e:
        st.error(f"An error occurred during summarization: {e}")
        return None

def main():
    st.set_page_config(page_title="PDF Circular Summarizer", page_icon="📄")
    
    st.title("📄 PDF Circular Summarization Tool")
    st.markdown("""
    ### Summary Features
    - Extract text from multiple PDF circulars
    - Generate structured, insightful summaries
    - Powered by GPT-4o
    """)

    # Sidebar for API Key and File Upload
    with st.sidebar:
        st.header("Configuration")
        openai_api_key = st.text_input("OpenAI API Key", type="password")
        
        st.header("Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose PDF files", 
            type=['pdf'], 
            accept_multiple_files=True
        )

    # Main action button
    if st.button("Generate Summary", disabled=not (uploaded_files and openai_api_key)):
        if not uploaded_files:
            st.warning("Please upload PDF files.")
            return
        
        if not openai_api_key:
            st.warning("Please enter your OpenAI API key.")
            return

        # Perform summarization
        summary = summarize_pdf_documents(uploaded_files, openai_api_key)
        
        if summary:
            # Display summary
            st.text_area("Consolidated Summary", summary, height=600)
            
            # Option to download summary
            st.download_button(
                label="Download Summary",
                data=summary,
                file_name="circular_summary.txt",
                mime="text/plain"
            )

if __name__ == "__main__":
    main()
