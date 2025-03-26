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

def extract_english_text(text):
    """Extract English words from the given text."""
    try:
        words = re.findall(r'\b\w+\b', text)
        
        english_words = []
        for word in words:
            try:
                if len(word) > 1:
                    lang = langdetect.detect(word)
                    if lang == 'en':
                        english_words.append(word)
            except langdetect.lang_detect_exception.LangDetectException:
                continue
        
        return ' '.join(english_words)
    
    except Exception as e:
        st.error(f"Language error: {e}")
        return text

def chunk_document(text, chunk_size=8000, chunk_overlap=500):
    """Split the document into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    return text_splitter.split_text(text)

def summarize_pdf_documents(uploaded_files, api_key):
    """
    Summarize uploaded PDF circulars and generate a consolidated PDF summary.
    
    Args:
        uploaded_files (list): List of uploaded PDF files
        api_key (str): OpenAI API key
    
    Returns:
        BytesIO: PDF summary file
    """
    # Validate API key
    if not api_key.startswith("sk-"):
        st.error("Invalid API key format. OpenAI API keys should start with 'sk-'.")
        return None

    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # Initialize LLM
        llm = ChatOpenAI(
            openai_api_key=api_key,
            model_name="gpt-4o-2024-08-06",
            temperature=0.2,
            top_p=0.2
        )

        # PII protection instructions
        pii_instructions = """
        IMPORTANT: DO NOT include any personally identifiable information (PII) in your summary, including:
        - Bank account numbers
        - Credit card numbers
        - Social security numbers
        - Passport numbers
        - Personal mobile numbers
        If you encounter such information, DO NOT include it in your summary.
        """

        # Prompts for summarization
        map_prompt = PromptTemplate(
            input_variables=["text"],
            template=f"""{pii_instructions}Read and summarize the following content in your own words, highlighting the main ideas, purpose, and important insights without including direct phrases or sentences from the original text in 15 bullet points.\n\n{{text}}
            """
        )

        combine_prompt = PromptTemplate(
            input_variables=["text"],
            template="""**Consolidated Overview Summary**

Each summary for a document should start with the document name (without extensions like .pdf or .docx).
Each summary should have a heading named, "Key Pointers:"
Combine the following individual summaries into a cohesive, insightful summary. Ensure that it is concise, capturing the core themes and purpose of the entire document in 15 bullet points:\n\n{text}
            """
        )

        # Prepare PDF output
        pdf_output = BytesIO()
        doc = SimpleDocTemplate(pdf_output, pagesize=A4)
        styles = getSampleStyleSheet()

        # Create a custom bullet point style if not exists
        if 'BulletPoint' not in styles:
            styles.add(ParagraphStyle(
                name='BulletPoint',
                parent=styles['BodyText'],
                firstLineIndent=-14,
                leftIndent=10,
                spaceBefore=6,
                spaceAfter=6,
                bulletIndent=0
            ))

        flowables = []

        # Process each uploaded PDF file
        for idx, uploaded_file in enumerate(uploaded_files, 1):
            status_text.info(f"Processing PDF {idx} of {len(uploaded_files)}")
            progress_bar.progress(idx / len(uploaded_files))

            # Extract text from PDF
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ''
            for page in pdf_reader.pages:
                content = page.extract_text()
                if content:
                    text += content + "\n"

            # Filter text
            filtered_text = extract_english_text(text)

            if filtered_text.strip():
                # Chunk the text
                text_chunks = chunk_document(filtered_text)
                docs = [Document(page_content=chunk) for chunk in text_chunks]

                # Summarize using map-reduce chain
                map_reduce_chain = load_summarize_chain(
                    llm,
                    chain_type="map_reduce",
                    map_prompt=map_prompt,
                    combine_prompt=combine_prompt
                )
                
                output_summary = map_reduce_chain.invoke(docs)
                summary = output_summary['output_text']

                # Format summary for PDF
                summary = re.sub(r'\*\*(.*?)\*\*', lambda m: f'<b>{m.group(1)}</b>', summary)
                paragraphs = summary.split('\n')
                
                for para in paragraphs:
                    if '**' in para:
                        flowables.append(Paragraph(para.replace('**', ''), styles['Heading1']))
                        flowables.append(Spacer(1, 24))  # Space after heading
                    else:
                        flowables.append(Paragraph(para, styles['BulletPoint']))

        # Build PDF
        doc.build(flowables)
        pdf_output.seek(0)

        # Mark progress as complete
        progress_bar.progress(100)
        status_text.success("PDF summarization complete!")

        return pdf_output

    except Exception as e:
        st.error(f"An error occurred during summarization: {e}")
        return None

def main():
    st.set_page_config(page_title="PDF Circular Summarizer", page_icon="📄")
    
    st.title("📄 PDF Circular Summarization Tool")
    st.markdown("""
    ### Summary Features
    - Extract text from multiple PDF circulars
    - Generate concise, PII-protected summaries
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
        summary_pdf = summarize_pdf_documents(uploaded_files, openai_api_key)
        
        if summary_pdf:
            st.download_button(
                label="Download Summary PDF",
                data=summary_pdf,
                file_name="circular_summary.pdf",
                mime="application/pdf"
            )

if __name__ == "__main__":
    main()
