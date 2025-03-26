import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from io import BytesIO
import re
import langdetect
from datetime import datetime

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

def summarize_circular_documents(uploaded_files, api_key):
    """Summarize PDF circulars and generate a consolidated PDF summary."""
    # Validate API key
    if not api_key.startswith("sk-"):
        st.error("Invalid API key format. OpenAI API keys should start with 'sk-'.")
        return None

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
        template=f"""{pii_instructions}Read and summarize the following content in your own words, highlighting the main ideas, purpose, and important insights without including direct phrases or sentences from the original text in 15 bullet points. Elaborate on the key highlights as per \n\n{{text}}
        """
    )

    combine_prompt = PromptTemplate(
        input_variables=["text"],
        template="""Each summary for a document should start with the document name (without extensions like .pdf or .docx).
        Each summary should have a heading named, "Key Pointers:"
        Combine the following individual summaries into a cohesive, insightful summary. Ensure that it is concise, capturing the core themes and purpose of the entire document in 15 bullet points:\n\n{text}
        """
    )

    # Prepare PDF output
    pdf_output = BytesIO()
    doc = SimpleDocTemplate(pdf_output, pagesize=A4, 
                            leftMargin=inch*0.5, 
                            rightMargin=inch*0.5, 
                            topMargin=inch*0.5, 
                            bottomMargin=inch*0.5)
    styles = getSampleStyleSheet()

    # Create custom styles
    styles.add(ParagraphStyle(
        name='MainTitle',
        parent=styles['Title'],
        fontSize=16,
        textColor='navy',
        alignment=1,  # Center alignment
        spaceAfter=12
    ))

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

    # Add Consolidated Overview Summary header
    flowables.append(Paragraph("Consolidated Overview Summary", styles['MainTitle']))
    flowables.append(Spacer(1, 12))

    # Process each uploaded PDF file
    for uploaded_file in uploaded_files:
        # Extract text from PDF
        pdf_reader = PdfReader(uploaded_file)
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

    # Build and save PDF
    doc.build(flowables)
    pdf_output.seek(0)

    return pdf_output.getvalue()

def main():
    st.set_page_config(page_title="PDF Circular Summarizer", page_icon="📄")
    
    st.title("🔍 PDF Circular Summarizer")
    
    # Sidebar for API Key input
    st.sidebar.header("Configuration")
    openai_api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload PDF Circulars", 
        type=['pdf'], 
        accept_multiple_files=True
    )
    
    # Summarization button
    if st.button("Summarize PDFs"):
        if not openai_api_key:
            st.error("Please enter your OpenAI API Key")
            return
        
        if not uploaded_files:
            st.error("Please upload at least one PDF file")
            return
        
        with st.spinner('Generating Summary...'):
            try:
                # Generate PDF summary
                summary_pdf = summarize_circular_documents(uploaded_files, openai_api_key)
                
                if summary_pdf:
                    # Create download button
                    st.download_button(
                        label="Download Summary PDF",
                        data=summary_pdf,
                        file_name=f"circulars_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf"
                    )
                    st.success("Summary PDF generated successfully!")
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

    # About section
    st.sidebar.markdown("---")
    st.sidebar.info("""
    ### How to Use
    1. Enter your OpenAI API Key
    2. Upload PDF circulars
    3. Click 'Summarize PDFs'
    4. Download the generated summary
    
    **Note:** Requires an OpenAI API key
    """)

if __name__ == "__main__":
    main()
