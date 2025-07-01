import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
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

def chunk_document(text, chunk_size=4000, chunk_overlap=200):
    """Split the document into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    return text_splitter.split_text(text)

def parse_dynamic_summary(summary):
    """
    Parse the LLM summary to dynamically extract sections and their bullet points.
    
    Args:
        summary (str): The generated summary from LLM
    
    Returns:
        dict: Parsed content with dynamic sections and bullet points
    """
    content = {
        'document_info': {},
        'sections': []
    }
    
    # Split by lines to process
    lines = summary.split('\n')
    current_section = None
    current_section_content = []
    
    # Common document info headers to capture
    doc_info_headers = [
        'document identity', 'title', 'issuing authority', 'notification date', 
        'effective date', 'applicability', 'objective', 'purpose'
    ]
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Remove emoji and special characters from the beginning
        clean_line = re.sub(r'^[🔹📄📋📊🔍✅❗️⚠️]*\s*', '', line)
        
        # Check if this is a section header (not a bullet point)
        is_bullet = re.match(r'^[-•*▪▫◦‣⁃]\s*', line) or re.match(r'^\d+[.)]\s*', line)
        
        # Check if this looks like a header (starts with capital, ends with colon, etc.)
        is_header = (
            not is_bullet and 
            (line.endswith(':') or 
             re.match(r'^[A-Z][^.]*[A-Za-z]$', clean_line) or
             any(keyword in clean_line.lower() for keyword in ['chapter', 'section', 'part', 'article', 'rule', 'regulation', 'provision', 'requirement', 'mandate', 'compliance', 'disclosure']))
        )
        
        # Check if it's document info
        is_doc_info = any(header in clean_line.lower() for header in doc_info_headers)
        
        if is_doc_info and not is_bullet:
            # Store document information
            key = clean_line.lower().replace(':', '').strip()
            content['document_info'][key] = clean_line
        elif is_header and not is_bullet:
            # Save previous section if exists
            if current_section:
                content['sections'].append({
                    'header': current_section,
                    'content': current_section_content.copy()
                })
            
            # Start new section
            current_section = clean_line.rstrip(':')
            current_section_content = []
        else:
            # Add content to current section
            if current_section:
                if is_bullet:
                    # Clean bullet point
                    bullet_text = re.sub(r'^[-•*▪▫◦‣⁃\d.)\s]+', '', line).strip()
                    if bullet_text:
                        current_section_content.append(bullet_text)
                else:
                    # Regular content that might be part of bullet points or descriptions
                    current_section_content.append(line)
    
    # Don't forget the last section
    if current_section and current_section_content:
        content['sections'].append({
            'header': current_section,
            'content': current_section_content.copy()
        })
    
    return content

def create_document_summary_chain(llm):
    
    pii_instructions = """
    CRITICAL: DO NOT include any personally identifiable information (PII) in your generated content, including:
    - Bank account numbers, credit card numbers, social security numbers
    - Personal names, addresses, phone numbers, email addresses
    - Any other personal identifiers
    Replace with generic terms like [REDACTED] if encountered.
    """

    # Dynamic regulatory-focused prompt template
    prompt_template = f"""{pii_instructions}

You are a senior regulatory analyst specialized in compliance documentation. Analyze the uploaded regulatory document and create a comprehensive, structured summary. 

IMPORTANT FORMATTING INSTRUCTIONS:
1. Identify ALL actual chapters, sections, parts, or major headings present in the document
2. Use the EXACT chapter/section names from the original document as headers
3. Under each header, provide bullet points using "-" symbol
4. Ensure each bullet point is on a separate line
5. Do not create generic sections - use only what exists in the document

OUTPUT STRUCTURE:

Document Identity:
[Extract document title, issuing authority, dates, etc.]

Title:
[Official document title]

Issuing Authority:
[Who issued this document]

Notification Date:
[When was it notified]

Effective Date:
[When does it become effective]

Applicability:
[Who/what does this apply to]

Objective:
[Main purpose of the regulation]

[ACTUAL CHAPTER/SECTION NAME FROM DOCUMENT]:
- [Key point 1]
- [Key point 2]
- [Key point 3]

[NEXT ACTUAL CHAPTER/SECTION NAME FROM DOCUMENT]:
- [Key point 1]
- [Key point 2]
- [Key point 3]

[Continue with all actual chapters/sections found in the document]

Critical Regulatory Mandates:
- [Key mandate 1]
- [Key mandate 2]
- [Key mandate 3]

Compliance and Disclosure Requirements:
- [Requirement 1]
- [Requirement 2]
- [Requirement 3]

Transitional Provisions:
- [Provision 1]
- [Provision 2]

CRITICAL RULES:
- Use ONLY the actual chapter/section headings found in the source document
- Do not create generic chapter names
- Ensure every bullet point starts with "-" and is on its own line
- Extract ALL significant rules, requirements, roles, and conditions
- Include specific thresholds, timelines, exemptions, penalties
- Mention committee types, responsibilities, and special cases

Document Content:
{{context}}

Analyze and summarize the document following the exact structure above, using the actual chapter/section names from the source document."""

    prompt = PromptTemplate(
        input_variables=["context"],
        template=prompt_template
    )
    
    # Create the stuff documents chain
    chain = create_stuff_documents_chain(llm, prompt)
    
    return chain

def summarize_single_document(text, document_name, llm):
    """Summarize a single document using the stuff documents chain."""
    
    # Create the summarization chain
    chain = create_document_summary_chain(llm)
    
    # Chunk the document
    text_chunks = chunk_document(text)
    
    # Create Document objects
    docs = [Document(page_content=chunk, metadata={"source": document_name}) for chunk in text_chunks]
    
    # If document is too large, process in batches and combine
    if len(docs) > 10:  # If more than 10 chunks, process in batches
        batch_summaries = []
        batch_size = 8
        
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i + batch_size]
            try:
                batch_result = chain.invoke({"context": batch})
                batch_summaries.append(batch_result)
            except Exception as e:
                st.warning(f"Error processing batch {i//batch_size + 1}: {str(e)}")
                continue
        
        # Combine batch summaries
        if batch_summaries:
            combined_text = "\n\n".join(batch_summaries)
            # Create a final summary from the batch summaries
            final_docs = [Document(page_content=combined_text, metadata={"source": document_name})]
            summary = chain.invoke({"context": final_docs})
        else:
            summary = f"Document Name: {document_name}\nError: Unable to process document due to errors."
    else:
        # Process all chunks at once
        try:
            summary = chain.invoke({"context": docs})
        except Exception as e:
            st.error(f"Error processing document {document_name}: {str(e)}")
            summary = f"Document Name: {document_name}\nError: Unable to process document due to error: {str(e)}"
    
    return summary

def add_parsed_content_to_pdf(flowables, parsed_content, styles, doc_name):
    """
    Add parsed content to PDF flowables with proper dynamic formatting.
    
    Args:
        flowables (list): List of PDF flowables
        parsed_content (dict): Parsed content dictionary
        styles: ReportLab styles
        doc_name (str): Document name
    """
    
    # Add document name
    flowables.append(Paragraph(f"Document: {doc_name}", styles['DocumentName']))
    flowables.append(Spacer(1, 8))
    
    # Add document information section
    if parsed_content['document_info']:
        flowables.append(Paragraph("Document Information", styles['Heading3']))
        for key, value in parsed_content['document_info'].items():
            if value:
                flowables.append(Paragraph(value, styles['BodyText']))
        flowables.append(Spacer(1, 8))
    
    # Add all dynamic sections
    for section in parsed_content['sections']:
        header = section['header']
        content_items = section['content']
        
        # Add section header
        flowables.append(Paragraph(header, styles['Heading3']))
        flowables.append(Spacer(1, 4))
        
        # Add bullet points for this section
        if content_items:
            for item in content_items:
                if item.strip():
                    # Clean the item (remove any existing bullets)
                    clean_item = re.sub(r'^[-•*▪▫◦‣⁃\d.)\s]*', '', item).strip()
                    if clean_item:
                        flowables.append(Paragraph(f"• {clean_item}", styles['BulletPoint']))
        
        flowables.append(Spacer(1, 8))
    
    # Add a separator between documents
    flowables.append(Spacer(1, 12))

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
        temperature=0.1,  # Lower temperature for more consistent formatting
        top_p=0.1
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
        fontSize=18,
        textColor='navy',
        alignment=1,  # Center alignment
        spaceAfter=16,
        fontName='Helvetica-Bold'
    ))

    styles.add(ParagraphStyle(
        name='DocumentName',
        parent=styles['Heading1'],
        textColor='darkblue',
        fontSize=14,
        spaceAfter=8,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    ))

    # Enhanced bullet point style
    styles.add(ParagraphStyle(
        name='BulletPoint',
        parent=styles['BodyText'],
        firstLineIndent=0,
        leftIndent=24,
        spaceBefore=2,
        spaceAfter=2,
        bulletIndent=12,
        fontSize=10,
        fontName='Helvetica'
    ))

    # Enhanced heading style for dynamic sections
    styles.add(ParagraphStyle(
        name='SectionHeader',
        parent=styles['Heading3'],
        textColor='darkgreen',
        fontSize=12,
        spaceAfter=4,
        spaceBefore=8,
        fontName='Helvetica-Bold'
    ))

    flowables = []

    # Add main title
    flowables.append(Paragraph("Consolidated Regulatory Document Summary", styles['MainTitle']))
    flowables.append(Spacer(1, 16))

    # Process each uploaded PDF file
    progress_bar = st.progress(0)
    total_files = len(uploaded_files)
    
    for idx, uploaded_file in enumerate(uploaded_files):
        # Update progress
        progress_bar.progress((idx + 1) / total_files)
        st.text(f"Processing: {uploaded_file.name}")
        
        try:
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
                # Extract document name from uploaded file
                doc_name = os.path.splitext(uploaded_file.name)[0]
                
                # Summarize the document
                summary = summarize_single_document(filtered_text, doc_name, llm)
                
                # Parse the summary content dynamically
                parsed_content = parse_dynamic_summary(summary)

                # Add parsed content to PDF with dynamic sections
                add_parsed_content_to_pdf(flowables, parsed_content, styles, doc_name)
                
            else:
                st.warning(f"No readable text found in {uploaded_file.name}")
                
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            continue

    # Build and save PDF
    try:
        doc.build(flowables)
        pdf_output.seek(0)
        return pdf_output.getvalue()
    except Exception as e:
        st.error(f"Error generating final PDF: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="PDF Circular Summarizer", page_icon="📄")
    
    st.title("🔍 PDF Circular Summarizer")
    st.markdown("*Powered by LangChain with Dynamic Section Detection*")
    
    # Sidebar for API Key input
    st.sidebar.header("Configuration")
    openai_api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")
    
    # Advanced settings in sidebar
    st.sidebar.header("Advanced Settings")
    chunk_size = st.sidebar.slider("Chunk Size", min_value=1000, max_value=8000, value=4000, step=500)
    chunk_overlap = st.sidebar.slider("Chunk Overlap", min_value=50, max_value=500, value=200, step=50)
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload PDF Circulars", 
        type=['pdf'], 
        accept_multiple_files=True,
        help="Upload one or more PDF files to generate a consolidated summary with dynamic section detection"
    )
    
    # Display uploaded files
    if uploaded_files:
        st.write(f"**{len(uploaded_files)} file(s) uploaded:**")
        for file in uploaded_files:
            st.write(f"- {file.name} ({file.size} bytes)")
    
    # Summarization button
    if st.button("Summarize PDFs", type="primary"):
        if not openai_api_key:
            st.error("Please enter your OpenAI API Key")
            return
        
        if not uploaded_files:
            st.error("Please upload at least one PDF file")
            return
        
        with st.spinner('Generating Summary with Dynamic Section Detection...'):
            try:
                # Generate PDF summary with dynamic section detection
                summary_pdf = summarize_circular_documents(uploaded_files, openai_api_key)
                
                if summary_pdf:
                    # Create download button
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"dynamic_circulars_summary_{timestamp}.pdf"
                    
                    st.download_button(
                        label="📥 Download Summary PDF",
                        data=summary_pdf,
                        file_name=filename,
                        mime="application/pdf"
                    )
                    st.success("✅ Summary PDF generated successfully with dynamic sections!")
                    st.balloons()
                else:
                    st.error("Failed to generate summary PDF")
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.write("Please check your API key and try again.")

    # About section
    st.sidebar.markdown("---")
    st.sidebar.info("""
    ### How to Use
    1. Enter your OpenAI API Key
    2. Upload PDF circulars (multiple files supported)
    3. Adjust advanced settings if needed
    4. Click 'Summarize PDFs'
    5. Download the generated summary
    
    ### Features
    - **Dynamic Section Detection**: Automatically identifies actual chapters/sections from each document
    - Modern LangChain implementation
    - Batch processing for large documents
    - PII protection
    - Customizable chunking
    - Progress tracking
    - Smart bullet point formatting
    
    **Note:** Requires an OpenAI API key
    """)
    
    # Footer
    st.markdown("---")
    st.markdown("*Built with Streamlit and LangChain - Now with Dynamic Section Detection*")

if __name__ == "__main__":
    main()
