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

def parse_summary_content(summary):
    """
    Parse the LLM summary to extract structured content including bullet points.
    
    Args:
        summary (str): The generated summary from LLM
    
    Returns:
        dict: Parsed content with sections and bullet points
    """
    content = {
        'document_identity': '',
        'title': '',
        'issuing_authority': '',
        'notification_date': '',
        'effective_date': '',
        'applicability': '',
        'objective': '',
        'chapter_summaries': [],
        'regulatory_mandates': [],
        'compliance_requirements': [],
        'transitional_provisions': [],
        'other_content': []
    }
    
    # Split by common section headers
    sections = re.split(r'\n(?=🔹|Document Identity|Title|Issuing Authority|Notification Date|Effective Date|Applicability|Objective|Chapter-wise Summary|Critical Regulatory Mandates|Compliance and Disclosure Requirements|Transitional Provisions)', summary)
    
    current_section = 'other_content'
    
    for section in sections:
        section = section.strip()
        if not section:
            continue
            
        # Identify section type
        if section.startswith('Document Identity'):
            current_section = 'document_identity'
            content[current_section] = section
        elif section.startswith('Title'):
            current_section = 'title'
            content[current_section] = section
        elif section.startswith('Issuing Authority'):
            current_section = 'issuing_authority'
            content[current_section] = section
        elif section.startswith('Notification Date'):
            current_section = 'notification_date'
            content[current_section] = section
        elif section.startswith('Effective Date'):
            current_section = 'effective_date'
            content[current_section] = section
        elif section.startswith('Applicability'):
            current_section = 'applicability'
            content[current_section] = section
        elif section.startswith('Objective'):
            current_section = 'objective'
            content[current_section] = section
        elif 'Chapter-wise Summary' in section or 'Chapter' in section:
            current_section = 'chapter_summaries'
            content[current_section].append(section)
        elif 'Critical Regulatory Mandates' in section or 'Regulatory Mandates' in section:
            current_section = 'regulatory_mandates'
            content[current_section].append(section)
        elif 'Compliance and Disclosure' in section or 'Compliance Requirements' in section:
            current_section = 'compliance_requirements'
            content[current_section].append(section)
        elif 'Transitional Provisions' in section or 'Transitional' in section:
            current_section = 'transitional_provisions'
            content[current_section].append(section)
        else:
            content['other_content'].append(section)
    
    return content

def extract_bullet_points(text):
    """
    Extract bullet points from text, handling various formats.
    
    Args:
        text (str): Text containing bullet points
    
    Returns:
        list: List of cleaned bullet points
    """
    bullet_points = []
    
    # Split by lines
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if line is a bullet point (starts with -, •, *, number, etc.)
        if re.match(r'^[-•*▪▫◦‣⁃]\s*', line) or re.match(r'^\d+[.)]\s*', line):
            # Remove bullet markers and clean
            clean_point = re.sub(r'^[-•*▪▫◦‣⁃\d.)\s]+', '', line).strip()
            if clean_point:
                bullet_points.append(clean_point)
        elif line and not line.startswith(('Chapter', 'Section', 'Part', '🔹', 'Document', 'Title', 'Issuing', 'Notification', 'Effective', 'Applicability', 'Objective', 'Critical', 'Compliance', 'Transitional')):
            # If it's not a header but has content, treat as potential bullet point
            bullet_points.append(line)
    
    return bullet_points

def standardize_key_pointers(summary):
    """
    Standardize key pointers to ensure consistent formatting.
    
    Args:
        summary (str): The generated summary
    
    Returns:
        str: Standardized summary with consistent key pointers
    """
    # Split the summary into sections
    sections = summary.split('2. Key Pointers:')
    
    if len(sections) > 1:
        # Extract the key pointers
        pointers = sections[1].strip().split('\n')
        
        # Clean and standardize pointers
        cleaned_pointers = []
        for pointer in pointers:
            # Remove any bullet points or numbering
            clean_pointer = re.sub(r'^[-•*\d.)\s]+', '', pointer).strip()
            
            # Capitalize first letter, ensure it ends with a period
            if clean_pointer:
                clean_pointer = clean_pointer[0].upper() + clean_pointer[1:]
                if not clean_pointer.endswith('.'):
                    clean_pointer += '.'
                
                cleaned_pointers.append(clean_pointer)
        
        # Reconstruct the summary with standardized pointers
        standardized_summary = f"{sections[0].strip()}\n\n2. Key Pointers:\n"
        standardized_summary += '\n'.join([f"- {point}" for point in cleaned_pointers])
        
        return standardized_summary
    
    return summary

def create_document_summary_chain(llm):
    
    pii_instructions = """
    CRITICAL: DO NOT include any personally identifiable information (PII) in your generated content, including:
    - Bank account numbers, credit card numbers, social security numbers
    - Personal names, addresses, phone numbers, email addresses
    - Any other personal identifiers
    Replace with generic terms like [REDACTED] if encountered.
    """

    # Regulatory-focused prompt template
    prompt_template = f"""{pii_instructions}

    You are a senior regulatory analyst specialized in compliance documentation. Read and extract all relevant details from the uploaded regulatory document. Your task is to produce a highly accurate, structured, and complete summary of the document as per the following format:

🔹 Output Format:
Document Identity

Title

Issuing Authority

Notification Date & Gazette Reference

Effective Date

Applicability (including inclusions and exclusions)

Objective / Purpose

Clearly list the purpose(s) stated in the regulation.

Chapter-wise Summary

For each Chapter (or Part), use the format:

Chapter Name (or Section title if no chaptering)

Provide a bullet-point list of rules, requirements, roles, and key conditions.

Mention mandatory thresholds, timelines, exemptions, or penalties.

Include committee types, responsibilities, and special cases.

Critical Regulatory Mandates

Highlight key mandates such as:

- Board composition and committee requirements.
- Appointment norms for KMPs and Statutory Auditors.
- ESG framework, Risk Management, Compliance Reporting.
- Capital requirements and succession planning.

Compliance and Disclosure Requirements

Mention roles like Chief Compliance Officer (CCO) and reporting frequency.

Note what must be filed with the Authority and when.

Transitional Provisions / Clarifications

Detail any transitional guidelines, future circulars, or powers reserved by the Authority.

🔹 Key Instructions:
Use official terminology (e.g., Independent Directors, Board of Insurers, KMPs).

Do not omit sub-points even if they seem repetitive.

Ensure completeness: if a chapter lists multiple roles, functions, or exceptions—capture all.

Avoid interpretation: Just extract and rephrase clearly without editorializing.

Format all lists as proper bullet points using "-" or "•" symbols.

    Document Content:
    {{context}}

    Summarize the following PDF using the prompt format above"""

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
            summary = f"1. Document Name: {document_name}\n2. Key Pointers:\n- Unable to process document due to errors."
    else:
        # Process all chunks at once
        try:
            summary = chain.invoke({"context": docs})
        except Exception as e:
            st.error(f"Error processing document {document_name}: {str(e)}")
            summary = f"1. Document Name: {document_name}\n2. Key Pointers:\n- Unable to process document due to error: {str(e)}"
    
    return summary

def add_content_to_pdf(flowables, content, styles):
    """
    Add parsed content to PDF flowables with proper formatting.
    
    Args:
        flowables (list): List of PDF flowables
        content (dict): Parsed content dictionary
        styles: ReportLab styles
    """
    
    # Add each section with proper formatting
    sections = [
        ('document_identity', 'Document Identity'),
        ('title', 'Title'),
        ('issuing_authority', 'Issuing Authority'),
        ('notification_date', 'Notification Date'),
        ('effective_date', 'Effective Date'),
        ('applicability', 'Applicability'),
        ('objective', 'Objective / Purpose')
    ]
    
    for key, header in sections:
        if content[key]:
            flowables.append(Paragraph(header, styles['Heading3']))
            
            # Extract bullet points if present
            bullet_points = extract_bullet_points(content[key])
            if bullet_points:
                for point in bullet_points:
                    flowables.append(Paragraph(f"• {point}", styles['BulletPoint']))
            else:
                # Add as regular text if no bullet points
                clean_text = re.sub(r'^' + re.escape(header) + r'\s*', '', content[key]).strip()
                if clean_text:
                    flowables.append(Paragraph(clean_text, styles['BodyText']))
            
            flowables.append(Spacer(1, 6))
    
    # Add chapter summaries
    if content['chapter_summaries']:
        flowables.append(Paragraph("Chapter-wise Summary", styles['Heading3']))
        for chapter in content['chapter_summaries']:
            bullet_points = extract_bullet_points(chapter)
            if bullet_points:
                for point in bullet_points:
                    flowables.append(Paragraph(f"• {point}", styles['BulletPoint']))
            else:
                flowables.append(Paragraph(chapter, styles['BodyText']))
        flowables.append(Spacer(1, 6))
    
    # Add regulatory mandates
    if content['regulatory_mandates']:
        flowables.append(Paragraph("Critical Regulatory Mandates", styles['Heading3']))
        for mandate in content['regulatory_mandates']:
            bullet_points = extract_bullet_points(mandate)
            if bullet_points:
                for point in bullet_points:
                    flowables.append(Paragraph(f"• {point}", styles['BulletPoint']))
            else:
                flowables.append(Paragraph(mandate, styles['BodyText']))
        flowables.append(Spacer(1, 6))
    
    # Add compliance requirements
    if content['compliance_requirements']:
        flowables.append(Paragraph("Compliance and Disclosure Requirements", styles['Heading3']))
        for req in content['compliance_requirements']:
            bullet_points = extract_bullet_points(req)
            if bullet_points:
                for point in bullet_points:
                    flowables.append(Paragraph(f"• {point}", styles['BulletPoint']))
            else:
                flowables.append(Paragraph(req, styles['BodyText']))
        flowables.append(Spacer(1, 6))
    
    # Add transitional provisions
    if content['transitional_provisions']:
        flowables.append(Paragraph("Transitional Provisions / Clarifications", styles['Heading3']))
        for provision in content['transitional_provisions']:
            bullet_points = extract_bullet_points(provision)
            if bullet_points:
                for point in bullet_points:
                    flowables.append(Paragraph(f"• {point}", styles['BulletPoint']))
            else:
                flowables.append(Paragraph(provision, styles['BodyText']))
        flowables.append(Spacer(1, 6))
    
    # Add other content
    if content['other_content']:
        for other in content['other_content']:
            bullet_points = extract_bullet_points(other)
            if bullet_points:
                for point in bullet_points:
                    flowables.append(Paragraph(f"• {point}", styles['BulletPoint']))
            else:
                flowables.append(Paragraph(other, styles['BodyText']))
        flowables.append(Spacer(1, 6))

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

    styles.add(ParagraphStyle(
        name='DocumentName',
        parent=styles['Heading2'],
        textColor='darkblue',
        fontSize=14,
        spaceAfter=6
    ))

    # Improved bullet point style
    styles.add(ParagraphStyle(
        name='BulletPoint',
        parent=styles['BodyText'],
        firstLineIndent=0,
        leftIndent=20,
        spaceBefore=3,
        spaceAfter=3,
        bulletIndent=0,
        fontSize=10
    ))

    flowables = []

    # Add Consolidated Overview Summary header
    flowables.append(Paragraph("Consolidated Document Summary", styles['MainTitle']))
    flowables.append(Spacer(1, 12))

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
                
                # Parse the summary content
                parsed_content = parse_summary_content(summary)

                # Add document name
                flowables.append(Paragraph(f"Document: {doc_name}", styles['DocumentName']))
                flowables.append(Spacer(1, 6))

                # Add parsed content to PDF
                add_content_to_pdf(flowables, parsed_content, styles)
                
                flowables.append(Spacer(1, 12))  # Add space between document summaries
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
    st.markdown("*Powered by LangChain's create_stuff_documents_chain*")
    
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
        help="Upload one or more PDF files to generate a consolidated summary"
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
        
        with st.spinner('Generating Summary...'):
            try:
                # Generate PDF summary with custom chunk settings
                summary_pdf = summarize_circular_documents(uploaded_files, openai_api_key)
                
                if summary_pdf:
                    # Create download button
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"circulars_summary_{timestamp}.pdf"
                    
                    st.download_button(
                        label="📥 Download Summary PDF",
                        data=summary_pdf,
                        file_name=filename,
                        mime="application/pdf"
                    )
                    st.success("✅ Summary PDF generated successfully!")
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
    - Modern LangChain implementation
    - Batch processing for large documents
    - PII protection
    - Customizable chunking
    - Progress tracking
    - Improved bullet point formatting
    
    **Note:** Requires an OpenAI API key
    """)
    
    # Footer
    st.markdown("---")
    st.markdown("*Built with Streamlit and LangChain*")

if __name__ == "__main__":
    main()
