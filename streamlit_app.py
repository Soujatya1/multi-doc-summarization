import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
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

def standardize_summary_format(summary, doc_name):
    """
    Standardize summary to match the desired format structure.
    
    Args:
        summary (str): The generated summary
        doc_name (str): Document name for abbreviation
    
    Returns:
        str: Formatted summary matching the sample structure
    """
    # Create document abbreviation (first few letters of document name)
    doc_abbrev = ''.join([c.upper() for c in doc_name if c.isalpha()])[:3]
    
    # Clean and format the summary
    lines = summary.split('\n')
    formatted_lines = []
    
    # Add header
    formatted_lines.append(f"**[{doc_abbrev}]**")
    formatted_lines.append("")
    formatted_lines.append("[Summary:]")
    formatted_lines.append("")
    
    # Process the content
    in_content = False
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Skip header lines from LLM output
        if line.startswith("**[") or line.startswith("[Summary:]"):
            in_content = True
            continue
            
        if in_content and line:
            # Format bullet points
            if line.startswith('-'):
                formatted_lines.append(line)
            elif line.startswith('•') or line.startswith('*'):
                formatted_lines.append(f"-{line[1:]}")
            else:
                # Add as regular bullet point if not already formatted
                if not line.startswith('**[') and not line.startswith('[Summary'):
                    formatted_lines.append(f"- {line}")
    
    return '\n'.join(formatted_lines)

def create_summarization_chain(llm):
    """Create a single LLM chain for document summarization."""
    
    # Single comprehensive prompt matching the desired format
    summarization_prompt = PromptTemplate(
        input_variables=["text", "document_name"],
        template="""
        IMPORTANT: DO NOT include any personally identifiable information (PII) in your summary, including:
        - Bank account numbers, credit card numbers, social security numbers, passport numbers, personal mobile numbers
        If you encounter such information, DO NOT include it in your summary.
        
        Analyze the following circular document and create a comprehensive compliance-focused summary.
        
        Document Name: {document_name}
        Document Content: {text}
        
        Provide a detailed summary with the following EXACT structure and formatting:
        
        **[DOCUMENT_ABBREVIATION]**
        
        [Summary:]
        
        Create organized bullet points with clear sub-sections where applicable. Use the following structure:
        
        - Main regulatory requirements and compliance obligations
        - Key definitions and terminology changes
        - Board composition and governance requirements
            - Sub-points for specific board requirements
            - Director appointment and removal procedures
            - Independence requirements
        - Committee requirements and responsibilities
            - Mandatory committee functions
            - Risk management committee specifics
        - Related party transaction requirements
        - Succession planning requirements
        - Key Management Personnel (KMP) requirements
            - Chief Compliance Officer requirements
            - Duties and responsibilities
            - Reporting and notification requirements
        - Remuneration policies and requirements
        - Auditor requirements and qualifications
        - Disclosure and reporting requirements
            - Annual compliance reporting
            - Board composition disclosures
        - Any other specific regulatory frameworks (ESG, Stewardship, etc.)
        
        Format Guidelines:
        - Use clear main bullet points (-)
        - Use sub-bullet points for detailed requirements under main sections
        - Include specific timeframes, deadlines, and quantitative requirements
        - Highlight key regulatory changes or new requirements
        - Use underlined text formatting [text] for emphasis on important terms
        - Each point should be comprehensive yet concise
        - Focus on actionable compliance requirements
        - Include specific regulation references and authority requirements
        
        Ensure the summary covers all critical compliance aspects that require immediate attention or action.
        """
    )
    
    # Create single LLM chain
    summarization_chain = LLMChain(
        llm=llm,
        prompt=summarization_prompt,
        output_key="summary"
    )
    
    return summarization_chain

def summarize_circular_documents(uploaded_files, api_key):
    """Summarize PDF circulars using sequential chain and generate a consolidated PDF summary."""
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

    # Create summarization chain
    summarization_chain = create_summarization_chain(llm)

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
    flowables.append(Paragraph("Consolidated Document Summary", styles['MainTitle']))
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
            # For sequential chain, we'll process the entire document at once
            # If the document is too large, we can chunk it and process each chunk
            doc_name = os.path.splitext(uploaded_file.name)[0]
            
            # Check if text is too long for a single processing
            if len(filtered_text) > 15000:  # Adjust threshold as needed
                # Chunk the text and process each chunk
                text_chunks = chunk_document(filtered_text, chunk_size=12000, chunk_overlap=1000)
                
                chunk_summaries = []
                for i, chunk in enumerate(text_chunks):
                    try:
                        # Process each chunk with summarization chain
                        result = summarization_chain.invoke({
                            "text": chunk,
                            "document_name": f"{doc_name} (Part {i+1})"
                        })
                        chunk_summaries.append(result["summary"])
                    except Exception as e:
                        st.warning(f"Error processing chunk {i+1} of {doc_name}: {str(e)}")
                        continue
                
                # Combine chunk summaries
                if chunk_summaries:
                    # Create a final summary by combining all chunk summaries
                    combined_text = "\n\n".join(chunk_summaries)
                    
                    # Use the same chain to consolidate summaries
                    final_result = summarization_chain.invoke({
                        "text": f"Consolidate these summaries into one comprehensive summary:\n\n{combined_text}",
                        "document_name": doc_name
                    })
                    summary = final_result["summary"]
                else:
                    summary = f"1. Document Name: {doc_name}\n2. Key Pointers:\n- Unable to process document due to errors."
            else:
                # Process the entire document at once
                try:
                    result = summarization_chain.invoke({
                        "text": filtered_text,
                        "document_name": doc_name
                    })
                    summary = result["summary"]
                except Exception as e:
                    st.error(f"Error processing {doc_name}: {str(e)}")
                    summary = f"1. Document Name: {doc_name}\n2. Key Pointers:\n- Error processing document: {str(e)}"

            # Standardize the summary format
            summary = standardize_summary_format(summary, doc_name)

            # Format summary for PDF - parse the new format
            summary_lines = summary.split('\n')
            
            # Find the document abbreviation and summary content
            for i, line in enumerate(summary_lines):
                line = line.strip()
                
                if line.startswith('**[') and line.endswith(']**'):
                    # Add document abbreviation as title
                    doc_abbrev = line.replace('**[', '').replace(']**', '')
                    flowables.append(Paragraph(f"Document: {doc_name} ({doc_abbrev})", styles['DocumentName']))
                    flowables.append(Spacer(1, 6))
                    continue
                
                if line == "[Summary:]":
                    flowables.append(Paragraph("Summary:", styles['Heading3']))
                    continue
                
                if line.startswith('- ') and line.strip():
                    # Main bullet point
                    bullet_text = line[2:].strip()
                    flowables.append(Paragraph(f"• {bullet_text}", styles['BulletPoint']))
                elif line.startswith('    - ') or line.startswith('        - '):
                    # Sub bullet point (indented)
                    sub_bullet_text = line.strip()[2:].strip()
                    flowables.append(Paragraph(f"    ◦ {sub_bullet_text}", styles['BulletPoint']))
                elif line.strip() and not line.startswith('**[') and not line.startswith('[Summary'):
                    # Regular text
                    if line.strip():
                        flowables.append(Paragraph(line.strip(), styles['BodyText']))
            
            flowables.append(Spacer(1, 12))  # Add space between document summaries

    # Build and save PDF
    doc.build(flowables)
    pdf_output.seek(0)

    return pdf_output.getvalue()

def main():
    st.set_page_config(page_title="PDF Circular Summarizer", page_icon="📄")
    
    st.title("🔍 PDF Circular Summarizer (Single Chain)")
    
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
    
    **Features:**
    - Single comprehensive prompt
    - Detailed compliance-focused summaries
    - Structured bullet point format
    - Handles large documents with chunking
    
    **Note:** Requires an OpenAI API key
    """)

if __name__ == "__main__":
    main()
