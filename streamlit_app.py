import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain, SequentialChain
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

def create_sequential_chain(llm):
    """Create a sequential chain for document summarization."""
    
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
    
    # Step 1: Extract key information
    extraction_prompt = PromptTemplate(
        input_variables=["text"],
        template=f"""{pii_instructions}
        
        Step 1: Extract key information from this circular document.
        Focus on identifying:
        - Regulatory requirements and compliance obligations
        - New rules, amendments, or modifications
        - Implementation dates and deadlines
        - Penalties and consequences
        - Required procedures and processes
        - Reporting requirements
        - Exemptions and special conditions
        
        Extract all relevant details with specific regulation numbers, dates, and requirements.
        
        Document text:
        {{text}}
        
        Extracted information:
        """
    )
    
    # Step 2: Structure the summary
    structuring_prompt = PromptTemplate(
        input_variables=["extracted_info", "document_name"],
        template="""Step 2: Structure the extracted information into a comprehensive summary.
        
        Create a summary with the following EXACT structure:
        1. Document Name: {document_name}
        2. Key Pointers:
        
        Transform the extracted information into detailed bullet points that include:
        - Specific regulation name, number, or reference code
        - Nature of the regulatory change
        - Detailed compliance requirements
        - Implementation timeline with specific dates
        - Required actions and procedures
        - Documentation and reporting requirements
        - Penalties for non-compliance
        - Impact on existing policies
        - Applicable entities affected
        - Exemptions or special conditions
        
        Each bullet point MUST:
        * Start with a capitalized first letter
        * End with a period
        * Include specific regulatory details and dates
        * Be comprehensive yet concise
        
        Extracted information to structure:
        {extracted_info}
        
        Structured summary:
        """
    )
    
    # Step 3: Refine and finalize
    refinement_prompt = PromptTemplate(
        input_variables=["structured_summary"],
        template="""Step 3: Refine and finalize the summary for maximum clarity and compliance focus.
        
        Review the structured summary and ensure:
        - All critical compliance requirements are highlighted
        - Dates and deadlines are clearly stated
        - Regulatory references are accurate
        - Bullet points are well-formatted and actionable
        - The most important information is prioritized
        - Language is clear and professional
        
        Structured summary to refine:
        {structured_summary}
        
        Final refined summary:
        """
    )
    
    # Create individual chains
    extraction_chain = LLMChain(
        llm=llm,
        prompt=extraction_prompt,
        output_key="extracted_info"
    )
    
    structuring_chain = LLMChain(
        llm=llm,
        prompt=structuring_prompt,
        output_key="structured_summary"
    )
    
    refinement_chain = LLMChain(
        llm=llm,
        prompt=refinement_prompt,
        output_key="final_summary"
    )
    
    # Create sequential chain
    sequential_chain = SequentialChain(
        chains=[extraction_chain, structuring_chain, refinement_chain],
        input_variables=["text", "document_name"],
        output_variables=["final_summary"],
        verbose=True
    )
    
    return sequential_chain

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

    # Create sequential chain
    sequential_chain = create_sequential_chain(llm)

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
                        # Process each chunk with sequential chain
                        result = sequential_chain.invoke({
                            "text": chunk,
                            "document_name": f"{doc_name} (Part {i+1})"
                        })
                        chunk_summaries.append(result["final_summary"])
                    except Exception as e:
                        st.warning(f"Error processing chunk {i+1} of {doc_name}: {str(e)}")
                        continue
                
                # Combine chunk summaries
                if chunk_summaries:
                    # Create a final summary by combining all chunk summaries
                    combined_text = "\n\n".join(chunk_summaries)
                    
                    # Use a simple combination approach
                    final_result = sequential_chain.invoke({
                        "text": f"Combine and consolidate the following summaries into one comprehensive summary:\n\n{combined_text}",
                        "document_name": doc_name
                    })
                    summary = final_result["final_summary"]
                else:
                    summary = f"1. Document Name: {doc_name}\n2. Key Pointers:\n- Unable to process document due to errors."
            else:
                # Process the entire document at once
                try:
                    result = sequential_chain.invoke({
                        "text": filtered_text,
                        "document_name": doc_name
                    })
                    summary = result["final_summary"]
                except Exception as e:
                    st.error(f"Error processing {doc_name}: {str(e)}")
                    summary = f"1. Document Name: {doc_name}\n2. Key Pointers:\n- Error processing document: {str(e)}"

            # Standardize the summary
            summary = standardize_key_pointers(summary)

            # Format summary for PDF
            # Add document name
            flowables.append(Paragraph(f"Document: {doc_name}", styles['DocumentName']))
            flowables.append(Spacer(1, 6))

            # Add "Key Pointers" section
            flowables.append(Paragraph("Key Pointers:", styles['Heading3']))
            
            # Split summary into bullet points
            if '2. Key Pointers:' in summary:
                key_pointers_section = summary.split('2. Key Pointers:')[1].strip()
                bullet_points = key_pointers_section.split('\n')
                
                for point in bullet_points:
                    if point.strip():
                        flowables.append(Paragraph(point.strip(), styles['BulletPoint']))
            else:
                # Fallback if summary format is different
                lines = summary.split('\n')
                for line in lines:
                    if line.strip() and not line.startswith('1. Document Name:'):
                        flowables.append(Paragraph(line.strip(), styles['BulletPoint']))
            
            flowables.append(Spacer(1, 12))  # Add space between document summaries

    # Build and save PDF
    doc.build(flowables)
    pdf_output.seek(0)

    return pdf_output.getvalue()

def main():
    st.set_page_config(page_title="PDF Circular Summarizer", page_icon="📄")
    
    st.title("🔍 PDF Circular Summarizer (Sequential Chain)")
    
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
        
        with st.spinner('Generating Summary using Sequential Chain...'):
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
                    st.success("Summary PDF generated successfully using Sequential Chain!")
            
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
    - Uses Sequential Chain for step-by-step processing
    - Extracts → Structures → Refines summaries
    - Handles large documents with chunking
    
    **Note:** Requires an OpenAI API key
    """)

if __name__ == "__main__":
    main()
