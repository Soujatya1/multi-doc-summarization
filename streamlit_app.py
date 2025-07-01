import streamlit as st
import tempfile
import os
import re
from io import BytesIO
from datetime import datetime

# Language detection
import langdetect
from langdetect.lang_detect_exception import LangDetectException

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
            except LangDetectException:
                continue
        
        return ' '.join(english_words)
    
    except Exception as e:
        st.error(f"Language error: {e}")
        return text

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
        
        # Extract English text from each document
        for doc in documents:
            doc.page_content = extract_english_text(doc.page_content)
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return documents
    
    except Exception as e:
        st.error(f"Error loading document: {str(e)}")
        return None

def split_documents(documents, chunk_size=2000, chunk_overlap=400):
    """Split documents into smaller chunks with larger size for better context"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return text_splitter.split_documents(documents)

def create_detailed_summary_chain(api_key, model_name="gpt-4o"):
    """Create the detailed summarization chain"""
    llm = ChatOpenAI(
        api_key=api_key,
        model_name=model_name,
        temperature=0.1
    )
    
    # Enhanced prompt template for detailed structured summary
    prompt_template = ChatPromptTemplate.from_template("""
    You are an expert regulatory document analyst specializing in insurance and corporate governance regulations. Create a comprehensive, detailed summary that captures EVERY important detail from the document.

    CRITICAL REQUIREMENTS:
    1. **PRESERVE ALL SPECIFIC DETAILS**: Include exact timeframes (days, months, years), monetary amounts, percentages, committee compositions, tenure limits, and procedural requirements
    2. **EXTRACT TECHNICAL DEFINITIONS**: Capture all definitions of terms like KMPs, Independent Directors, Competent Authority, etc.
    3. **DETAILED COMPLIANCE REQUIREMENTS**: Include all regulatory obligations, approval processes, reporting timelines, and penalty/consequence information
    4. **COMMITTEE STRUCTURES**: Detail all committee compositions, roles, responsibilities, quorum requirements, and meeting frequencies
    5. **PROCEDURAL STEPS**: Capture step-by-step processes, approval workflows, and notification requirements

    FORMATTING REQUIREMENTS:
    1. Use clear hierarchical structure with main sections and sub-sections based on the document's actual organization
    2. Create section headers that match or reflect the document's original structure (e.g., "Board:", "Committees:", "CEO and KMPs:")
    3. Under each section, provide comprehensive bullet points that capture ALL key information without summarizing away important details
    4. Use bold formatting for important terms, requirements, and section headers
    5. Include ALL specific numbers, timeframes, monetary limits, and quantitative data EXACTLY as stated
    6. Preserve technical terminology and regulatory language
    7. Use sub-bullets to show detailed breakdowns and related requirements

    CONTENT EXTRACTION PRIORITIES:
    - **Definitions**: Extract ALL defined terms with their complete definitions
    - **Timeframes**: Include ALL deadlines, tenure limits, notice periods, and timing requirements
    - **Monetary Amounts**: Capture all financial limits, thresholds, and remuneration details
    - **Approval Processes**: Detail who approves what, when approvals are needed, and approval timelines
    - **Committee Details**: Include composition requirements, roles, responsibilities, and operational details
    - **Compliance Obligations**: Extract all reporting requirements, certification needs, and regulatory obligations
    - **Procedural Requirements**: Include step-by-step processes and operational procedures
    - **Exceptions and Special Cases**: Note any exemptions, special circumstances, or conditional requirements

    EXAMPLE OUTPUT STRUCTURE (adapt to actual document content):
    **[Main Section as it appears in document]**
    
    [Subsection Title from document]:
    - Specific requirement with exact details (include numbers, timeframes, amounts)
        - Sub-requirement with precise specifications
        - Additional procedural details with exact compliance timelines
    - Another requirement capturing technical definitions or processes
        - Detailed breakdown of components or steps
        - Specific conditions or exceptions

    Document content:
    {context}
    
    Comprehensive Detailed Summary (capture ALL important details):
    """)
    
    # Create the stuff documents chain
    chain = create_stuff_documents_chain(llm, prompt_template)
    
    return chain

def parse_structured_summary(summary_text):
    """Parse the structured summary to extract sections and points"""
    lines = summary_text.strip().split('\n')
    structured_summary = []
    current_section = None
    current_subsection = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check for main section headers (bold)
        if line.startswith('**') and line.endswith('**'):
            if current_section:
                structured_summary.append(current_section)
            current_section = {
                'title': line.strip('*'),
                'subsections': [],
                'points': []
            }
            current_subsection = None
            
        # Check for subsection headers (ending with colon)
        elif line.endswith(':') and not line.startswith(('-', '•', '*')):
            if current_section:
                if current_subsection:
                    current_section['subsections'].append(current_subsection)
                current_subsection = {
                    'title': line.rstrip(':'),
                    'points': []
                }
                
        # Check for bullet points
        elif line.startswith(('-', '•', '*')):
            point = line[1:].strip()
            if current_subsection:
                current_subsection['points'].append(point)
            elif current_section:
                current_section['points'].append(point)
                
        # Check for indented sub-points
        elif line.startswith('    ') and line.strip().startswith(('-', '•', '*')):
            sub_point = line.strip()[1:].strip()
            if current_subsection and current_subsection['points']:
                # Add as sub-point to the last main point
                last_point = current_subsection['points'][-1]
                current_subsection['points'][-1] = f"{last_point}\n        • {sub_point}"
            elif current_section and current_section['points']:
                last_point = current_section['points'][-1]
                current_section['points'][-1] = f"{last_point}\n        • {sub_point}"
    
    # Add the last section
    if current_section:
        if current_subsection:
            current_section['subsections'].append(current_subsection)
        structured_summary.append(current_section)
    
    return structured_summary

def create_detailed_pdf_summary(structured_summary, original_filename, raw_summary):
    """Create PDF with detailed structured summary using ReportLab"""
    buffer = BytesIO()
    
    # Create PDF document
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=50,
        leftMargin=50,
        topMargin=50,
        bottomMargin=50
    )
    
    # Get styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=20,
        alignment=TA_CENTER,
        textColor='darkblue'
    )
    
    section_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading1'],
        fontSize=14,
        spaceAfter=12,
        spaceBefore=20,
        textColor='darkblue'
    )
    
    subsection_style = ParagraphStyle(
        'SubsectionHeader',
        parent=styles['Heading2'],
        fontSize=12,
        spaceAfter=8,
        spaceBefore=12,
        textColor='darkgreen'
    )
    
    bullet_style = ParagraphStyle(
        'BulletPoint',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=6,
        leftIndent=20,
        bulletIndent=10,
    )
    
    sub_bullet_style = ParagraphStyle(
        'SubBulletPoint',
        parent=styles['Normal'],
        fontSize=9,
        spaceAfter=4,
        leftIndent=40,
        bulletIndent=30,
    )
    
    # Build content
    content = []
    
    # Title
    content.append(Paragraph("Detailed Document Summary", title_style))
    content.append(Spacer(1, 12))
    
    # Document info
    content.append(Paragraph(f"<b>Source Document:</b> {original_filename}", styles['Normal']))
    content.append(Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    content.append(Spacer(1, 20))
    
    # Add structured content
    if structured_summary:
        for section in structured_summary:
            # Section title
            content.append(Paragraph(section['title'], section_style))
            
            # Section points (if any)
            for point in section['points']:
                if '\n        •' in point:
                    # Handle multi-level points
                    main_point, sub_points = point.split('\n        •', 1)
                    content.append(Paragraph(f"• {main_point}", bullet_style))
                    for sub_point in sub_points.split('\n        •'):
                        content.append(Paragraph(f"◦ {sub_point.strip()}", sub_bullet_style))
                else:
                    content.append(Paragraph(f"• {point}", bullet_style))
            
            # Subsections
            for subsection in section['subsections']:
                content.append(Paragraph(f"{subsection['title']}:", subsection_style))
                
                for point in subsection['points']:
                    if '\n        •' in point:
                        # Handle multi-level points
                        main_point, sub_points = point.split('\n        •', 1)
                        content.append(Paragraph(f"• {main_point}", bullet_style))
                        for sub_point in sub_points.split('\n        •'):
                            content.append(Paragraph(f"◦ {sub_point.strip()}", sub_bullet_style))
                    else:
                        content.append(Paragraph(f"• {point}", bullet_style))
            
            content.append(Spacer(1, 12))
    else:
        # Fallback: display raw summary if parsing fails
        content.append(Paragraph("Summary Content:", section_style))
        # Split long text into paragraphs
        paragraphs = raw_summary.split('\n\n')
        for para in paragraphs:
            if para.strip():
                content.append(Paragraph(para.strip(), styles['Normal']))
                content.append(Spacer(1, 8))
    
    # Build PDF
    doc.build(content)
    buffer.seek(0)
    
    return buffer

def main():
    st.set_page_config(
        page_title="Detailed Document Summarizer",
        page_icon="📋",
        layout="wide"
    )
    
    st.title("📋 Advanced Document Summarizer")
    st.markdown("Generate comprehensive, structured summaries with hierarchical organization and detailed bullet points!")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key"
        )
        
        # Model selection - defaulting to GPT-4 for better quality
        model_options = ["gpt-4o", "gpt-4-turbo-preview", "gpt-3.5-turbo"]
        selected_model = st.selectbox("Select Model", model_options, help="GPT-4 recommended for detailed summaries")
        
        # Processing settings
        st.info("📊 **Enhanced Processing Settings**")
        chunk_size = st.slider("Chunk Size", 1000, 4000, 2000, help="Larger chunks preserve more context")
        chunk_overlap = st.slider("Chunk Overlap", 200, 800, 400, help="Higher overlap ensures continuity")
        
        st.text("✅ English Text Extraction: Enabled")
        st.text("✅ Structured Output: Enabled")
        st.text("✅ Hierarchical Organization: Enabled")
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📁 Upload Document")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['pdf', 'txt', 'md'],
            help="Upload PDF, TXT, or Markdown files for detailed analysis"
        )
        
        if uploaded_file is not None:
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            
            # Display file info
            file_size = len(uploaded_file.getvalue())
            st.info(f"📊 File size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
    
    with col2:
        st.header("🚀 Generate Summary")
        
        # Summary type selection
        summary_type = st.radio(
            "Summary Detail Level:",
            ["Comprehensive", "Standard"],
            help="Comprehensive provides more detailed analysis"
        )
        
        if st.button("🔄 Generate Detailed Summary", type="primary", disabled=not (uploaded_file and api_key)):
            if not api_key:
                st.error("❌ Please enter your OpenAI API key in the sidebar.")
            elif not uploaded_file:
                st.error("❌ Please upload a document first.")
            else:
                with st.spinner("🔍 Analyzing document and generating comprehensive summary..."):
                    try:
                        # Load and process document
                        documents = load_document(uploaded_file)
                        if documents is None:
                            st.stop()
                        
                        # Split documents with configurable chunk size
                        doc_chunks = split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                        st.info(f"📄 Document processed into {len(doc_chunks)} chunks for analysis")
                        
                        # Create summary chain
                        chain = create_detailed_summary_chain(api_key, selected_model)
                        
                        # Generate summary
                        summary = chain.invoke({"context": doc_chunks})
                        
                        # Store in session state
                        st.session_state.summary = summary
                        st.session_state.filename = uploaded_file.name
                        st.session_state.structured_summary = parse_structured_summary(summary)
                        
                        st.success("✅ Detailed summary generated successfully!")
                        
                    except Exception as e:
                        st.error(f"❌ Error generating summary: {str(e)}")
    
    # Display summary
    if hasattr(st.session_state, 'summary') and st.session_state.summary:
        st.header("📋 Generated Detailed Summary")
        
        # Display structured summary
        if st.session_state.structured_summary:
            for section in st.session_state.structured_summary:
                st.subheader(f"📌 {section['title']}")
                
                # Section points
                for point in section['points']:
                    if '\n        •' in point:
                        # Handle multi-level points
                        main_point, sub_points = point.split('\n        •', 1)
                        st.markdown(f"• **{main_point}**")
                        for sub_point in sub_points.split('\n        •'):
                            st.markdown(f"    ◦ {sub_point.strip()}")
                    else:
                        st.markdown(f"• {point}")
                
                # Subsections
                for subsection in section['subsections']:
                    st.markdown(f"**{subsection['title']}:**")
                    for point in subsection['points']:
                        if '\n        •' in point:
                            # Handle multi-level points
                            main_point, sub_points = point.split('\n        •', 1)
                            st.markdown(f"  • **{main_point}**")
                            for sub_point in sub_points.split('\n        •'):
                                st.markdown(f"      ◦ {sub_point.strip()}")
                        else:
                            st.markdown(f"  • {point}")
                
                st.markdown("---")
        else:
            # Fallback display
            st.markdown(st.session_state.summary)
        
        # PDF Export section
        st.header("📄 Export Summary")
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("🔧 Generate PDF", type="secondary"):
                with st.spinner("📝 Creating detailed PDF summary..."):
                    try:
                        pdf_buffer = create_detailed_pdf_summary(
                            st.session_state.structured_summary,
                            st.session_state.filename,
                            st.session_state.summary
                        )
                        st.session_state.pdf_buffer = pdf_buffer
                        st.success("✅ PDF generated successfully!")
                    except Exception as e:
                        st.error(f"❌ Error creating PDF: {str(e)}")
        
        with col2:
            if hasattr(st.session_state, 'pdf_buffer'):
                st.download_button(
                    label="⬇️ Download PDF Summary",
                    data=st.session_state.pdf_buffer,
                    file_name=f"detailed_summary_{st.session_state.filename.split('.')[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )
        
        with col3:
            # Option to download raw text
            if st.button("📝 Download Text"):
                st.download_button(
                    label="⬇️ Download Text Summary",
                    data=st.session_state.summary,
                    file_name=f"summary_{st.session_state.filename.split('.')[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
    
    # Footer with tips
    st.markdown("---")
    with st.expander("💡 Tips for Better Summaries"):
        st.markdown("""
        **For Best Results:**
        - Use GPT-4 model for more detailed and accurate summaries
        - Upload clear, well-structured documents
        - Increase chunk size for longer documents with complex structure
        - Use 'Comprehensive' mode for regulatory or technical documents
        
        **Features:**
        - ✅ Hierarchical structure preservation
        - ✅ Detailed bullet points with sub-points
        - ✅ Technical term preservation
        - ✅ Compliance requirement extraction
        - ✅ Multi-format export (PDF, Text)
        """)
    
    st.markdown("**Built with:** Streamlit • LangChain • OpenAI • ReportLab")

if __name__ == "__main__":
    main()
