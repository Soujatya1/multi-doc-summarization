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
from langchain.document_loaders import PyPDFLoader
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
    """Extract English words from the given text - but preserve structure"""
    try:
        # Instead of word-by-word detection, use sentence-level detection
        sentences = re.split(r'[.!?]+', text)
        english_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Only process substantial sentences
                try:
                    lang = langdetect.detect(sentence)
                    if lang == 'en':
                        english_sentences.append(sentence)
                except LangDetectException:
                    # If detection fails, assume it might be English if it contains common English patterns
                    if re.search(r'\b(the|and|or|of|to|in|for|with|by|from|at|is|are|was|were)\b', sentence.lower()):
                        english_sentences.append(sentence)
        
        return '. '.join(english_sentences) + '.'
    
    except Exception as e:
        st.warning(f"Language detection error: {e}. Using original text.")
        return text

def load_document(uploaded_file):
    """Load document from uploaded file with better error handling"""
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Load PDF file
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        
        # Extract English text from each document - but preserve more content
        processed_docs = []
        for doc in documents:
            # Only apply language filtering if the document seems to have mixed languages
            original_content = doc.page_content
            if len(re.findall(r'[^\x00-\x7F]', original_content)) > len(original_content) * 0.1:
                # High ratio of non-ASCII characters, apply filtering
                doc.page_content = extract_english_text(original_content)
            else:
                # Mostly ASCII, keep original
                doc.page_content = original_content
            
            # Only keep documents with substantial content
            if len(doc.page_content.strip()) > 50:
                processed_docs.append(doc)
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return processed_docs
    
    except Exception as e:
        st.error(f"Error loading document: {str(e)}")
        return None

def split_documents(documents, chunk_size=1500, chunk_overlap=500):
    """Split documents into chunks with fixed optimal settings"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
        keep_separator=True  # Preserve separators for better context
    )
    return text_splitter.split_documents(documents)

def create_enhanced_summary_chain(api_key, model_name="gpt-4o"):
    """Create the enhanced summarization chain with built-in detail enhancement"""
    llm = ChatOpenAI(
        api_key=api_key,
        model_name=model_name,
        temperature=0.1,
        max_tokens=4000  # Ensure enough tokens for detailed output
    )
    
    # Enhanced prompt template with built-in detail enhancement
    prompt_template = ChatPromptTemplate.from_template("""
    You are an expert document analyst. Your task is to create an EXHAUSTIVE, ENHANCED summary that captures EVERY important detail from the document.

    CRITICAL EXTRACTION REQUIREMENTS:
    1. **PRESERVE EXACT SPECIFICATIONS**: Include ALL numbers, percentages, monetary amounts, dates, timeframes, and quantities EXACTLY as stated
    2. **CAPTURE COMPLETE DEFINITIONS**: Extract full definitions of technical terms, roles, and concepts
    3. **DETAIL ALL PROCEDURES**: Include step-by-step processes, approval workflows, and operational requirements
    4. **EXTRACT ORGANIZATIONAL DETAILS**: Committee structures, reporting relationships, roles, and responsibilities
    5. **COMPLIANCE REQUIREMENTS**: All regulatory obligations, deadlines, reporting requirements, and penalties
    6. **CONDITIONAL REQUIREMENTS**: Exceptions, special cases, and alternative procedures

    ENHANCEMENT PROCESS:
    - First pass: Extract all visible information systematically
    - Second pass: Cross-reference and identify any missing details
    - Third pass: Ensure all quantitative data and technical specifications are captured
    - Final pass: Organize and structure for maximum clarity and completeness

    FORMATTING REQUIREMENTS:
    - Use hierarchical structure with clear section headers
    - Create comprehensive bullet points that preserve ALL details
    - Use sub-bullets for related details and specifications
    - Bold important terms, amounts, and requirements
    - Preserve technical language and regulatory terminology
    - Include exact quotes for critical requirements (in quotation marks)

    CONTENT ANALYSIS PRIORITIES:
    1. **Quantitative Data**: Every number, percentage, amount, timeline, limit, or threshold
    2. **Qualifications**: Required experience, skills, tenure, independence criteria
    3. **Processes**: Approval steps, notification requirements, meeting procedures
    4. **Compositions**: Committee sizes, member types, quorum requirements
    5. **Timelines**: Deadlines, notice periods, tenure limits, meeting frequencies
    6. **Authorities**: Who has what powers, approval rights, and responsibilities
    7. **Exceptions**: Special circumstances, exemptions, alternative procedures

    EXAMPLE STRUCTURE (adapt to actual content):
    **[Section Title from Document]**
    
    [Subsection]:
    - Complete requirement with ALL specifications (include exact numbers/dates)
        - Sub-requirement with precise details
        - Additional specifications or conditions
    - Next requirement with full technical details
        - Related procedural steps
        - Specific compliance obligations

    Document content:
    {context}
    
    EXHAUSTIVE ENHANCED DETAILED SUMMARY (capture EVERY specification and requirement with built-in enhancement):
    """)
    
    # Create the stuff documents chain
    chain = create_stuff_documents_chain(llm, prompt_template)
    
    return chain

def validate_summary_completeness(summary, original_chunks):
    """Validate if the summary captures key elements from the original document"""
    # Extract numbers from original document
    original_numbers = set()
    for chunk in original_chunks:
        numbers = re.findall(r'\b\d+(?:\.\d+)?%?\b', chunk.page_content)
        original_numbers.update(numbers)
    
    # Extract numbers from summary
    summary_numbers = set(re.findall(r'\b\d+(?:\.\d+)?%?\b', summary))
    
    # Calculate coverage
    if original_numbers:
        coverage = len(summary_numbers.intersection(original_numbers)) / len(original_numbers)
        return coverage, original_numbers, summary_numbers
    
    return 1.0, set(), set()

def generate_comprehensive_enhanced_summary(doc_chunks, api_key, model_name):
    """Generate comprehensive summary with built-in enhancement"""
    
    # Step 1: Generate initial enhanced summary
    chain = create_enhanced_summary_chain(api_key, model_name)
    initial_summary = chain.invoke({"context": doc_chunks})
    
    # Step 2: Validate completeness
    coverage, orig_numbers, summ_numbers = validate_summary_completeness(initial_summary, doc_chunks)
    
    # Step 3: If coverage is insufficient, perform additional enhancement
    if coverage < 0.8:        
        llm = ChatOpenAI(api_key=api_key, model_name=model_name, temperature=0.1)
        
        enhancement_prompt = ChatPromptTemplate.from_template("""
        Review the following summary against the original document chunks to identify and add any missing critical details.
        
        MISSING DETAILS ANALYSIS:
        Original numbers found: {orig_numbers}
        Summary numbers captured: {summ_numbers}
        Coverage: {coverage:.1%}
        
        ENHANCEMENT INSTRUCTIONS:
        1. Identify all missing quantitative data from the original document
        2. Add any missing specifications, procedures, or requirements
        3. Ensure all technical definitions are complete
        4. Verify all organizational details are captured
        5. Cross-reference to eliminate any gaps
        
        Original Summary:
        {summary}
        
        Original Document Chunks (for reference):
        {chunks}
        
        ENHANCED COMPREHENSIVE SUMMARY with ALL missing details integrated:
        """)
        
        # Combine chunks for context (limit to avoid token limits)
        combined_chunks = "\n\n".join([chunk.page_content for chunk in doc_chunks[:8]])
        
        try:
            enhanced_summary = llm.invoke(enhancement_prompt.format(
                summary=initial_summary,
                chunks=combined_chunks,
                orig_numbers=list(orig_numbers),
                summ_numbers=list(summ_numbers),
                coverage=coverage
            ))
            
            # Validate the enhanced version
            final_coverage, _, _ = validate_summary_completeness(enhanced_summary.content, doc_chunks)
            
            st.success(f"✅ Enhancement complete! Coverage improved from {coverage:.1%} to {final_coverage:.1%}")
            
            return enhanced_summary.content, final_coverage
            
        except Exception as e:
            st.warning(f"Enhancement failed: {e}. Using initial summary.")
            return initial_summary, coverage
    
    else:
        st.success(f"✅ Initial summary achieved {coverage:.1%} coverage - enhancement not needed")
        return initial_summary, coverage

def parse_structured_summary(summary_text):
    """Enhanced parsing to better handle structured content"""
    lines = summary_text.strip().split('\n')
    structured_summary = []
    current_section = None
    current_subsection = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check for main section headers (bold or all caps)
        if (line.startswith('**') and line.endswith('**')) or line.isupper():
            if current_section:
                if current_subsection:
                    current_section['subsections'].append(current_subsection)
                structured_summary.append(current_section)
            current_section = {
                'title': line.strip('*'),
                'subsections': [],
                'points': []
            }
            current_subsection = None
            
        # Check for subsection headers
        elif line.endswith(':') and not line.startswith(('-', '•', '*', '◦')):
            if current_section:
                if current_subsection:
                    current_section['subsections'].append(current_subsection)
                current_subsection = {
                    'title': line.rstrip(':'),
                    'points': []
                }
                
        # Check for bullet points
        elif re.match(r'^[-•*◦]\s', line):
            point = re.sub(r'^[-•*◦]\s', '', line)
            if current_subsection:
                current_subsection['points'].append(point)
            elif current_section:
                current_section['points'].append(point)
                
        # Check for numbered points
        elif re.match(r'^\d+\.\s', line):
            point = re.sub(r'^\d+\.\s', '', line)
            if current_subsection:
                current_subsection['points'].append(point)
            elif current_section:
                current_section['points'].append(point)
                
        # Check for indented sub-points
        elif re.match(r'^\s{2,}[-•*◦]\s', line):
            sub_point = re.sub(r'^\s*[-•*◦]\s', '', line)
            if current_subsection and current_subsection['points']:
                last_point = current_subsection['points'][-1]
                current_subsection['points'][-1] = f"{last_point}\n        • {sub_point}"
            elif current_section and current_section['points']:
                last_point = current_section['points'][-1]
                current_section['points'][-1] = f"{last_point}\n        • {sub_point}"
        
        # Handle lines that might be continuation of previous points
        elif current_subsection and current_subsection['points']:
            # If line doesn't start with bullet or number, it might be continuation
            if not re.match(r'^[-•*◦\d]', line):
                current_subsection['points'][-1] += f" {line}"
        elif current_section and current_section['points']:
            if not re.match(r'^[-•*◦\d]', line):
                current_section['points'][-1] += f" {line}"
    
    # Add the last section
    if current_section:
        if current_subsection:
            current_section['subsections'].append(current_subsection)
        structured_summary.append(current_section)
    
    return structured_summary

def create_detailed_pdf_summary(structured_summary, original_filename, raw_summary):
    """Create PDF with enhanced formatting and better content organization"""
    buffer = BytesIO()

    # Create PDF document with more generous margins
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    # Get styles and create enhanced custom styles
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=20,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor='darkblue'
    )

    section_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading1'],
        fontSize=14,
        spaceAfter=12,
        spaceBefore=20,
        textColor='darkblue',
        keepWithNext=True
    )

    subsection_style = ParagraphStyle(
        'SubsectionHeader',
        parent=styles['Heading2'],
        fontSize=12,
        spaceAfter=8,
        spaceBefore=12,
        textColor='darkgreen',
        keepWithNext=True
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

    # Title and metadata with document name
    content.append(Paragraph(f"Summary of {os.path.splitext(original_filename)[0]}", title_style))
    content.append(Spacer(1, 12))

    # Add structured content with better formatting
    if structured_summary:
        for i, section in enumerate(structured_summary):
            # Skip generic initial summary section
            if i == 0 and 'Regulatory and Development Authority' in section['title']:
                continue

            # Section title
            content.append(Paragraph(f"{i+1}. {section['title']}", section_style))

            # Section points
            for point in section['points']:
                if '\n        •' in point:
                    main_point, sub_points = point.split('\n        •', 1)
                    content.append(Paragraph(f"• {main_point}", bullet_style))
                    for sub_point in sub_points.split('\n        •'):
                        if sub_point.strip():
                            content.append(Paragraph(f"◦ {sub_point.strip()}", sub_bullet_style))
                else:
                    content.append(Paragraph(f"• {point}", bullet_style))

            # Subsections
            for subsection in section['subsections']:
                content.append(Paragraph(f"{subsection['title']}:", subsection_style))

                for point in subsection['points']:
                    if '\n        •' in point:
                        main_point, sub_points = point.split('\n        •', 1)
                        content.append(Paragraph(f"• {main_point}", bullet_style))
                        for sub_point in sub_points.split('\n        •'):
                            if sub_point.strip():
                                content.append(Paragraph(f"◦ {sub_point.strip()}", sub_bullet_style))
                    else:
                        content.append(Paragraph(f"• {point}", bullet_style))

            # Add spacing between sections
            if i < len(structured_summary) - 1:
                content.append(Spacer(1, 16))
    else:
        content.append(Paragraph("Summary Content:", section_style))
        paragraphs = raw_summary.split('\n\n')
        for para in paragraphs:
            if para.strip():
                if para.strip().startswith('**') and para.strip().endswith('**'):
                    content.append(Paragraph(para.strip('*'), section_style))
                elif para.strip().startswith(('- ', '• ', '* ')):
                    content.append(Paragraph(para.strip()[2:], bullet_style))
                else:
                    content.append(Paragraph(para.strip(), styles['Normal']))
                content.append(Spacer(1, 8))

    # Build PDF
    doc.build(content)
    buffer.seek(0)

    return buffer

def main():
    st.set_page_config(
        page_title="Enhanced Document Summarizer",
        page_icon="📋",
        layout="wide"
    )
    
    st.title("📋 Enhanced Detailed Document Summarizer")
    st.markdown("Generate comprehensive, enhanced summaries that capture EVERY important detail!")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key"
        )
        
        # Model selection
        model_options = ["gpt-4o", "gpt-4-turbo-preview", "gpt-3.5-turbo"]
        selected_model = st.selectbox("Select Model", model_options, help="GPT-4 strongly recommended for detailed summaries")
        
        # Processing settings
        st.subheader("🔧 Processing Settings")
        
        # Fixed chunk settings
        st.info("📋 **Fixed Optimal Settings:**")
        st.text("• Chunk Size: 1500 characters")
        st.text("• Chunk Overlap: 500 characters")
        st.text("• Enhanced Summary: Enabled")
        
        st.success("✅ Enhanced processing with automatic detail enhancement enabled")
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📁 Upload Document")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload PDF files for detailed analysis"
        )
        
        if uploaded_file is not None:
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            
            # Display file info
            file_size = len(uploaded_file.getvalue())
            st.info(f"📊 File size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
    
    with col2:
        st.header("🚀 Generate Enhanced Summary")
        
        if st.button("🔄 Generate Comprehensive Enhanced Summary", type="primary", disabled=not (uploaded_file and api_key)):
            if not api_key:
                st.error("❌ Please enter your OpenAI API key in the sidebar.")
            elif not uploaded_file:
                st.error("❌ Please upload a document first.")
            else:
                with st.spinner("🔍 Analyzing document and generating comprehensive enhanced summary..."):
                    try:
                        # Load and process document
                        documents = load_document(uploaded_file)
                        if documents is None:
                            st.stop()
                        
                        # Calculate total content length
                        total_content = sum(len(doc.page_content) for doc in documents)
                        st.info(f"📄 Document loaded: {len(documents)} pages, {total_content:,} characters")
                        
                        # Split documents with fixed optimal settings
                        doc_chunks = split_documents(documents, chunk_size=1500, chunk_overlap=500)
                        st.info(f"📑 Document processed into {len(doc_chunks)} chunks for analysis")
                        
                        # Generate comprehensive enhanced summary
                        summary, coverage = generate_comprehensive_enhanced_summary(
                            doc_chunks, api_key, selected_model
                        )
                        
                        # Store in session state
                        st.session_state.summary = summary
                        st.session_state.filename = uploaded_file.name
                        st.session_state.structured_summary = parse_structured_summary(summary)
                        st.session_state.coverage = coverage
                        
                        st.success("✅ Comprehensive enhanced summary generated successfully!")
                        st.balloons()
                        
                    except Exception as e:
                        st.error(f"❌ Error generating summary: {str(e)}")
                        st.exception(e)  # Show full traceback for debugging
    
    # Display summary with enhanced formatting
    if hasattr(st.session_state, 'summary') and st.session_state.summary:
        st.header("📋 Generated Enhanced Summary")
        
        # Display structured summary with better formatting
        if st.session_state.structured_summary:
            for i, section in enumerate(st.session_state.structured_summary):
                with st.expander(f"📌 {section['title']}", expanded=True):
                    
                    # Section points
                    for point in section['points']:
                        if '\n        •' in point:
                            main_point, sub_points = point.split('\n        •', 1)
                            st.markdown(f"• **{main_point}**")
                            for sub_point in sub_points.split('\n        •'):
                                if sub_point.strip():
                                    st.markdown(f"    ◦ {sub_point.strip()}")
                        else:
                            st.markdown(f"• {point}")
                    
                    # Subsections
                    for subsection in section['subsections']:
                        st.markdown(f"**{subsection['title']}:**")
                        for point in subsection['points']:
                            if '\n        •' in point:
                                main_point, sub_points = point.split('\n        •', 1)
                                st.markdown(f"  • **{main_point}**")
                                for sub_point in sub_points.split('\n        •'):
                                    if sub_point.strip():
                                        st.markdown(f"      ◦ {sub_point.strip()}")
                            else:
                                st.markdown(f"  • {point}")
        else:
            # Fallback display with better formatting
            st.markdown(st.session_state.summary)
        
        # Export section
        st.header("📄 Export Summary")
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("🔧 Generate PDF", type="secondary"):
                with st.spinner("📝 Creating enhanced PDF summary..."):
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
                    file_name=f"enhanced_summary_{st.session_state.filename.split('.')[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )
        
        with col3:
            if st.button("📝 Download Text"):
                st.download_button(
                    label="⬇️ Download Text Summary",
                    data=st.session_state.summary,
                    file_name=f"summary_{st.session_state.filename.split('.')[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )

if __name__ == "__main__":
    main()
