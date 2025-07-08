import streamlit as st
import tempfile
import os
import re
from io import BytesIO
from datetime import datetime
import langdetect
from langdetect.lang_detect_exception import LangDetectException
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain.schema import Document
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT, TA_CENTER

def extract_english_text(text):
    try:
        sentences = re.split(r'[.!?]+', text)
        english_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:
                try:
                    lang = langdetect.detect(sentence)
                    if lang == 'en':
                        english_sentences.append(sentence)
                except LangDetectException:
                    if re.search(r'\b(the|and|or|of|to|in|for|with|by|from|at|is|are|was|were)\b', sentence.lower()):
                        english_sentences.append(sentence)
        
        return '. '.join(english_sentences) + '.'
    
    except Exception as e:
        st.warning(f"Language detection error: {e}. Using original text.")
        return text

def load_document(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        
        processed_docs = []
        for doc in documents:
            original_content = doc.page_content
            if len(re.findall(r'[^\x00-\x7F]', original_content)) > len(original_content) * 0.1:
                doc.page_content = extract_english_text(original_content)
            else:
                doc.page_content = original_content
            
            if len(doc.page_content.strip()) > 50:
                processed_docs.append(doc)
        
        os.unlink(tmp_file_path)
        
        return processed_docs
    
    except Exception as e:
        st.error(f"Error loading document: {str(e)}")
        return None

def split_documents(documents, chunk_size=1500, chunk_overlap=500):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
        keep_separator=True
    )
    return text_splitter.split_documents(documents)

def create_enhanced_summary_chain(api_key, model_name="gpt-4o"):
    llm = AzureChatOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=api_key,
        api_version=api_version,
        deployment_name=deployment_name,
        temperature=0.1,
        max_tokens=4000
    )
    
    prompt_template = ChatPromptTemplate.from_template("""
    You are a specialized IRDAI regulatory document analyst with deep expertise in Indian insurance regulations. Create an ULTRA-COMPREHENSIVE, FORENSIC-LEVEL summary that captures EVERY regulatory detail, requirement, and nuance.

    CRITICAL REGULATORY EXTRACTION REQUIREMENTS:

    1. **REGULATORY FRAMEWORK ANALYSIS**:
       - Extract ALL regulatory references (circulars, notifications, guidelines)
       - Identify ALL sections, sub-sections, clauses, and sub-clauses
       - Capture ALL cross-references to other regulations
       - Note ALL amendments, modifications, and updates
       - Detail ALL effective dates, implementation timelines, and transition periods

    2. **COMPLIANCE SPECIFICATIONS**:
       - Extract ALL compliance requirements with exact wording
       - Capture ALL numerical thresholds, limits, ratios, and percentages
       - Detail ALL reporting obligations with frequencies and formats
       - Note ALL penalties, sanctions, and enforcement actions
       - Include ALL exceptions, exemptions, and special provisions

    3. **OPERATIONAL REQUIREMENTS**:
       - Extract ALL procedural steps and approval processes
       - Detail ALL documentation requirements and formats
       - Capture ALL timelines, deadlines, and notice periods
       - Note ALL roles, responsibilities, and authorities
       - Include ALL governance and oversight requirements

    4. **FINANCIAL AND QUANTITATIVE DETAILS**:
       - Extract ALL monetary amounts, capital requirements, and financial ratios
       - Capture ALL percentage requirements and calculation methodologies
       - Detail ALL investment limits, asset allocation rules, and restrictions
       - Note ALL valuation methods and accounting treatments
       - Include ALL risk management and solvency requirements

    5. **ENTITY-SPECIFIC REQUIREMENTS**:
       - Extract requirements for insurers, reinsurers, brokers, agents, and TPAs
       - Detail licensing, registration, and authorization requirements
       - Capture ownership, management, and operational criteria
       - Note conduct of business rules and customer protection measures
       - Include market conduct and fair practices requirements

    6. **TECHNICAL AND ACTUARIAL SPECIFICATIONS**:
       - Extract ALL technical reserves and liability calculations
       - Detail ALL actuarial requirements and certification processes
       - Capture ALL product approval and filing requirements
       - Note ALL underwriting and claims settlement guidelines
       - Include ALL risk assessment and management protocols

    FORENSIC ENHANCEMENT PROCESS:
    - **First Pass**: Systematic extraction of all visible regulatory elements
    - **Second Pass**: Cross-referencing and context analysis for implied requirements
    - **Third Pass**: Validation against IRDAI regulatory framework and terminology
    - **Fourth Pass**: Integration of all quantitative data and technical specifications
    - **Final Pass**: Comprehensive structuring with regulatory hierarchy preservation

    SPECIALIZED FORMATTING FOR IRDAI DOCUMENTS:
    - Use precise regulatory section numbering (e.g., "Section 3.2.1(a)(i)")
    - Preserve exact regulatory language and terminology
    - Create detailed sub-hierarchies for complex requirements
    - Use regulatory-specific markers: [MANDATORY], [OPTIONAL], [CONDITIONAL]
    - Include implementation phases: [IMMEDIATE], [PHASED], [TRANSITIONAL]
    - Mark compliance levels: [MINIMUM], [ENHANCED], [BEST PRACTICE]

    CRITICAL CONTENT CATEGORIES FOR IRDAI:

    **REGULATORY AUTHORITY & SCOPE**
    - Legal basis and statutory powers
    - Applicability scope and entity coverage
    - Jurisdictional boundaries and limitations

    **LICENSING & REGISTRATION**
    - Application procedures and documentation
    - Eligibility criteria and qualification requirements
    - Capital requirements and financial strength measures
    - Fit and proper criteria for key personnel

    **OPERATIONAL COMPLIANCE**
    - Business conduct requirements
    - Customer protection and grievance handling
    - Market conduct and fair practices
    - Anti-fraud and risk management measures

    **FINANCIAL REGULATIONS**
    - Capital adequacy and solvency requirements
    - Investment guidelines and restrictions
    - Accounting and reporting standards
    - Audit and actuarial requirements

    **PRODUCT & PRICING**
    - Product approval and filing requirements
    - Pricing guidelines and restrictions
    - Policy terms and conditions standards
    - Claims settlement procedures

    **GOVERNANCE & OVERSIGHT**
    - Board composition and responsibilities
    - Risk management framework
    - Internal audit and compliance functions
    - Outsourcing guidelines and controls

    EXAMPLE ENHANCED STRUCTURE:
    **[REGULATION TITLE] - [EFFECTIVE DATE]**

    **1. REGULATORY FRAMEWORK**
    - **Legal Basis**: [Exact statutory reference]
        - **Authority**: [Specific powers exercised]
        - **Scope**: [Detailed applicability with entity types]
        - **Enforcement**: [Penalties and sanctions with amounts]

    **2. COMPLIANCE REQUIREMENTS** [MANDATORY]
    - **Requirement 2.1**: [Exact requirement with all specifications]
        - **Threshold**: [Precise numerical limits with units]
        - **Timeline**: [Specific deadlines and implementation phases]
        - **Documentation**: [Required formats and submission procedures]
        - **Reporting**: [Frequency, format, and recipient details]
        - **Exceptions**: [Specific conditions and alternative procedures]

    Document content to analyze:
    {context}
    
    ULTRA-COMPREHENSIVE FORENSIC REGULATORY SUMMARY (capture EVERY specification with enhanced regulatory context):
    """)
    
    chain = create_stuff_documents_chain(llm, prompt_template)
    
    return chain

def validate_summary_completeness(summary, original_chunks):
    original_numbers = set()
    for chunk in original_chunks:
        numbers = re.findall(r'\b\d+(?:\.\d+)?%?\b', chunk.page_content)
        original_numbers.update(numbers)
    
    summary_numbers = set(re.findall(r'\b\d+(?:\.\d+)?%?\b', summary))
    
    if original_numbers:
        coverage = len(summary_numbers.intersection(original_numbers)) / len(original_numbers)
        return coverage, original_numbers, summary_numbers
    
    return 1.0, set(), set()

def generate_comprehensive_enhanced_summary(doc_chunks, api_key, model_name):
    
    chain = create_enhanced_summary_chain(api_key, model_name)
    initial_summary = chain.invoke({"context": doc_chunks})
    
    coverage, orig_numbers, summ_numbers = validate_summary_completeness(initial_summary, doc_chunks)
    
    if coverage < 0.8:        
        llm = AzureChatOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version,
            deployment_name=deployment_name,
            temperature=0.1)
        
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
        
        combined_chunks = "\n\n".join([chunk.page_content for chunk in doc_chunks[:8]])
        
        try:
            enhanced_summary = llm.invoke(enhancement_prompt.format(
                summary=initial_summary,
                chunks=combined_chunks,
                orig_numbers=list(orig_numbers),
                summ_numbers=list(summ_numbers),
                coverage=coverage
            ))
            
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
    lines = summary_text.strip().split('\n')
    structured_summary = []
    current_section = None
    current_subsection = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
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
            
        elif line.endswith(':') and not line.startswith(('-', '•', '*', '◦')):
            if current_section:
                if current_subsection:
                    current_section['subsections'].append(current_subsection)
                current_subsection = {
                    'title': line.rstrip(':'),
                    'points': []
                }
                
        elif re.match(r'^[-•*◦]\s', line):
            point = re.sub(r'^[-•*◦]\s', '', line)
            if current_subsection:
                current_subsection['points'].append(point)
            elif current_section:
                current_section['points'].append(point)
                
        elif re.match(r'^\d+\.\s', line):
            point = re.sub(r'^\d+\.\s', '', line)
            if current_subsection:
                current_subsection['points'].append(point)
            elif current_section:
                current_section['points'].append(point)
                
        elif re.match(r'^\s{2,}[-•*◦]\s', line):
            sub_point = re.sub(r'^\s*[-•*◦]\s', '', line)
            if current_subsection and current_subsection['points']:
                last_point = current_subsection['points'][-1]
                current_subsection['points'][-1] = f"{last_point}\n        • {sub_point}"
            elif current_section and current_section['points']:
                last_point = current_section['points'][-1]
                current_section['points'][-1] = f"{last_point}\n        • {sub_point}"
        
        elif current_subsection and current_subsection['points']:
            if not re.match(r'^[-•*◦\d]', line):
                current_subsection['points'][-1] += f" {line}"
        elif current_section and current_section['points']:
            if not re.match(r'^[-•*◦\d]', line):
                current_section['points'][-1] += f" {line}"
    
    if current_section:
        if current_subsection:
            current_section['subsections'].append(current_subsection)
        structured_summary.append(current_section)
    
    return structured_summary

def create_detailed_pdf_summary(structured_summary, original_filename, raw_summary):
    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

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

    content = []

    content.append(Paragraph(f"Summary of {os.path.splitext(original_filename)[0]}", title_style))
    content.append(Spacer(1, 12))

    if structured_summary:
        for i, section in enumerate(structured_summary):
            for point in section['points']:
                formatted_point = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', point)
                
                if '\n        •' in formatted_point:
                    main_point, sub_points = formatted_point.split('\n        •', 1)
                    content.append(Paragraph(f"• {main_point}", bullet_style))
                    for sub_point in sub_points.split('\n        •'):
                        if sub_point.strip():
                            formatted_sub_point = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', sub_point.strip())
                            content.append(Paragraph(f"◦ {formatted_sub_point}", sub_bullet_style))
                else:
                    content.append(Paragraph(f"• {formatted_point}", bullet_style))

            for subsection in section['subsections']:
                for point in subsection['points']:
                    formatted_point = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', point)
                    
                    if '\n        •' in formatted_point:
                        main_point, sub_points = formatted_point.split('\n        •', 1)
                        content.append(Paragraph(f"• {main_point}", bullet_style))
                        for sub_point in sub_points.split('\n        •'):
                            if sub_point.strip():
                                formatted_sub_point = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', sub_point.strip())
                                content.append(Paragraph(f"◦ {formatted_sub_point}", sub_bullet_style))
                    else:
                        content.append(Paragraph(f"• {formatted_point}", bullet_style))

            if i < len(structured_summary) - 1:
                content.append(Spacer(1, 6))
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
    
    with st.sidebar:
        st.header("⚙️ Azure OpenAI Configuration")
        
        # Azure OpenAI specific fields
        azure_endpoint = st.text_input(
            "Azure OpenAI Endpoint",
            placeholder="https://your-resource.openai.azure.com/",
            help="Your Azure OpenAI endpoint URL"
        )
        
        api_key = st.text_input(
            "Azure OpenAI API Key",
            type="password",
            help="Enter your Azure OpenAI API key"
        )
        
        api_version = st.selectbox(
            "API Version",
            ["2024-02-01", "2023-12-01-preview", "2023-05-15"],
            help="Select Azure OpenAI API version"
        )
        
        # Azure deployment names instead of model names
        deployment_name = st.text_input(
            "Deployment Name",
            placeholder="gpt-4o",
            help="Enter your Azure OpenAI deployment name"
        )
    
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
                        documents = load_document(uploaded_file)
                        if documents is None:
                            st.stop()
                        
                        total_content = sum(len(doc.page_content) for doc in documents)
                        st.info(f"📄 Document loaded: {len(documents)} pages, {total_content:,} characters")
                        
                        doc_chunks = split_documents(documents, chunk_size=1500, chunk_overlap=500)
                        st.info(f"📑 Document processed into {len(doc_chunks)} chunks for analysis")
                        
                        summary, coverage = generate_comprehensive_enhanced_summary(
                            doc_chunks, api_key, selected_model
                        )
                        
                        st.session_state.summary = summary
                        st.session_state.filename = uploaded_file.name
                        st.session_state.structured_summary = parse_structured_summary(summary)
                        st.session_state.coverage = coverage
                        
                        st.success("✅ Comprehensive enhanced summary generated successfully!")
                        st.balloons()
                        
                    except Exception as e:
                        st.error(f"❌ Error generating summary: {str(e)}")
                        st.exception(e)
    
    if hasattr(st.session_state, 'summary') and st.session_state.summary:
        st.header("📋 Generated Enhanced Summary")
        
        if st.session_state.structured_summary:
            for i, section in enumerate(st.session_state.structured_summary):
                with st.expander(f"📌 {section['title']}", expanded=True):
                    
                    for point in section['points']:
                        if '\n        •' in point:
                            main_point, sub_points = point.split('\n        •', 1)
                            st.markdown(f"• **{main_point}**")
                            for sub_point in sub_points.split('\n        •'):
                                if sub_point.strip():
                                    st.markdown(f"    ◦ {sub_point.strip()}")
                        else:
                            st.markdown(f"• {point}")
                    
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
            st.markdown(st.session_state.summary)
        
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

if __name__ == "__main__":
    main()
