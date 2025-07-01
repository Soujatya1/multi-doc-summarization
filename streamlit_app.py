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
    doc_abbrev = ''.join([c.upper() for c in doc_name if c.isalpha()])[:4]
    
    # Clean and format the summary
    lines = summary.split('\n')
    formatted_lines = []
    
    # Add header
    formatted_lines.append(f"**[{doc_abbrev}]**")
    formatted_lines.append("")
    formatted_lines.append("[Detailed Compliance Summary:]")
    formatted_lines.append("")
    
    # Process the content
    in_content = False
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Skip header lines from LLM output
        if line.startswith("**[") or line.startswith("[Summary:]") or line.startswith("[Detailed"):
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
                if not line.startswith('**[') and not line.startswith('[Summary') and not line.startswith('[Detailed'):
                    formatted_lines.append(f"- {line}")
    
    return '\n'.join(formatted_lines)

def create_summarization_chain(llm):
    """Create a comprehensive LLM chain for detailed compliance-focused document summarization."""
    
    # Enhanced comprehensive prompt for detailed compliance analysis
    summarization_prompt = PromptTemplate(
        input_variables=["text", "document_name"],
        template="""
        CRITICAL INSTRUCTIONS:
        1. DO NOT include any personally identifiable information (PII) including bank account numbers, credit card numbers, SSNs, passport numbers, personal mobile numbers
        2. Focus on DETAILED COMPLIANCE REQUIREMENTS with specific actionable items
        3. Extract ALL regulatory deadlines, timeframes, and quantitative requirements
        4. Include specific section references, regulation numbers, and authority citations
        5. Provide comprehensive analysis suitable for compliance officers and legal teams
        
        Document: {document_name}
        Content: {text}
        
        Create an EXTREMELY DETAILED compliance-focused summary following this EXACT structure:
        
        **[DOCUMENT_ABBREVIATION]**
        
        [Detailed Compliance Summary:]
        
        - **REGULATORY FRAMEWORK & AUTHORITY**
            - Issuing authority and regulation reference numbers
            - Legal basis and statutory provisions
            - Effective date and implementation timeline
            - Superseded regulations or amendments
            - Cross-references to other applicable regulations
        
        - **MANDATORY COMPLIANCE REQUIREMENTS**
            - Primary obligations with specific deadlines
            - Secondary compliance requirements
            - Ongoing monitoring and maintenance obligations
            - Documentation and record-keeping requirements
            - Reporting frequencies and submission deadlines
        
        - **BOARD GOVERNANCE & COMPOSITION**
            - Board size requirements (minimum/maximum members)
            - Independent director requirements (specific percentages)
            - Director qualification criteria and restrictions
            - Appointment, removal, and tenure procedures
            - Board meeting frequency and quorum requirements
            - Director training and certification requirements
            - Conflict of interest management protocols
        
        - **COMMITTEE STRUCTURES & RESPONSIBILITIES**
            - Mandatory committee requirements (Audit, Risk, Nomination, etc.)
            - Committee composition and independence requirements
            - Specific duties and decision-making authorities
            - Meeting frequencies and documentation requirements
            - Reporting obligations to board and regulators
            - Performance evaluation criteria
        
        - **RISK MANAGEMENT & INTERNAL CONTROLS**
            - Risk management framework requirements
            - Internal audit function specifications
            - Control testing and validation procedures
            - Risk appetite and tolerance definitions
            - Escalation procedures and thresholds
            - Business continuity and disaster recovery requirements
        
        - **KEY MANAGEMENT PERSONNEL (KMP) REQUIREMENTS**
            - Chief Compliance Officer (CCO) qualifications and duties
            - Risk Officer appointment and responsibilities
            - Chief Financial Officer requirements
            - Succession planning obligations
            - Fit and proper criteria assessments
            - Performance evaluation and compensation guidelines
        
        - **RELATED PARTY TRANSACTIONS**
            - Definition and identification criteria
            - Approval processes and authority limits
            - Disclosure requirements and formats
            - Monitoring and review procedures
            - Exemptions and materiality thresholds
            - Penalty provisions for non-compliance
        
        - **DISCLOSURE & TRANSPARENCY REQUIREMENTS**
            - Mandatory disclosure items and formats
            - Public disclosure timelines and channels
            - Regulatory filing requirements and frequencies
            - Website disclosure obligations
            - Shareholder communication requirements
            - Material event notification procedures
        
        - **AUDIT & ASSURANCE REQUIREMENTS**
            - External auditor appointment and rotation
            - Internal audit function scope and reporting
            - Audit committee oversight responsibilities
            - Audit findings remediation timelines
            - Regulatory examination cooperation requirements
        
        - **REMUNERATION & COMPENSATION**
            - Remuneration policy framework requirements
            - Variable compensation guidelines and caps
            - Clawback and malus provisions
            - Disclosure requirements for executive compensation
            - Shareholder approval requirements
        
        - **PENALTIES & ENFORCEMENT**
            - Monetary penalties and fines structure
            - Administrative actions and sanctions
            - Director/officer liability provisions
            - Remediation requirements and timelines
            - Appeal processes and procedures
        
        - **IMPLEMENTATION TIMELINE & ACTION ITEMS**
            - Immediate action items (within 30/60/90 days)
            - Medium-term implementation requirements (3-12 months)
            - Long-term compliance objectives (beyond 12 months)
            - Critical deadlines and milestones
            - Resource requirements and budget implications
        
        - **SPECIFIC QUANTITATIVE REQUIREMENTS**
            - Numerical thresholds, percentages, and limits
            - Capital adequacy ratios or financial metrics
            - Time limits for various processes
            - Frequency requirements for activities
            - Materiality thresholds for disclosures
        
        - **ADDITIONAL REGULATORY FRAMEWORKS**
            - ESG (Environmental, Social, Governance) requirements
            - Cybersecurity and data protection obligations
            - Anti-money laundering (AML) compliance
            - Know Your Customer (KYC) requirements
            - Business conduct and ethics standards
            - Consumer protection measures
        
        FORMATTING REQUIREMENTS:
        - Use detailed sub-bullet points for each main category
        - Include specific dates, percentages, and quantitative requirements
        - Highlight critical deadlines with [URGENT - DATE] tags
        - Use [MANDATORY] tags for non-negotiable requirements
        - Include regulation section references in brackets [Sec. X.Y.Z]
        - Emphasize new or changed requirements with [NEW] or [AMENDED] tags
        - Each bullet point should be comprehensive and actionable
        - Include cross-references to related requirements
        
        Ensure every compliance aspect requiring immediate attention, ongoing monitoring, or future action is captured in detail.
        """
    )
    
    # Create comprehensive LLM chain
    summarization_chain = LLMChain(
        llm=llm,
        prompt=summarization_prompt,
        output_key="summary"
    )
    
    return summarization_chain

def summarize_circular_documents(uploaded_files, api_key):
    """Summarize PDF circulars using detailed compliance analysis and generate a consolidated PDF summary."""
    # Validate API key
    if not api_key.startswith("sk-"):
        st.error("Invalid API key format. OpenAI API keys should start with 'sk-'.")
        return None

    # Initialize LLM with enhanced parameters for detailed analysis
    llm = ChatOpenAI(
        openai_api_key=api_key,
        model_name="gpt-4o-2024-08-06",
        temperature=0.1,  # Lower temperature for more consistent detailed output
        top_p=0.1,        # Lower top_p for more focused responses
        max_tokens=4000   # Increased token limit for detailed summaries
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

    # Create enhanced custom styles for detailed output
    styles.add(ParagraphStyle(
        name='MainTitle',
        parent=styles['Title'],
        fontSize=18,
        textColor='navy',
        alignment=1,  # Center alignment
        spaceAfter=15,
        fontName='Helvetica-Bold'
    ))

    styles.add(ParagraphStyle(
        name='DocumentName',
        parent=styles['Heading2'],
        textColor='darkblue',
        fontSize=14,
        spaceAfter=8,
        fontName='Helvetica-Bold'
    ))

    styles.add(ParagraphStyle(
        name='SectionHeader',
        parent=styles['Heading3'],
        textColor='darkgreen',
        fontSize=12,
        spaceAfter=6,
        fontName='Helvetica-Bold'
    ))

    # Enhanced bullet point styles
    styles.add(ParagraphStyle(
        name='MainBulletPoint',
        parent=styles['BodyText'],
        firstLineIndent=-20,
        leftIndent=20,
        spaceBefore=4,
        spaceAfter=4,
        bulletIndent=0,
        fontSize=10,
        fontName='Helvetica-Bold'
    ))

    styles.add(ParagraphStyle(
        name='SubBulletPoint',
        parent=styles['BodyText'],
        firstLineIndent=-16,
        leftIndent=36,
        spaceBefore=2,
        spaceAfter=2,
        bulletIndent=0,
        fontSize=9
    ))

    styles.add(ParagraphStyle(
        name='DetailBulletPoint',
        parent=styles['BodyText'],
        firstLineIndent=-12,
        leftIndent=48,
        spaceBefore=1,
        spaceAfter=1,
        bulletIndent=0,
        fontSize=8
    ))

    flowables = []

    # Add Enhanced Consolidated Overview Summary header
    flowables.append(Paragraph("COMPREHENSIVE COMPLIANCE ANALYSIS REPORT", styles['MainTitle']))
    flowables.append(Paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y at %H:%M')}", styles['BodyText']))
    flowables.append(Spacer(1, 15))

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
            doc_name = os.path.splitext(uploaded_file.name)[0]
            
            # Enhanced chunking for better detailed analysis
            if len(filtered_text) > 12000:  # Reduced threshold for better detail
                # Chunk the text and process each chunk
                text_chunks = chunk_document(filtered_text, chunk_size=10000, chunk_overlap=1500)
                
                chunk_summaries = []
                for i, chunk in enumerate(text_chunks):
                    try:
                        # Process each chunk with enhanced summarization chain
                        result = summarization_chain.invoke({
                            "text": chunk,
                            "document_name": f"{doc_name} (Section {i+1})"
                        })
                        chunk_summaries.append(result["summary"])
                    except Exception as e:
                        st.warning(f"Error processing section {i+1} of {doc_name}: {str(e)}")
                        continue
                
                # Combine chunk summaries with enhanced consolidation
                if chunk_summaries:
                    combined_text = "\n\n".join(chunk_summaries)
                    
                    # Enhanced consolidation prompt
                    consolidation_prompt = f"""
                    Consolidate these detailed compliance summaries into one comprehensive analysis.
                    Merge similar sections, eliminate redundancy, and ensure all specific requirements, 
                    deadlines, and quantitative details are preserved:
                    
                    {combined_text}
                    
                    Maintain the detailed structure and include ALL compliance requirements.
                    """
                    
                    final_result = summarization_chain.invoke({
                        "text": consolidation_prompt,
                        "document_name": doc_name
                    })
                    summary = final_result["summary"]
                else:
                    summary = f"**[ERR]**\n\n[Processing Error:]\n- Document: {doc_name}\n- Status: Unable to process due to technical errors"
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
                    summary = f"**[ERR]**\n\n[Processing Error:]\n- Document: {doc_name}\n- Error: {str(e)}"

            # Standardize the summary format
            summary = standardize_summary_format(summary, doc_name)

            # Enhanced formatting for PDF with detailed structure
            summary_lines = summary.split('\n')
            
            # Process the enhanced summary format
            for i, line in enumerate(summary_lines):
                line = line.strip()
                
                if line.startswith('**[') and line.endswith(']**'):
                    # Add document abbreviation as title
                    doc_abbrev = line.replace('**[', '').replace(']**', '')
                    flowables.append(Paragraph(f"DOCUMENT: {doc_name} [{doc_abbrev}]", styles['DocumentName']))
                    flowables.append(Spacer(1, 8))
                    continue
                
                if line.startswith("[Detailed Compliance Summary:]"):
                    flowables.append(Paragraph("DETAILED COMPLIANCE ANALYSIS", styles['SectionHeader']))
                    flowables.append(Spacer(1, 6))
                    continue
                
                # Enhanced bullet point processing
                if line.startswith('- **') and line.endswith('**'):
                    # Main section header
                    section_title = line[4:-2]  # Remove '- **' and '**'
                    flowables.append(Paragraph(section_title, styles['SectionHeader']))
                elif line.startswith('- ') and line.strip():
                    # Main bullet point
                    bullet_text = line[2:].strip()
                    
                    # Check for special tags
                    if '[MANDATORY]' in bullet_text:
                        bullet_text = bullet_text.replace('[MANDATORY]', '<b>[MANDATORY]</b>')
                    if '[URGENT' in bullet_text:
                        bullet_text = re.sub(r'\[URGENT[^\]]*\]', lambda m: f'<b><font color="red">{m.group()}</font></b>', bullet_text)
                    if '[NEW]' in bullet_text:
                        bullet_text = bullet_text.replace('[NEW]', '<b><font color="green">[NEW]</font></b>')
                    
                    flowables.append(Paragraph(f"• {bullet_text}", styles['MainBulletPoint']))
                elif line.startswith('    - ') or line.startswith('        - '):
                    # Sub bullet point (indented)
                    indent_level = (len(line) - len(line.lstrip())) // 4
                    sub_bullet_text = line.strip()[2:].strip()
                    
                    # Apply special formatting
                    if '[MANDATORY]' in sub_bullet_text:
                        sub_bullet_text = sub_bullet_text.replace('[MANDATORY]', '<b>[MANDATORY]</b>')
                    if '[URGENT' in sub_bullet_text:
                        sub_bullet_text = re.sub(r'\[URGENT[^\]]*\]', lambda m: f'<b><font color="red">{m.group()}</font></b>', sub_bullet_text)
                    
                    if indent_level <= 1:
                        flowables.append(Paragraph(f"    ◦ {sub_bullet_text}", styles['SubBulletPoint']))
                    else:
                        flowables.append(Paragraph(f"        ▪ {sub_bullet_text}", styles['DetailBulletPoint']))
                elif line.strip() and not line.startswith('**[') and not line.startswith('['):
                    # Regular text with enhanced formatting
                    formatted_line = line.strip()
                    if '[MANDATORY]' in formatted_line:
                        formatted_line = formatted_line.replace('[MANDATORY]', '<b>[MANDATORY]</b>')
                    flowables.append(Paragraph(formatted_line, styles['BodyText']))
            
            flowables.append(Spacer(1, 20))  # Larger space between documents

    # Build and save PDF
    doc.build(flowables)
    pdf_output.seek(0)

    return pdf_output.getvalue()

def main():
    st.set_page_config(page_title="Enhanced PDF Compliance Analyzer", page_icon="⚖️", layout="wide")
    
    st.title("⚖️ Enhanced PDF Compliance Document Analyzer")
    st.subheader("Comprehensive Regulatory Compliance Analysis Tool")
    
    # Enhanced sidebar
    st.sidebar.header("🔧 Configuration")
    openai_api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password", help="Your OpenAI API key for GPT-4 analysis")
    
    # Analysis parameters
    st.sidebar.subheader("Analysis Parameters")
    analysis_depth = st.sidebar.selectbox(
        "Analysis Depth",
        ["Comprehensive", "Standard", "Quick"],
        index=0,
        help="Comprehensive provides the most detailed compliance analysis"
    )
    
    # File uploader with enhanced description
    st.markdown("### 📄 Upload Regulatory Documents")
    st.markdown("Upload PDF circulars, regulations, or compliance documents for detailed analysis.")
    
    uploaded_files = st.file_uploader(
        "Select PDF Files", 
        type=['pdf'], 
        accept_multiple_files=True,
        help="Multiple PDF files can be uploaded for batch processing"
    )
    
    # Display file information
    if uploaded_files:
        st.markdown("### 📋 Selected Files")
        for i, file in enumerate(uploaded_files, 1):
            st.write(f"{i}. {file.name} ({file.size:,} bytes)")
    
    # Enhanced analysis button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔍 Generate Detailed Compliance Analysis", type="primary", use_container_width=True):
            if not openai_api_key:
                st.error("⚠️ Please enter your OpenAI API Key in the sidebar")
                return
            
            if not uploaded_files:
                st.error("⚠️ Please upload at least one PDF file")
                return
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("🔄 Initializing analysis engine...")
                progress_bar.progress(10)
                
                status_text.text("📖 Processing documents and extracting text...")
                progress_bar.progress(30)
                
                status_text.text("🤖 Generating detailed compliance analysis...")
                progress_bar.progress(50)
                
                # Generate enhanced PDF summary
                summary_pdf = summarize_circular_documents(uploaded_files, openai_api_key)
                
                progress_bar.progress(90)
                status_text.text("📝 Finalizing comprehensive report...")
                
                if summary_pdf:
                    progress_bar.progress(100)
                    status_text.text("✅ Analysis complete!")
                    
                    # Enhanced download section
                    st.markdown("### 📥 Download Results")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            label="📊 Download Detailed Compliance Report",
                            data=summary_pdf,
                            file_name=f"detailed_compliance_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf",
                            type="primary",
                            use_container_width=True
                        )
                    
                    with col2:
                        st.info(f"📋 Report generated for {len(uploaded_files)} document(s)")
                    
                    st.success("🎉 Comprehensive compliance analysis completed successfully!")
                    
                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()
            
            except Exception as e:
                st.error(f"❌ An error occurred during analysis: {str(e)}")
                progress_bar.empty()
                status_text.empty()

    # Enhanced about section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📖 How to Use")
    st.sidebar.markdown("""
    **Step 1:** Enter your OpenAI API Key
    **Step 2:** Upload PDF regulatory documents
    **Step 3:** Click 'Generate Analysis'
    **Step 4:** Download comprehensive report
    """)
    
    st.sidebar.markdown("### ✨ Enhanced Features")
    st.sidebar.info("""
    🔍 **Comprehensive Analysis**
    - Detailed compliance requirements
    - Regulatory deadlines & timelines
    - Quantitative thresholds
    - Implementation action items
    
    📊 **Advanced Formatting**
    - Structured section organization
    - Priority tagging system
    - Cross-referenced requirements
    - Professional PDF output
    
    ⚖️ **Compliance Focus**
    - Board governance details
    - Risk management frameworks
    - Audit & assurance requirements
    - Penalty & enforcement provisions
    """)
    
    st.sidebar.warning("**Note:** Requires OpenAI API key with GPT-4 access")

if __name__ == "__main__":
    main()
