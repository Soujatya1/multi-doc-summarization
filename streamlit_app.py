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

def get_summary_prompt(text, page_count):
    return f"""
You are a domain expert in insurance compliance and regulation.
Your task is to generate a **clean, concise, section-wise summary** of the input IRDAI/regulatory document while preserving the **original structure and flow** of the document.
---
### Mandatory Summarization Rules:
1. **Follow the original structure strictly** — maintain the same order of:
   - Section headings
   - Subheadings
   - Bullet points
   - Tables
   - Date-wise event history
   - UIDAI / IRDAI / eGazette circulars
2. **Do NOT rename or reformat section titles** — retain the exact headings from the original file.
3. **Each section should be summarized in 1–5 lines**, proportional to its original length:
   - Keep it brief, but **do not omit the core message**.
   - Avoid generalizations or overly descriptive rewriting.
4. If a section contains **definitions**, summarize them line by line (e.g., Definition A: …).
5. If the section contains **tabular data**, preserve **column-wise details**:
   - Include every row and column in a concise bullet or structured format.
   - Do not merge or generalize rows — maintain data fidelity.
6. If a section contains **violations, fines, or penalties**, mention each item clearly:
   - List out exact violation titles and actions taken or proposed.
7. For **date-wise circulars or history**, ensure that:
   - **No dates are skipped or merged.**
   - Maintain **chronological order**.
   - Mention full references such as "IRDAI Circular dated 12-May-2022".
---
### Output Format:
- Follow the exact **order and structure** of the input file.
- Do **not invent new headings** or sections.
- Avoid decorative formatting, markdown, or unnecessary bolding — use **clean plain text**.
---
### Guideline:
Ensure that the **total summary length does not exceed ~50% of the English content pages** from the input document (total pages: {page_count}).
Now, generate a section-wise structured summary of the document below:
--------------------
{text}
"""

def create_summary_chain(azure_endpoint, api_key, api_version, deployment_name, model_name="gpt-4o"):
    llm = AzureChatOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=api_key,
        api_version=api_version,
        deployment_name=deployment_name,
        temperature=0.1,
        max_tokens=4000
    )
    
    prompt_template = ChatPromptTemplate.from_template("{context}")
    chain = create_stuff_documents_chain(llm, prompt_template)
    
    return chain

def generate_document_summary(doc_chunks, azure_endpoint, api_key, api_version, deployment_name, page_count):
    # Combine all chunks into a single text
    combined_text = "\n\n".join([chunk.page_content for chunk in doc_chunks])
    
    # Get the custom prompt
    custom_prompt = get_summary_prompt(combined_text, page_count)
    
    # Create a single document with the custom prompt as content
    prompt_doc = Document(page_content=custom_prompt)
    
    # Create the chain
    chain = create_summary_chain(azure_endpoint, api_key, api_version, deployment_name)
    
    # Generate summary
    summary = chain.invoke({"context": [prompt_doc]})
    
    return summary

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
        page_title="IRDAI Document Summarizer",
        page_icon="📋",
        layout="wide"
    )
    
    st.title("📋 IRDAI Document Summarizer")
    st.markdown("Generate clean, section-wise summaries of IRDAI regulatory documents!")
    
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
            ["2025-01-01-preview"],
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
        st.header("🚀 Generate Summary")
        
        if st.button("🔄 Generate Summary", type="primary", disabled=not (uploaded_file and api_key)):
            if not api_key:
                st.error("❌ Please enter your OpenAI API key in the sidebar.")
            elif not uploaded_file:
                st.error("❌ Please upload a document first.")
            else:
                with st.spinner("🔍 Analyzing document and generating summary..."):
                    try:
                        documents = load_document(uploaded_file)
                        if documents is None:
                            st.stop()
                        
                        total_content = sum(len(doc.page_content) for doc in documents)
                        page_count = len(documents)
                        st.info(f"📄 Document loaded: {page_count} pages, {total_content:,} characters")
                        
                        doc_chunks = split_documents(documents, chunk_size=1500, chunk_overlap=500)
                        st.info(f"📑 Document processed into {len(doc_chunks)} chunks for analysis")
                        
                        summary = generate_document_summary(
                            doc_chunks, azure_endpoint, api_key, api_version, deployment_name, page_count
                        )
                        
                        st.session_state.summary = summary
                        st.session_state.filename = uploaded_file.name
                        st.session_state.structured_summary = parse_structured_summary(summary)
                        
                        st.success("✅ Summary generated successfully!")
                        st.balloons()
                        
                    except Exception as e:
                        st.error(f"❌ Error generating summary: {str(e)}")
                        st.exception(e)
    
    if hasattr(st.session_state, 'summary') and st.session_state.summary:
        st.header("📋 Generated Summary")
        
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
                with st.spinner("📝 Creating PDF summary..."):
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
                    file_name=f"summary_{st.session_state.filename.split('.')[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )

if __name__ == "__main__":
    main()
