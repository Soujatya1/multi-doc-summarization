import streamlit as st
import tempfile
import os
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.llms import OpenAI
from langchain.schema import Document
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Flowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from datetime import datetime
import re

class BulletPointFlowable(Flowable):
    """Custom flowable for bullet points"""
    def __init__(self, text, bullet_char="•", indent=20):
        Flowable.__init__(self)
        self.text = text
        self.bullet_char = bullet_char
        self.indent = indent
        self.width = letter[0] - 2*inch
        self.height = 20

    def draw(self):
        canvas = self.canv
        canvas.setFont("Helvetica", 11)
        canvas.drawString(self.indent, 0, f"{self.bullet_char} {self.text}")

def load_document(uploaded_file):
    """Load document from uploaded file"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        # Load document based on file type
        if uploaded_file.name.endswith('.pdf'):
            loader = PyPDFLoader(tmp_file_path)
            documents = loader.load()
        elif uploaded_file.name.endswith('.txt'):
            loader = TextLoader(tmp_file_path)
            documents = loader.load()
        else:
            # For other text files, read as text
            with open(tmp_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            documents = [Document(page_content=content)]
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return documents
    except Exception as e:
        st.error(f"Error loading document: {str(e)}")
        return None

def create_summarization_chain(api_key, chain_type="stuff", min_bullets=6):
    """Create LangChain summarization chain"""
    try:
        llm = OpenAI(
            temperature=0.3,
            openai_api_key=api_key,
            max_tokens=1500
        )
        
        # Enhanced prompt for multiple bullet-point summaries
        from langchain.prompts import PromptTemplate
        
        prompt_template = f"""
        Please provide a comprehensive summary of the following text in multiple bullet points format.
        Create AT LEAST {min_bullets} bullet points covering different aspects of the content.
        Each bullet point should capture a distinct key idea, main concept, or important detail.
        
        Guidelines:
        - Start each bullet point with "•"
        - Make each point concise but informative (1-2 sentences max)
        - Cover different themes/topics from the text
        - Include important facts, figures, or conclusions
        - Organize points logically from most to least important
        - Ensure you provide at least {min_bullets} distinct bullet points
        
        Text to summarize:
        {{text}}
        
        SUMMARY IN MULTIPLE BULLET POINTS:
        """
        
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
        
        chain = load_summarize_chain(
            llm=llm,
            chain_type=chain_type,
            prompt=PROMPT,
            verbose=True
        )
        
        return chain
    except Exception as e:
        st.error(f"Error creating summarization chain: {str(e)}")
        return None

def extract_bullet_points(summary_text):
    """Extract and clean bullet points from the summary text"""
    lines = summary_text.split('\n')
    bullet_points = []
    
    for line in lines:
        line = line.strip()
        if line:
            # Check if line already has bullet point markers
            if line.startswith('•') or line.startswith('-') or line.startswith('*'):
                # Remove existing bullet and clean
                cleaned_line = re.sub(r'^[•\-\*]\s*', '', line).strip()
            elif re.match(r'^\d+\.', line):
                # Handle numbered lists
                cleaned_line = re.sub(r'^\d+\.\s*', '', line).strip()
            else:
                # Regular line - might be a bullet point without marker
                cleaned_line = line.strip()
            
            # Only add non-empty, substantial lines (avoid headers/fluff)
            if cleaned_line and len(cleaned_line) > 10:
                bullet_points.append(cleaned_line)
    
    # If we don't have enough bullet points, try to split long paragraphs
    if len(bullet_points) < 3:
        # Split by sentences for additional bullet points
        for line in lines:
            if len(line.strip()) > 50:
                sentences = re.split(r'[.!?]+', line)
                for sentence in sentences:
                    sentence = sentence.strip()
                    if sentence and len(sentence) > 15:
                        # Remove bullet markers if present
                        sentence = re.sub(r'^[•\-\*]\s*', '', sentence).strip()
                        if sentence not in bullet_points:
                            bullet_points.append(sentence)
    
    # Ensure we have at least some bullet points
    if not bullet_points and summary_text.strip():
        # Fallback: split the entire summary into sentences
        sentences = re.split(r'[.!?]+', summary_text)
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 15:
                bullet_points.append(sentence)
    
    return bullet_points[:15]  # Limit to 15 bullet points max

def create_pdf_report(summary_text, original_filename, output_filename):
    """Create PDF report with bullet points"""
    doc = SimpleDocTemplate(
        output_filename,
        pagesize=A4,
        rightMargin=inch,
        leftMargin=inch,
        topMargin=inch,
        bottomMargin=inch
    )
    
    # Get styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=30,
        textColor=HexColor('#2E4057'),
        alignment=1  # Center alignment
    )
    
    header_style = ParagraphStyle(
        'CustomHeader',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=12,
        textColor=HexColor('#2E4057')
    )
    
    bullet_style = ParagraphStyle(
        'BulletStyle',
        parent=styles['Normal'],
        fontSize=11,
        leftIndent=20,
        spaceAfter=8,
        bulletIndent=10,
        bulletFontName='Helvetica',
        bulletText='•'
    )
    
    # Build PDF content
    story = []
    
    # Title
    story.append(Paragraph("Document Summary Report", title_style))
    story.append(Spacer(1, 12))
    
    # Document info
    story.append(Paragraph(f"<b>Original Document:</b> {original_filename}", styles['Normal']))
    story.append(Paragraph(f"<b>Generated on:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Summary header
    story.append(Paragraph("Summary", header_style))
    story.append(Spacer(1, 12))
    
    # Extract and add bullet points
    bullet_points = extract_bullet_points(summary_text)
    
    if bullet_points:
        for point in bullet_points:
            if point.strip():
                bullet_para = Paragraph(f"• {point}", bullet_style)
                story.append(bullet_para)
    else:
        # Fallback: add raw summary if no bullet points found
        story.append(Paragraph(summary_text, styles['Normal']))
    
    # Build PDF
    doc.build(story)
    return output_filename

def main():
    st.set_page_config(
        page_title="Document Summarizer",
        page_icon="📄",
        layout="wide"
    )
    
    st.title("📄 Document Summarizer with PDF Export")
    st.markdown("Upload a document to get an AI-powered summary in bullet points, then download as PDF")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # OpenAI API Key input
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key to use the summarization service"
        )
        
        # Chain type selection
        chain_type = st.selectbox(
            "Summarization Method",
            ["stuff", "map_reduce", "refine"],
            help="Choose the summarization chain type"
        )
        
        # Chunk size for large documents
        chunk_size = st.slider(
            "Chunk Size",
            min_value=500,
            max_value=4000,
            value=1000,
            help="Size of text chunks for processing large documents"
        )
        
        # Number of bullet points setting
        min_bullet_points = st.slider(
            "Minimum Bullet Points",
            min_value=3,
            max_value=15,
            value=6,
            help="Minimum number of bullet points to generate"
        )
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Upload Document")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['pdf', 'txt', 'docx'],
            help="Upload PDF, TXT, or DOCX files"
        )
        
        if uploaded_file is not None:
            st.success(f"File uploaded: {uploaded_file.name}")
            
            # Display file details
            file_details = {
                "Filename": uploaded_file.name,
                "File size": f"{uploaded_file.size} bytes",
                "File type": uploaded_file.type
            }
            st.json(file_details)
    
    with col2:
        st.header("Summary Options")
        
        # Summary button
        if st.button("Generate Summary", type="primary", disabled=not (uploaded_file and api_key)):
            if not api_key:
                st.error("Please enter your OpenAI API key")
            elif not uploaded_file:
                st.error("Please upload a document")
            else:
                with st.spinner("Processing document..."):
                    # Load document
                    documents = load_document(uploaded_file)
                    
                    if documents:
                        # Split documents if they're too large
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=chunk_size,
                            chunk_overlap=200
                        )
                        
                        if len(documents) > 1 or len(documents[0].page_content) > chunk_size:
                            documents = text_splitter.split_documents(documents)
                        
                        # Create summarization chain
                        chain = create_summarization_chain(api_key, chain_type, min_bullet_points)
                        
                        if chain:
                            try:
                                # Generate summary
                                summary = chain.run(documents)
                                
                                # Store summary in session state
                                st.session_state.summary = summary
                                st.session_state.filename = uploaded_file.name
                                
                                st.success("Summary generated successfully!")
                                
                            except Exception as e:
                                st.error(f"Error generating summary: {str(e)}")
    
    # Display summary if available
    if hasattr(st.session_state, 'summary'):
        st.header("Generated Summary")
        
        # Extract and display bullet points
        bullet_points = extract_bullet_points(st.session_state.summary)
        
        st.markdown(f"### Summary with {len(bullet_points)} Bullet Points:")
        
        # Display summary in bullet format
        summary_container = st.container()
        with summary_container:
            if bullet_points:
                for i, point in enumerate(bullet_points, 1):
                    st.markdown(f"**{i}.** • {point}")
            else:
                st.markdown(st.session_state.summary)
        
        # Show raw summary in expander for reference
        with st.expander("View Raw Summary"):
            st.text(st.session_state.summary)
        
        # PDF export section
        st.header("Export to PDF")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            pdf_filename = st.text_input(
                "PDF Filename",
                value=f"summary_{st.session_state.filename.split('.')[0]}.pdf",
                help="Enter the desired PDF filename"
            )
        
        with col2:
            if st.button("Generate PDF", type="secondary"):
                try:
                    # Create temporary PDF file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_pdf:
                        pdf_path = tmp_pdf.name
                    
                    # Generate PDF
                    create_pdf_report(
                        st.session_state.summary,
                        st.session_state.filename,
                        pdf_path
                    )
                    
                    # Read PDF file for download
                    with open(pdf_path, 'rb') as pdf_file:
                        pdf_data = pdf_file.read()
                    
                    # Clean up temporary file
                    os.unlink(pdf_path)
                    
                    # Download button
                    st.download_button(
                        label="📥 Download PDF",
                        data=pdf_data,
                        file_name=pdf_filename,
                        mime="application/pdf"
                    )
                    
                    st.success("PDF generated successfully!")
                    
                except Exception as e:
                    st.error(f"Error generating PDF: {str(e)}")
    
    # Instructions
    with st.expander("How to use"):
        st.markdown("""
        1. **Enter your OpenAI API Key** in the sidebar
        2. **Upload a document** (PDF, TXT, or DOCX)
        3. **Configure settings** in the sidebar if needed
        4. **Click "Generate Summary"** to create bullet-point summary
        5. **Generate PDF** to download the summary as a formatted PDF
        
        **Chain Types:**
        - **Stuff**: Best for smaller documents, processes all text at once
        - **Map Reduce**: Good for larger documents, summarizes in chunks then combines
        - **Refine**: Iteratively refines summary, best quality but slower
        """)

if __name__ == "__main__":
    main()
