import os
import tempfile
import streamlit as st
from PyPDF2 import PdfReader
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


# Function to extract text from PDF
def extract_text_from_pdf(uploaded_pdf):
    reader = PdfReader(uploaded_pdf)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


# Function to create bullet-point PDF using ReportLab
def create_bullet_pdf(bullets, output_path):
    c = canvas.Canvas(output_path, pagesize=A4)
    width, height = A4
    textobject = c.beginText(50, height - 50)
    textobject.setFont("Helvetica", 12)

    textobject.textLine("Summary:")
    textobject.textLine("")

    for bullet in bullets:
        lines = bullet.strip().split("\n")
        for line in lines:
            if line.strip():
                textobject.textLine(f"• {line.strip()}")
        textobject.textLine("")

    c.drawText(textobject)
    c.save()


# Streamlit UI
st.title("📄 Document Summarizer with PDF Output")

uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])
openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")

if uploaded_pdf and openai_api_key:
    text = extract_text_from_pdf(uploaded_pdf)
    
    # LangChain setup
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    documents = [Document(page_content=chunk) for chunk in chunks]

    prompt_template = PromptTemplate.from_template(
        """You are a legal expert assistant. Summarize the following content into concise bullet points in simple formal English:
        \n\n{text}\n\nSummary in bullet points:"""
    )
    llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4", temperature=0.3)
    summarizer_chain = create_stuff_documents_chain(llm, prompt_template)

    with st.spinner("Generating summary..."):
        response = summarizer_chain.invoke({"input_documents": documents})
        summary_text = response.get("output", "").strip()

    # Convert summary to bullet points
    bullets = [line.strip("-• ") for line in summary_text.split("\n") if line.strip()]
    
    # Save to PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
        pdf_path = tmpfile.name
        create_bullet_pdf(bullets, pdf_path)

    st.success("Summary generated and saved as PDF.")
    with open(pdf_path, "rb") as f:
        st.download_button("📥 Download Summary PDF", f, file_name="summary_output.pdf")
