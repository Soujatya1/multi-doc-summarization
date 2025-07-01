import streamlit as st
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
import tempfile
import os

# Streamlit UI
st.title("📝 PDF Summarizer to Bullet Point PDF")

# Input OpenAI API Key
openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")

uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

if uploaded_file and openai_api_key:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    # Load PDF as documents
    loader = PyPDFLoader(tmp_file_path)
    documents = loader.load()

    # Define Prompt Template
    prompt = PromptTemplate(
        template="""
        You are a legal assistant. Summarize the following PDF text into clear and concise bullet points suitable for a regulatory report:

        {context}

        Bullet Point Summary:
        """,
        input_variables=["context"]
    )

    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4", temperature=0, api_key=openai_api_key)

    # Create chain
    chain = create_stuff_documents_chain(llm=llm, prompt=prompt)

    # Combine all page content into a single string
    full_context = "\n\n".join(doc.page_content for doc in documents)

    # Generate bullet points
    summary_text = chain.invoke({"context": full_context})

    # Parse bullet points for PDF writing
    bullet_points = summary_text.split("\n")
    bullet_points = [bp.strip("-• ") for bp in bullet_points if bp.strip()]

    # Generate PDF with ReportLab
    output_path = os.path.join(tempfile.gettempdir(), "summary_output.pdf")
    doc = SimpleDocTemplate(output_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = [Paragraph("Summary (Bullet Points)", styles['Title']), Spacer(1, 0.2 * inch)]

    for bullet in bullet_points:
        story.append(Paragraph(f"• {bullet}", styles['Normal']))
        story.append(Spacer(1, 0.1 * inch))

    doc.build(story)

    with open(output_path, "rb") as f:
        st.download_button("Download Summary PDF", f, file_name="summary_output.pdf")

elif uploaded_file:
    st.warning("Please enter your OpenAI API key to proceed.")
