import streamlit as st
from PyPDF2 import PdfReader
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from docx import Document as DocxDocument
from docx.shared import Pt

st.title("Multi-Document Summary Generator")
uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    llm = ChatOpenAI(
        openai_api_key="sk-t4V6IkGaaXbQw4667VHAT3B1bkFJYh2J4ZAgECM0LCY7vvdC",
        model_name="gpt-4",
        temperature=0.1,
        top_p=0.2
    )
    
    map_prompt = PromptTemplate(
        input_variables=["text"],
        template="Read and summarize the following content in your own words, highlighting the main ideas, purpose, and important insights without including direct phrases or sentences from the original text.\n\n{text}"
    )
    
    combine_prompt = PromptTemplate(
        input_variables=["text"],
        template="Combine the following individual summaries into a cohesive, insightful summary. Ensure that it is concise, capturing the core themes and purpose of the entire document in 10 lines or less:\n\n{text}"
    )
    
    doc = DocxDocument()
    for uploaded_file in uploaded_files:
        pdf = PdfReader(uploaded_file)
        # Extract text
        text = ''
        for page in pdf.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
                
        if text.strip():
            docs = [Document(page_content=text)]
            map_reduce_chain = load_summarize_chain(
                llm,
                chain_type="map_reduce",
                map_prompt=map_prompt,
                combine_prompt=combine_prompt
            )
            
            output_summary = map_reduce_chain.invoke(docs)
            summary = output_summary['output_text']
            
            # Display in UI
            st.write(f"### Summary for {uploaded_file.name}:")
            st.write(summary)
            
            # Save to DOCX
            doc.add_paragraph(uploaded_file.name, style='Heading 1')
            para = doc.add_paragraph()
            para.add_run(summary).font.size = Pt(11)
            
    doc_output_path = "multi_doc_summary.docx"
    doc.save(doc_output_path)
    
    with open(doc_output_path, "rb") as doc_file:
        st.download_button(
            "Download Summaries DOCX",
            doc_file,
            file_name="multi_doc_summary.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
else:
    st.write("Please upload one or more PDF files.")
