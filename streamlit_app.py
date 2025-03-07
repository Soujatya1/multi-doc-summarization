import streamlit as st
from PyPDF2 import PdfReader
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from docx import Document as DocxDocument
from docx.shared import Pt
import os

# Set title
st.title("Multi-Document Summary Generator")

# API key input
api_key = st.text_input("Enter your OpenAI API Key", type="password")
st.caption("Your API key should start with 'sk-' and will not be stored")

# File uploader
uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files and api_key:
    if not api_key.startswith("sk-"):
        st.error("Invalid API key format. OpenAI API keys should start with 'sk-'")
    else:
        try:
            llm = ChatOpenAI(
                openai_api_key=api_key,
                model_name="gpt-4o-2024-08-06",
                temperature=0.2,
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
            
            with st.spinner("Processing documents..."):
                for uploaded_file in uploaded_files:
                    file_progress = st.empty()
                    file_progress.text(f"Processing {uploaded_file.name}...")
                    
                    pdf = PdfReader(uploaded_file)
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
                        
                        st.write(f"### Summary for {uploaded_file.name}:")
                        st.write(summary)
                        
                        doc.add_paragraph(uploaded_file.name, style='Heading 1')
                        para = doc.add_paragraph()
                        para.add_run(summary).font.size = Pt(11)
                        
                    file_progress.text(f"Completed {uploaded_file.name}")
                
            doc_output_path = "multi_doc_summary.docx"
            doc.save(doc_output_path)
            
            with open(doc_output_path, "rb") as doc_file:
                st.download_button(
                    "Download Summaries DOCX",
                    doc_file,
                    file_name="multi_doc_summary.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("Please check that your API key is correct and that you have access to the selected model.")
elif uploaded_files:
    st.info("Please enter your OpenAI API key to process the documents.")
else:
    st.write("Please upload one or more PDF files and provide your OpenAI API key.")
