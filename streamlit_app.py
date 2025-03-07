import streamlit as st
from PyPDF2 import PdfReader
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from docx import Document as DocxDocument
from docx.shared import Pt, RGBColor
import os

# Set title
st.title("Legal Case Summary Generator")

# API key input
api_key = st.text_input("Enter your OpenAI API Key", type="password")
st.caption("Your API key should start with 'sk-' and will not be stored")

# File uploader
uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

# Add summarize button
summarize_button = st.button("Summarize")

if summarize_button and uploaded_files and api_key:
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
            
            # Updated prompt template to extract structured information
            map_prompt = PromptTemplate(
                input_variables=["text"],
                template="""Analyze the following legal case document and extract:
                
                1. The case name and citation
                2. Parties Involved (names of complainant/appellant and respondent)
                3. Key Events (summary of what happened in the case)
                4. Key Findings (important rulings or conclusions)
                
                Format your response exactly as:
                
                **[Case Name and Citation]**
                · **Parties Involved:** [Names]
                · **Key Events:** [Summary of events]
                · **Key Findings:** [Summary of findings]
                
                Here is the text to analyze:
                
                {text}
                """
            )
            
            # Updated combine prompt to create the consolidated overview
            combine_prompt = PromptTemplate(
                input_variables=["text"],
                template="""Create a consolidated overview of the following legal case summaries. 
                
                Start with the heading "**Consolidated Overview Summary**" and then list each case summary in order.
                Keep the exact formatting from the individual summaries, using bullet points with the "·" character.
                
                {text}
                """
            )
            
            doc = DocxDocument()
            
            with st.spinner("Processing documents..."):
                all_summaries = []
                
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
                        all_summaries.append(summary)
                        
                        st.write(summary)
                        
                    file_progress.text(f"Completed {uploaded_file.name}")
                
                # Create a consolidated summary
                if all_summaries:
                    st.write("### Consolidated Overview Summary")
                    
                    # Add to Word document with formatting
                    heading = doc.add_paragraph()
                    run = heading.add_run("Consolidated Overview Summary")
                    run.bold = True
                    run.font.size = Pt(14)
                    
                    for summary in all_summaries:
                        doc.add_paragraph(summary)
                        st.write(summary)
                        st.write("---")
                
            # Save Word document
            doc_output_path = "legal_case_summaries.docx"
            doc.save(doc_output_path)
            
            with open(doc_output_path, "rb") as doc_file:
                st.download_button(
                    "Download Summaries DOCX",
                    doc_file,
                    file_name="legal_case_summaries.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("Please check that your API key is correct and that you have access to the selected model.")
elif summarize_button and (not uploaded_files or not api_key):
    if not uploaded_files:
        st.warning("Please upload PDF files before summarizing.")
    if not api_key:
        st.warning("Please enter your OpenAI API key before summarizing.")
else:
    st.info("Upload PDF files, enter your OpenAI API key, and click 'Summarize' to process documents.")
