import streamlit as st
from PyPDF2 import PdfReader
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from docx import Document as DocxDocument
from docx.shared import Pt
import re
import os

st.title("Legal Case Summary Generator")

api_key = st.text_input("Enter your OpenAI API Key", type="password")
st.caption("Your API key should start with 'sk-' and will not be stored")

uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

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
            
            map_prompt = PromptTemplate(
                input_variables=["text"],
                template="""Read and summarize the following content in your own words, highlighting the main ideas, purpose, and important insights without including direct phrases or sentences from the original text in 15 bullet points.\n\n{text}
                """
            )
            
            combine_prompt = PromptTemplate(
                input_variables=["text"],
                template="""Each summary for a document should start with the document name (without extensions like .pdf or .docx).
                Each summary should have a heading named, "Key Pointers:"
                Combine the following individual summaries into a cohesive, insightful summary. Ensure that it is concise, capturing the core themes and purpose of the entire document in 15 bullet points:\n\n{text}
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
                        
                        # Display in Streamlit
                        st.write(summary)
                        
                    file_progress.text(f"Completed {uploaded_file.name}")
                
                if all_summaries:
                    st.write("### Consolidated Overview Summary")
                    
                    heading = doc.add_paragraph()
                    heading_run = heading.add_run("Consolidated Overview Summary")
                    heading_run.bold = True
                    heading_run.font.size = Pt(16)
                    
                    for summary in all_summaries:
                        # Extract document name (first line of the summary)
                        lines = summary.strip().split('\n')
                        doc_name = lines[0] if lines else "Document"
                        
                        # Add document name to Word doc
                        doc_heading = doc.add_paragraph()
                        heading_run = doc_heading.add_run(doc_name)
                        heading_run.bold = True
                        heading_run.font.size = Pt(14)
                        
                        # Find "Key Pointers:" section
                        key_pointers_match = re.search(r'Key Pointers:', summary)
                        if key_pointers_match:
                            # Add "Key Pointers:" heading
                            pointers_heading = doc.add_paragraph()
                            pointers_run = pointers_heading.add_run("Key Pointers:")
                            pointers_run.bold = True
                            pointers_run.font.size = Pt(12)
                            
                            # Extract the content after "Key Pointers:"
                            pointers_start = key_pointers_match.end()
                            pointers_content = summary[pointers_start:].strip()
                            
                            # Try to find bullet points - this regex looks for lines that start with numbers,
                            # asterisks, hyphens, etc., which are common bullet point markers
                            bullet_pattern = r'(?:^|\n)(?:\d+\.|\*|\-|\•)\s*(.*?)(?=(?:\n(?:\d+\.|\*|\-|\•)|\Z))'
                            bullet_points = re.findall(bullet_pattern, pointers_content, re.DOTALL)
                            
                            if bullet_points:
                                # Add each bullet point
                                for point in bullet_points:
                                    bullet_para = doc.add_paragraph()
                                    bullet_para.style = 'List Bullet'
                                    bullet_para.add_run(point.strip()).font.size = Pt(11)
                            else:
                                # If no bullet points found with regex, just add the content as paragraphs
                                for line in pointers_content.split('\n'):
                                    if line.strip():
                                        para = doc.add_paragraph()
                                        para.add_run(line.strip()).font.size = Pt(11)
                        else:
                            # If "Key Pointers:" not found, add all content after the first line
                            remaining_content = '\n'.join(lines[1:]) if len(lines) > 1 else ""
                            if remaining_content:
                                content_para = doc.add_paragraph()
                                content_para.add_run(remaining_content).font.size = Pt(11)
                        
                        # Add a separator
                        doc.add_paragraph()
                        
                        # Display in Streamlit
                        st.write(summary)
                        st.write("---")
                
            doc_output_path = "document_summaries.docx"
            doc.save(doc_output_path)
            
            with open(doc_output_path, "rb") as doc_file:
                st.download_button(
                    "Download Summaries DOCX",
                    doc_file,
                    file_name="document_summaries.docx",
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
