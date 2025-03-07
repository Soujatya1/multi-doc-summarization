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

# Same Streamlit UI setup...

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
                        
                        # Also add to Word document directly
                        # Extract document name from the file name
                        doc_name = os.path.splitext(uploaded_file.name)[0]
                        doc_heading = doc.add_paragraph()
                        heading_run = doc_heading.add_run(doc_name)
                        heading_run.bold = True
                        heading_run.font.size = Pt(14)
                        
                        # Add the "Key Pointers:" heading
                        key_pointers = doc.add_paragraph()
                        pointers_run = key_pointers.add_run("Key Pointers:")
                        pointers_run.bold = True
                        pointers_run.font.size = Pt(12)
                        
                        # Extract bullet points - adjust regex based on actual format
                        # Try to find bullet points (assuming they start with * or - or •)
                        bullet_points = re.findall(r'(?:•|\*|\-)\s+(.*?)(?=(?:•|\*|\-|$))', summary, re.DOTALL)
                        
                        if bullet_points:
                            for point in bullet_points:
                                bullet_para = doc.add_paragraph()
                                bullet_para.style = 'List Bullet'
                                bullet_para.add_run(point.strip()).font.size = Pt(11)
                        else:
                            # If no bullet points found, just add the raw text
                            text_para = doc.add_paragraph()
                            text_para.add_run(summary.strip()).font.size = Pt(11)
                        
                        # Add a separator
                        doc.add_paragraph()
                        
                    file_progress.text(f"Completed {uploaded_file.name}")
                
                if all_summaries:
                    st.write("### Consolidated Overview Summary")
                    
                    # Add consolidated summary to Word doc
                    heading = doc.add_paragraph()
                    heading_run = heading.add_run("Consolidated Overview Summary")
                    heading_run.bold = True
                    heading_run.font.size = Pt(16)
                    
                    # Combine all summaries for the consolidated section
                    consolidated_text = "\n\n".join(all_summaries)
                    
                    # Add the consolidated text to the document
                    consol_para = doc.add_paragraph()
                    consol_para.add_run(consolidated_text).font.size = Pt(11)
                        
                    st.write("---")
                
            doc_output_path = "circulars.docx"
            doc.save(doc_output_path)
            
            with open(doc_output_path, "rb") as doc_file:
                st.download_button(
                    "Download Summaries DOCX",
                    doc_file,
                    file_name="circulars.docx",
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
