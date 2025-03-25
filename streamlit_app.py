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
import langdetect

st.title("Circulars Summary Generator")

api_key = st.text_input("Enter your OpenAI API Key", type="password")
st.caption("Your API key should start with 'sk-' and will not be stored")

uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

summarize_button = st.button("Summarize")

def is_english(text, min_confidence=0.7):
    """
    Check if the text is primarily in English
    
    Args:
        text (str): Input text to detect language
        min_confidence (float): Minimum confidence threshold for English detection
    
    Returns:
        bool: True if text is in English, False otherwise
    """
    try:
        # Detect language of the first 1000 characters to reduce processing time
        detected_lang = langdetect.detect(text[:1000])
        return detected_lang == 'en'
    except Exception:
        # If language detection fails, fallback to English
        return True

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
                non_english_files = []
                
                for uploaded_file in uploaded_files:
                    file_progress = st.empty()
                    file_progress.text(f"Processing {uploaded_file.name}...")
                    
                    pdf = PdfReader(uploaded_file)
                    text = ''
                    for page in pdf.pages:
                        content = page.extract_text()
                        if content:
                            text += content + "\n"
                    
                    # Check if the document is in English
                    if not is_english(text):
                        non_english_files.append(uploaded_file.name)
                        file_progress.text(f"Skipped {uploaded_file.name} - Not in English")
                        continue
                    
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
                
                # Display non-English files warning
                if non_english_files:
                    st.warning(f"The following files were skipped as they are not in English: {', '.join(non_english_files)}")
                
                if all_summaries:
                    st.write("### Consolidated Overview Summary")
                    
                    heading = doc.add_paragraph()
                    heading_run = heading.add_run("Consolidated Overview Summary")
                    heading_run.bold = True
                    heading_run.font.size = Pt(16)
                    
                    for summary in all_summaries:
                        lines = summary.strip().split('\n')
                        doc_name = lines[0] if lines else "Document"
                        
                        doc_heading = doc.add_paragraph()
                        heading_run = doc_heading.add_run(doc_name.replace("*", ""))
                        heading_run.bold = True
                        heading_run.font.size = Pt(14)
                        
                        key_pointers_match = re.search(r'Key Pointers:', summary)
                        if key_pointers_match:
                            pointers_heading = doc.add_paragraph()
                            pointers_run = pointers_heading.add_run("Key Pointers:")
                            pointers_run.bold = True
                            pointers_run.font.size = Pt(12)
                            pointers_start = key_pointers_match.end()
                            pointers_content = summary[pointers_start:].strip()
                            bullet_pattern = r'(?m)^(?:\d+\.\s+|\•\s+|\-\s+|\*\s+)(.+)$'
                            bullet_points = re.findall(bullet_pattern, pointers_content)
                            
                            if bullet_points:
                                for idx, point in enumerate(bullet_points):
                                    if idx == 0 and point.strip() == "*":
                                        continue
                                    bullet_para = doc.add_paragraph()
                                    bullet_para.style = 'List Bullet'
                                    bullet_para.add_run(point.strip()).font.size = Pt(11)
                            else:
                                for line in pointers_content.split('\n'):
                                    if line.strip():
                                        para = doc.add_paragraph()
                                        para.add_run(line.strip()).font.size = Pt(11)
                        else:
                            remaining_content = '\n'.join(lines[1:]) if len(lines) > 1 else ""
                            if remaining_content:
                                content_para = doc.add_paragraph()
                                content_para.add_run(remaining_content).font.size = Pt(11)
                        
                        doc.add_paragraph()
                        
                        st.write(summary)
                        st.write("---")
                
            doc_output_path = "circulars_consolidated_summary.docx"
            doc.save(doc_output_path)
            
            with open(doc_output_path, "rb") as doc_file:
                st.download_button(
                    "Download Summaries DOCX",
                    doc_file,
                    file_name="circulars_consolidated_summary.docx",
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
