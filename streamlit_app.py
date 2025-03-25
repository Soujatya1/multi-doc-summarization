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

def is_english(text, min_characters=500, min_english_ratio=0.8):
    """
    Comprehensive language detection for English text
    
    Args:
        text (str): Input text to detect language
        min_characters (int): Minimum number of characters to analyze
        min_english_ratio (float): Minimum ratio of English confidence
    
    Returns:
        bool: True if text is predominantly English, False otherwise
    """
    # Remove whitespace and newlines
    clean_text = text.strip()
    
    # Check if text is long enough to analyze
    if len(clean_text) < min_characters:
        return False
    
    try:
        # Attempt to detect language for the whole text
        language_probabilities = {}
        total_length = len(clean_text)
        
        # Analyze text in chunks to get more comprehensive language detection
        chunk_size = 1000
        for i in range(0, total_length, chunk_size):
            chunk = clean_text[i:i+chunk_size]
            try:
                lang = langdetect.detect(chunk)
                language_probabilities[lang] = language_probabilities.get(lang, 0) + len(chunk)
            except langdetect.lang_detect_exception.LangDetectException:
                continue
        
        # Calculate language ratios
        total_analyzed = sum(language_probabilities.values())
        language_ratios = {lang: count/total_analyzed for lang, count in language_probabilities.items()}
        
        # Check if English is the predominant language
        english_ratio = language_ratios.get('en', 0)
        return english_ratio >= min_english_ratio
    
    except Exception as e:
        st.warning(f"Language detection error: {e}")
        return False

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
                
                # Rest of the code remains the same as in the previous version
                # ... (omitted for brevity, same as previous implementation)
                
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
