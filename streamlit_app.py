import streamlit as st
from PyPDF2 import PdfReader
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
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

def extract_english_text(text):
    """
    Extract only English words from the given text
    
    Args:
        text (str): Input text to filter
    
    Returns:
        str: Text containing only English words
    """
    try:
        # Use regex to split text into words
        words = re.findall(r'\b\w+\b', text)
        
        # Filter words and keep only those detected as English
        english_words = []
        for word in words:
            try:
                # Only consider words longer than 1 character
                if len(word) > 1:
                    lang = langdetect.detect(word)
                    if lang == 'en':
                        english_words.append(word)
            except langdetect.lang_detect_exception.LangDetectException:
                continue
        
        # Reconstruct text with only English words
        return ' '.join(english_words)
    
    except Exception as e:
        st.warning(f"Language filtering error: {e}")
        return text  # Fallback to original text if filtering fails

def chunk_document(text, chunk_size=4000, chunk_overlap=500):
    """
    Split document into manageable chunks
    
    Args:
        text (str): Input text to chunk
        chunk_size (int): Maximum number of characters per chunk
        chunk_overlap (int): Number of characters to overlap between chunks
    
    Returns:
        list: List of text chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    return text_splitter.split_text(text)

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
                template="""Read and summarize the following content chunk in your own words, highlighting the main ideas, purpose, and important insights without including direct phrases or sentences from the original text in 10 bullet points.\n\n{text}
                """
            )
            
            combine_prompt = PromptTemplate(
                input_variables=["text"],
                template="""Combine the following individual chunk summaries into a cohesive, insightful summary for the entire document. Ensure that it captures the core themes and purpose across all chunks in 15 bullet points:
                
                Key Themes and Insights:
                \n\n{text}
                """
            )
            
            with st.spinner("Processing documents..."):
                all_document_summaries = []
                partially_filtered_files = []
                
                for uploaded_file in uploaded_files:
                    file_progress = st.empty()
                    file_progress.text(f"Processing {uploaded_file.name}...")
                    
                    pdf = PdfReader(uploaded_file)
                    text = ''
                    for page in pdf.pages:
                        content = page.extract_text()
                        if content:
                            text += content + "\n"
                    
                    # Extract only English text
                    filtered_text = extract_english_text(text)
                    
                    # Check if any meaningful text remains after filtering
                    if not filtered_text.strip():
                        st.warning(f"No English text found in {uploaded_file.name}")
                        continue
                    
                    # Track if text was partially filtered
                    if len(filtered_text) < len(text) * 0.5:
                        partially_filtered_files.append(uploaded_file.name)
                    
                    # Chunk the filtered text
                    text_chunks = chunk_document(filtered_text)
                    
                    # Convert chunks to Langchain Documents
                    docs = [Document(page_content=chunk) for chunk in text_chunks]
                    
                    # Process chunks with map-reduce summarization
                    map_reduce_chain = load_summarize_chain(
                        llm,
                        chain_type="map_reduce",
                        map_prompt=map_prompt,
                        combine_prompt=combine_prompt
                    )
                    
                    output_summary = map_reduce_chain.invoke(docs)
                    summary = output_summary['output_text']
                    
                    # Add document name to summary
                    full_summary = f"Summary of {uploaded_file.name}:\n\n{summary}"
                    
                    all_document_summaries.append(full_summary)
                    st.write(full_summary)
                    
                    file_progress.text(f"Completed {uploaded_file.name}")
                
                # Display partially filtered files warning
                if partially_filtered_files:
                    st.warning(f"The following files had significant non-English content and were partially filtered: {', '.join(partially_filtered_files)}")
                
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
