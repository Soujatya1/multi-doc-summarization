import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from io import BytesIO
import re
import langdetect
from datetime import datetime

def create_timestamped_filename(output_folder, base_file_name):
    """Create a timestamped filename for the output PDF."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_name = f"{base_file_name}_{timestamp}.pdf"
    full_path = os.path.join(output_folder, file_name)
    return full_path

def extract_english_text(text):
    """Extract English words from the given text."""
    try:
        words = re.findall(r'\b\w+\b', text)
        
        english_words = []
        for word in words:
            try:
                if len(word) > 1:
                    lang = langdetect.detect(word)
                    if lang == 'en':
                        english_words.append(word)
            except langdetect.lang_detect_exception.LangDetectException:
                continue
        
        return ' '.join(english_words)
    
    except Exception as e:
        st.error(f"Language error: {e}")
        return text

def chunk_document(text, chunk_size=8000, chunk_overlap=500):
    """Split the document into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    return text_splitter.split_text(text)

def standardize_key_pointers(summary):
    """
    Standardize key pointers to ensure consistent formatting.
    
    Args:
        summary (str): The generated summary
    
    Returns:
        str: Standardized summary with consistent key pointers
    """
    # Split the summary into sections
    sections = summary.split('2. Key Pointers:')
    
    if len(sections) > 1:
        # Extract the key pointers
        pointers = sections[1].strip().split('\n')
        
        # Clean and standardize pointers
        cleaned_pointers = []
        for pointer in pointers:
            # Remove any bullet points or numbering
            clean_pointer = re.sub(r'^[-•*\d.)\s]+', '', pointer).strip()
            
            # Capitalize first letter, ensure it ends with a period
            if clean_pointer:
                clean_pointer = clean_pointer[0].upper() + clean_pointer[1:]
                if not clean_pointer.endswith('.'):
                    clean_pointer += '.'
                
                cleaned_pointers.append(clean_pointer)
        
        # Reconstruct the summary with standardized pointers
        standardized_summary = f"{sections[0].strip()}\n\n2. Key Pointers:\n"
        standardized_summary += '\n'.join([f"- {point}" for point in cleaned_pointers])
        
        return standardized_summary
    
    return summary

def create_summarization_chain(llm):
    """Create and configure the summarization chain using create_stuff_documents_chain."""
    
    # PII protection instructions
    pii_instructions = """
    IMPORTANT: DO NOT include any personally identifiable information (PII) in your summary, including:
    - Bank account numbers
    - Credit card numbers
    - Social security numbers
    - Passport numbers
    - Personal mobile numbers
    If you encounter such information, DO NOT include it in your summary.
    """

    # Single prompt for stuff documents chain
    stuff_prompt = PromptTemplate(
        input_variables=["context"],
        template=f"""{pii_instructions}
        
        You are analyzing PDF circular documents. Create a comprehensive summary from the following document content:
        
        Create a well-structured summary with the following EXACT format:
        
        1. Document Overview: [Brief description of the document's main purpose and scope]
        
        2. Key Pointers:
        - [Detailed bullet point covering major updates or changes]
        - [Specific regulatory or policy information with relevant details]
        - [Important deadlines, dates, or timeline information]
        - [Process changes or new procedures described]
        - [Any compliance requirements or mandatory actions]
        - [Other significant findings or announcements]
        
        Guidelines for Key Pointers:
        - Be specific and detailed rather than generic
        - Include relevant numbers, dates, and specific requirements
        - Focus on actionable information
        - Each point should be substantive and unique
        - Capture the most critical information from the document
        - Ensure each point starts with a capital letter and ends with a period
        - Focus on the level of details at a greater extent
        - Do not miss out on any specifications/details which are important
        - Create as many pointers as you see fit based on document content
        
        Document Content:
        {{context}}
        
        Summary:"""
    )

    # Create the stuff documents chain
    chain = create_stuff_documents_chain(llm, stuff_prompt)
    return chain
