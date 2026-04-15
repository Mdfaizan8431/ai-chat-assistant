"""
Document Processing Module
Handles file uploads, text extraction, and chunking
"""

from pypdf import PdfReader
from typing import List, Dict
import os

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF file"""
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        raise Exception(f"Error reading PDF: {str(e)}")

def extract_text_from_txt(file_path: str) -> str:
    """Extract text from TXT file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        raise Exception(f"Error reading TXT file: {str(e)}")

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks
    
    Args:
        text: Text to split
        chunk_size: Target size of each chunk (in characters)
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    if not text:
        return []
    
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        # Get chunk
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to break at sentence boundary
        if end < text_length:
            # Look for sentence endings
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            last_break = max(last_period, last_newline)
            
            if last_break > chunk_size * 0.5:  # At least 50% through
                chunk = chunk[:last_break + 1]
                end = start + last_break + 1
        
        chunks.append(chunk.strip())
        
        # Move to next chunk with overlap
        start = end - overlap
    
    return [c for c in chunks if c]  # Filter empty chunks

def process_file(file_path: str, filename: str) -> Dict:
    """
    Process uploaded file and extract text chunks
    
    Args:
        file_path: Path to the file
        filename: Original filename
        
    Returns:
        Dict with chunks and metadata
    """
    # Determine file type
    ext = filename.lower().split('.')[-1]
    
    # Extract text based on file type
    if ext == 'pdf':
        text = extract_text_from_pdf(file_path)
    elif ext == 'txt':
        text = extract_text_from_txt(file_path)
    else:
        raise Exception(f"Unsupported file type: {ext}")
    
    # Chunk the text
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    
    # Create metadata for each chunk
    metadatas = [
        {
            'source': filename,
            'chunk_index': i,
            'file_type': ext
        }
        for i in range(len(chunks))
    ]
    
    return {
        'chunks': chunks,
        'metadatas': metadatas,
        'total_chunks': len(chunks),
        'filename': filename
    }

def get_supported_extensions() -> List[str]:
    """Get list of supported file extensions"""
    return ['pdf', 'txt']