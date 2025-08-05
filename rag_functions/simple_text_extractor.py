"""
Simple text extraction fallback that doesn't require layout parser
================================================================

This module provides simple text extraction methods that can be used
when the full layout parser system is not available.
"""

import os
from pathlib import Path

def extract_with_pymupdf(pdf_path):
    """Extract text using PyMuPDF (fitz) - lightweight option"""
    try:
        import fitz  # PyMuPDF
        
        doc = fitz.open(pdf_path)
        text = ""
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_text = page.get_text()
            text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
        
        doc.close()
        return text
        
    except ImportError:
        print("⚠️ PyMuPDF not available")
        return None
    except Exception as e:
        print(f"❌ Error with PyMuPDF: {e}")
        return None

def extract_with_pdfplumber(pdf_path):
    """Extract text using pdfplumber - good for tables"""
    try:
        import pdfplumber
        
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
        
        return text
        
    except ImportError:
        print("⚠️ pdfplumber not available")
        return None
    except Exception as e:
        print(f"❌ Error with pdfplumber: {e}")
        return None

def extract_with_pypdf(pdf_path):
    """Extract text using PyPDF2/pypdf - basic extraction"""
    try:
        import pypdf
        
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
        
        return text
        
    except ImportError:
        print("⚠️ pypdf not available")
        return None
    except Exception as e:
        print(f"❌ Error with pypdf: {e}")
        return None

def extract_simple_text(pdf_path):
    """
    Extract text using the best available simple method.
    Tries multiple libraries in order of preference.
    """
    
    if not os.path.exists(pdf_path):
        return f"Error: File {pdf_path} not found"
    
    print(f"📄 Extracting text from: {Path(pdf_path).name}")
    
    # Try methods in order of preference
    methods = [
        ("PyMuPDF", extract_with_pymupdf),
        ("pdfplumber", extract_with_pdfplumber), 
        ("pypdf", extract_with_pypdf)
    ]
    
    for method_name, method_func in methods:
        print(f"🔄 Trying {method_name}...")
        result = method_func(pdf_path)
        if result:
            print(f"✅ Successfully extracted text using {method_name}")
            return result
    
    return "Error: Could not extract text with any available method. Please install PyMuPDF, pdfplumber, or pypdf."

# Simple usage test
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        text = extract_simple_text(pdf_path)
        print("\n" + "="*50)
        print("EXTRACTED TEXT")
        print("="*50)
        print(text[:1000] + "..." if len(text) > 1000 else text)
    else:
        print("Usage: python simple_text_extractor.py <pdf_file>")