import fitz
import logging
import os

logger = logging.getLogger(__name__)

def extract_text_from_pdf(path):
    """
    Extract text from PDF file with robust error handling
    
    Args:
        path (str): Path to the PDF file
        
    Returns:
        str: Extracted text content
        
    Raises:
        Exception: If PDF extraction fails
    """
    doc = None
    try:
        # Validate file exists
        if not os.path.exists(path):
            raise FileNotFoundError(f"PDF file not found: {path}")
        
        # Check file size (optional - prevent processing huge files)
        file_size = os.path.getsize(path)
        if file_size > 50 * 1024 * 1024:  # 50MB limit
            logger.warning(f"Large PDF file detected: {file_size / 1024 / 1024:.1f}MB")
        
        # Open PDF document
        doc = fitz.open(path)
        
        if doc.page_count == 0:
            raise ValueError("PDF contains no pages")
        
        logger.info(f"Processing PDF with {doc.page_count} pages")
        
        # Extract text from all pages
        text_parts = []
        for page_num in range(doc.page_count):
            try:
                page = doc[page_num]
                page_text = page.get_text()
                
                if page_text.strip():  # Only add non-empty pages
                    text_parts.append(page_text)
                else:
                    logger.warning(f"Page {page_num + 1} contains no text")
                    
            except Exception as e:
                logger.error(f"Error extracting text from page {page_num + 1}: {str(e)}")
                continue
        
        # Combine all text
        full_text = "\n".join(text_parts)
        
        # Validate extracted content
        if not full_text.strip():
            raise ValueError("No text content could be extracted from the PDF")
        
        # Clean up whitespace
        full_text = " ".join(full_text.split())
        
        logger.info(f"Successfully extracted {len(full_text)} characters from PDF")
        return full_text
        
    except fitz.FileDataError as e:
        logger.error(f"PDF file is corrupted or invalid: {str(e)}")
        raise Exception(f"Invalid PDF file: {str(e)}")
        
    except fitz.FileNotFoundError as e:
        logger.error(f"PDF file not found: {str(e)}")
        raise Exception(f"PDF file not found: {str(e)}")
        
    except Exception as e:
        logger.error(f"Unexpected error during PDF extraction: {str(e)}")
        raise Exception(f"PDF extraction failed: {str(e)}")
        
    finally:
        # Always close the document
        if doc:
            try:
                doc.close()
            except:
                pass  # Ignore errors during cleanup

def validate_pdf(path):
    """
    Validate if a file is a proper PDF
    
    Args:
        path (str): Path to the PDF file
        
    Returns:
        bool: True if valid PDF, False otherwise
    """
    try:
        doc = fitz.open(path)
        is_valid = doc.page_count > 0
        doc.close()
        return is_valid
    except:
        return False

def get_pdf_info(path):
    """
    Get basic information about a PDF file
    
    Args:
        path (str): Path to the PDF file
        
    Returns:
        dict: PDF metadata
    """
    try:
        doc = fitz.open(path)
        info = {
            'page_count': doc.page_count,
            'title': doc.metadata.get('title', ''),
            'author': doc.metadata.get('author', ''),
            'subject': doc.metadata.get('subject', ''),
            'creator': doc.metadata.get('creator', ''),
            'producer': doc.metadata.get('producer', ''),
            'creation_date': doc.metadata.get('creationDate', ''),
            'modification_date': doc.metadata.get('modDate', ''),
            'is_encrypted': doc.is_encrypted,
            'is_pdf': doc.is_pdf
        }
        doc.close()
        return info
    except Exception as e:
        logger.error(f"Error getting PDF info: {str(e)}")
        return {}