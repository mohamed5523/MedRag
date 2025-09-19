from pathlib import Path
from typing import Optional
import markdown
from docx import Document as DocxDocument
from pypdf import PdfReader
import logging

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Extract raw text from various document formats.
    Supports: PDF, DOCX, TXT, MD.
    """
    
    @staticmethod
    def load_document(file_path: Path) -> str:
        """Extract raw text from a file."""
        try:
            suffix = file_path.suffix.lower()
            
            if suffix == ".pdf":
                return DocumentProcessor._extract_from_pdf(file_path)
            elif suffix in {".doc", ".docx"}:
                return DocumentProcessor._extract_from_docx(file_path)
            elif suffix == ".txt":
                return DocumentProcessor._extract_from_txt(file_path)
            elif suffix == ".md":
                return DocumentProcessor._extract_from_md(file_path)
            else:
                raise ValueError(f"Unsupported file type: {suffix}")
                
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            raise
    
    @staticmethod
    def _extract_from_pdf(file_path: Path) -> str:
        """Extract text from PDF file."""
        reader = PdfReader(str(file_path)) 
        text_parts = []
        
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text.strip())
        
        return "\n\n".join(text_parts)
    
    @staticmethod
    def _extract_from_docx(file_path: Path) -> str:
        """Extract text from DOCX file."""
        doc = DocxDocument(str(file_path))
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(paragraphs)
    
    @staticmethod
    def _extract_from_txt(file_path: Path) -> str:
        """Extract text from TXT file."""
        return file_path.read_text(encoding="utf-8", errors="ignore")
    
    @staticmethod
    def _extract_from_md(file_path: Path) -> str:
        """Extract text from Markdown file."""
        md_content = file_path.read_text(encoding="utf-8", errors="ignore")
        return markdown.markdown(md_content)
    
    @staticmethod
    def get_supported_extensions() -> list[str]:
        """Return list of supported file extensions."""
        return [".pdf", ".docx", ".doc", ".txt", ".md"]
    
    @staticmethod
    def is_supported_file(filename: str) -> bool:
        """Check if file extension is supported."""
        return Path(filename).suffix.lower() in DocumentProcessor.get_supported_extensions()
    
