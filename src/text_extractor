from typing import Optional
import PyPDF2
import io


def extract_text_from_bytes(content: bytes, filename: str) -> Optional[str]:
    """
    Very simple extractor:
    - if .txt: decode as UTF-8
    - if .pdf: use PyPDF2
    """
    fname = filename.lower()

    if fname.endswith(".txt"):
        try:
            return content.decode("utf-8", errors="ignore")
        except Exception:
            return None

    if fname.endswith(".pdf"):
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
            text_chunks = []
            for page in pdf_reader.pages:
                page_text = page.extract_text() or ""
                text_chunks.append(page_text)
            return "\n".join(text_chunks)
        except Exception:
            return None

    # unsupported
    return None
