"""Text extraction for searchable, text-based PDF documents."""

from io import BytesIO
from pypdf import PdfReader


MAX_PDF_PAGES = 100
MAX_EXTRACTED_CHARACTERS = 60000


def extract_pdf_text(file_bytes):
    """Extract text page-by-page while preserving page numbers."""
    if not isinstance(file_bytes, (bytes, bytearray)):
        raise TypeError("PDF content must be supplied as bytes.")
    if not file_bytes:
        raise ValueError("The uploaded PDF is empty.")

    try:
        reader = PdfReader(BytesIO(file_bytes), strict=False)
    except Exception as error:
        raise ValueError(f"Unable to open this PDF: {error}") from error

    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception as error:
            raise ValueError("This PDF is password-protected and cannot be opened.") from error

    total_pages = len(reader.pages)
    if total_pages == 0:
        raise ValueError("The PDF does not contain any pages.")
    if total_pages > MAX_PDF_PAGES:
        raise ValueError(
            f"This PDF has {total_pages} pages. The current limit is "
            f"{MAX_PDF_PAGES} pages per document."
        )

    extracted_pages = []
    total_characters = 0
    pages_without_text = []

    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = (page.extract_text() or "").strip()
        except Exception:
            text = ""

        if not text:
            pages_without_text.append(page_number)
            continue

        remaining = MAX_EXTRACTED_CHARACTERS - total_characters
        if remaining <= 0:
            break

        text = text[:remaining]
        extracted_pages.append({
            "page": page_number,
            "text": text,
        })
        total_characters += len(text)

        if len(text) == remaining:
            break

    if not extracted_pages:
        raise ValueError(
            "No selectable text was found. This may be a scanned PDF; "
            "OCR is not supported in this version."
        )

    return {
        "total_pages": total_pages,
        "pages_with_text": len(extracted_pages),
        "pages_without_text": pages_without_text,
        "characters_extracted": total_characters,
        "truncated": total_characters >= MAX_EXTRACTED_CHARACTERS,
        "pages": extracted_pages,
    }
