"""
BrainOS — Document Ingester
Extracts text from URLs and PDFs.
"""

from __future__ import annotations

from pathlib import Path

import requests
from bs4 import BeautifulSoup

from src.utils.logger import log


class DocumentIngester:

    # ------------------------------------------------------------------
    # URL
    # ------------------------------------------------------------------

    def fetch_url(self, url: str, timeout: int = 15) -> str:
        """Fetch a web page and return cleaned text."""
        log.info(f"Fetching URL: {url}")
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (compatible; BrainOS/1.0)"
            )
        }
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            resp.raise_for_status()
        except requests.RequestException as e:
            log.error(f"Failed to fetch URL {url}: {e}")
            raise

        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove noise tags
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        text = soup.get_text(separator="\n")
        text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
        log.info(f"Extracted {len(text)} chars from URL.")
        return text

    # ------------------------------------------------------------------
    # PDF
    # ------------------------------------------------------------------

    def read_pdf(self, pdf_path: str | Path) -> str:
        """Extract text from a PDF file using pypdf."""
        from pypdf import PdfReader

        pdf_path = Path(pdf_path)
        log.info(f"Reading PDF: {pdf_path}")

        reader = PdfReader(str(pdf_path))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n\n".join(pages).strip()

        log.info(f"Extracted {len(text)} chars from {len(pages)}-page PDF.")
        return text


# Singleton
document_ingester = DocumentIngester()
