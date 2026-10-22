import abc
import logging
import os
from typing import Dict, List, Any, Optional
from pypdf import PdfReader
from pipeline.config import settings

logger = logging.getLogger(__name__)

_VISION_PROMPT = """Analyze this image extracted from a PDF document page.
If it is a chart, graph, diagram, or table, extract all numbers, trends, axis labels, titles, data points, and facts, and summarize it into a detailed paragraph of factual sentences (e.g. "Revenue increased from $2M in 2022 to $5M in 2024").
If it is an illustration or photo, describe its content and purpose concisely.
Output only the factual description. Do not include introductory text like 'This image shows...'."""

class ImageDescriber(abc.ABC):
    """Turns a chart/photo into retrievable text at ingest time."""
    @abc.abstractmethod
    def describe(self, image_bytes: bytes, mime_type: str) -> Optional[str]:
        pass

class OllamaVisionDescriber(ImageDescriber):
    """Local vision model served by Ollama (default: llama3.2-vision). No data leaves the host."""

    def __init__(self, base_url: Optional[str] = None, model_name: Optional[str] = None):
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model_name = model_name or settings.ollama_vision_model

    def describe(self, image_bytes: bytes, mime_type: str) -> Optional[str]:
        import base64
        import json
        import urllib.request
        body = json.dumps({
            "model": self.model_name,
            "prompt": _VISION_PROMPT,
            "images": [base64.b64encode(image_bytes).decode("ascii")],
            "stream": False,
            "options": {"temperature": 0},
        }).encode("utf-8")
        req = urllib.request.Request(f"{self.base_url}/api/generate", data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=settings.vision_timeout_seconds) as resp:
                text = json.loads(resp.read().decode("utf-8")).get("response", "")
            return text.strip() or None
        except Exception as e:
            logger.warning("Local image description failed (%s): %s", self.model_name, e)
            return None

class GeminiVisionDescriber(ImageDescriber):
    """Opt-in cloud vision. Sends images to Google; only used when VISION_PROVIDER=gemini."""

    def __init__(self, api_key: str):
        from google import genai
        self.client = genai.Client(api_key=api_key)

    def describe(self, image_bytes: bytes, mime_type: str) -> Optional[str]:
        try:
            from google.genai import types
            response = self.client.models.generate_content(
                model=settings.gemini_model,
                contents=[types.Part.from_bytes(data=image_bytes, mime_type=mime_type), _VISION_PROMPT],
            )
            return response.text.strip() if response.text else None
        except Exception as e:
            logger.warning("Gemini image description failed: %s", e)
            return None

def get_vision_client(force: bool = False) -> Optional[ImageDescriber]:
    """
    Image describer for ingestion, or None. Runs when DESCRIBE_IMAGES=true or
    the caller forces it (per-upload enrich flag). Local Ollama is the default.
    """
    if not (force or settings.describe_images):
        return None
    provider = (settings.vision_provider or "ollama").lower()
    if provider == "ollama":
        from pipeline.generator import probe_ollama
        if probe_ollama(settings.ollama_base_url):
            return OllamaVisionDescriber()
        logger.warning("Image description requested but Ollama is not reachable at %s; skipping.",
                       settings.ollama_base_url)
        return None
    if provider == "gemini" and settings.gemini_api_key:
        try:
            return GeminiVisionDescriber(settings.gemini_api_key)
        except Exception as e:
            logger.warning("Gemini vision unavailable: %s", e)
    return None

class BaseParser(abc.ABC):
    @abc.abstractmethod
    def parse(self, file_path: str, client: Any = None) -> Dict[str, Any]:
        """
        Parses a file and returns:
        {
            "metadata": {"source", "file_path", "title", "total_pages", ...},
            "content": "Full extracted text",
            "pages": [{"page_num": 1, "text": "..."}]
        }
        """
        pass

class SimpleParser(BaseParser):
    """Fallback parser: plain text / pypdf. Used when Docling is unavailable or fails."""

    def parse(self, file_path: str, client: Any = None) -> Dict[str, Any]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        filename = os.path.basename(file_path)

        metadata = {
            "source": filename,
            "file_path": file_path,
            "title": filename,
        }

        if ext == ".pdf":
            reader = PdfReader(file_path)
            pages_data = []
            full_text = []

            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                page_num = i + 1

                if client and page.images:
                    descriptions = []
                    for img in page.images:
                        mime_type = "image/jpeg"
                        name_lower = img.name.lower()
                        if name_lower.endswith(".png"):
                            mime_type = "image/png"
                        elif name_lower.endswith(".gif"):
                            mime_type = "image/gif"
                        elif name_lower.endswith(".webp"):
                            mime_type = "image/webp"
                        desc = client.describe(img.data, mime_type)
                        if desc:
                            descriptions.append(f"\n\n[Extracted Visual Info - Page {page_num}]:\n{desc}")
                    if descriptions:
                        text += "\n" + "\n".join(descriptions)

                pages_data.append({"page_num": page_num, "text": text})
                full_text.append(text)

            return {
                "metadata": {**metadata, "total_pages": len(reader.pages), "parser": "pypdf"},
                "content": "\n\n".join(full_text),
                "pages": pages_data,
            }

        # Plain text-like files (also the generic fallback)
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        return {
            "metadata": {**metadata, "total_pages": 1, "parser": "text"},
            "content": content,
            "pages": [{"page_num": 1, "text": content}],
        }

class DoclingParser(BaseParser):
    """
    Local multimodal document parser (IG steps 1-2):
      - structured text with headings preserved (markdown)
      - tables extracted as discrete markdown units (kept whole by the chunker)
      - OCR for scanned pages (Docling built-in, local)
      - page provenance preserved for correct page-number citations
      - charts/photos optionally described by a local Ollama vision model
        (ingest-time only; Gemini is an opt-in alternative)
    """

    def __init__(self):
        self.docling_available = False
        self.fallback_parser = SimpleParser()
        try:
            from docling.document_converter import DocumentConverter
            self._converter = DocumentConverter()
            self.docling_available = True
        except ImportError:
            pass

    def parse(self, file_path: str, client: Any = None) -> Dict[str, Any]:
        ext = os.path.splitext(file_path)[1].lower()
        # Docling is for rich documents; plain text/code goes straight to SimpleParser
        if not self.docling_available or ext in (".txt", ".md", ".markdown") or ext not in (".pdf", ".docx", ".pptx", ".html"):
            return self.fallback_parser.parse(file_path, client)

        filename = os.path.basename(file_path)
        try:
            result = self._converter.convert(file_path)
            document = result.document

            # Group exported content by page using item provenance
            pages_text: Dict[int, List[str]] = {}

            def add(page_no: int, text: str):
                if text and text.strip():
                    pages_text.setdefault(page_no, []).append(text.strip())

            try:
                from docling_core.types.doc import TableItem, PictureItem, SectionHeaderItem, TextItem
                for item, _level in document.iterate_items():
                    page_no = 1
                    prov = getattr(item, "prov", None)
                    if prov:
                        page_no = getattr(prov[0], "page_no", 1) or 1

                    if isinstance(item, TableItem):
                        try:
                            table_md = item.export_to_markdown(doc=document)
                        except TypeError:
                            table_md = item.export_to_markdown()
                        add(page_no, table_md)
                    elif isinstance(item, PictureItem):
                        if client:
                            try:
                                img = item.get_image(document)
                                if img is not None:
                                    import io
                                    buf = io.BytesIO()
                                    img.save(buf, format="PNG")
                                    desc = client.describe(buf.getvalue(), "image/png")
                                    if desc:
                                        add(page_no, f"[Extracted Visual Info - Page {page_no}]:\n{desc}")
                            except Exception as img_err:
                                logger.warning("Image description failed on page %s: %s", page_no, img_err)
                    elif isinstance(item, SectionHeaderItem):
                        level = getattr(item, "level", 1) or 1
                        add(page_no, f"{'#' * min(level, 6)} {item.text}")
                    elif isinstance(item, TextItem):
                        add(page_no, item.text)
                    else:
                        text = getattr(item, "text", None)
                        if text:
                            add(page_no, text)
            except Exception:
                # Provenance walk failed (docling API drift): fall back to
                # whole-document markdown as a single page rather than losing the doc
                pages_text = {1: [document.export_to_markdown()]}

            if not pages_text:
                pages_text = {1: [document.export_to_markdown()]}

            page_nums = sorted(pages_text.keys())
            pages_data = [
                {"page_num": p, "text": "\n\n".join(pages_text[p])}
                for p in page_nums
            ]
            full_content = "\n\n".join(pd["text"] for pd in pages_data)

            metadata = {
                "source": filename,
                "file_path": file_path,
                "title": getattr(getattr(document, "name", None), "strip", lambda: filename)() or filename,
                "total_pages": max(page_nums) if page_nums else 1,
                "parser": "docling",
            }

            return {
                "metadata": metadata,
                "content": full_content,
                "pages": pages_data,
            }
        except Exception as e:
            logger.warning("Docling parsing failed for %s: %s. Falling back to SimpleParser.", filename, e)
            return self.fallback_parser.parse(file_path, client)
