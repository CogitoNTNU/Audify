import os
from urllib.parse import urlparse
from typing import Optional, Callable
from pathlib import Path
import importlib
file_mod = importlib.import_module("preprocessing.file_to_md")
video_mod = importlib.import_module("preprocessing.video_to_md")

# Configurable constants
# Image file extensions supported for OCR (lower-case). Add more here if needed.
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif")
# Minimum number of characters extracted from a PDF before we decide NOT to run OCR
PDF_TEXT_MIN_CHARS = 50


def extract_text(
    input_source: str,
    logger: Optional[Callable[[str], None]] = None,
    pdf_text_min_chars: int = PDF_TEXT_MIN_CHARS,
    enable_pdf_ocr: bool = True,
    image_extensions: tuple = IMAGE_EXTENSIONS,
) -> str:
    """
    Detects input type (PDF, image, text, YouTube, URL)
    and extracts text using the appropriate method.
    """
    parsed = urlparse(input_source)
    # cache lowercase version to avoid repeated .lower() calls
    lower_source = input_source.lower()

    def _log(msg: str) -> None:
        if logger:
            try:
                logger(msg)
            except Exception:
                # swallow logger errors
                pass

    # YouTube
    if "youtube.com" in lower_source or "youtu.be" in lower_source:
        _log("extract_text: using YouTube extractor")
        video_mod = importlib.import_module("preprocessing.video_to_md")
        md_path = video_mod.youtube_to_markdown(input_source)
        with open(md_path, "r", encoding="utf-8") as f:
            return f.read()

    #PDF
    #Try to extract selectable text first; if empty or very small, fall back to OCR per page.
    if lower_source.endswith(".pdf"):
        _log("extract_text: handling PDF (try text extraction then OCR)")
        try:
            import fitz
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "PyMuPDF (fitz) is required for PDF text extraction. Install with `pip install pymupdf`."
            ) from exc

        pdf = fitz.open(input_source)
        extracted = []
        for page in pdf:
            extracted.append(page.get_text())
        text = "\n".join(extracted).strip()
        # Use the value passed in (function arg) so callers can override the heuristic
        if len(text) > pdf_text_min_chars:
            _log("extract_text: used PDF text extractor")
            return text
        # If the extracted selectable text is short, we won't run OCR here.
        # Historically this code attempted a heavy OCR fallback (pytesseract/Pillow).
        # That functionality has been removed to keep the code lightweight and
        # avoid depending on heavy native libraries during tests/runtime.
        if len(text) > pdf_text_min_chars:
            _log("extract_text: used PDF text extractor")
            return text

        _log("extract_text: PDF OCR fallback removed; returning extracted text (may be short)")
        return text

    #Website or other files handled by markitdown
    elif parsed.scheme in ("http", "https") or lower_source.endswith((".docx", ".html")):
        _log("extract_text: using markitdown converter")
        output_name = _make_output_name(input_source, parsed)
        file_mod = importlib.import_module("preprocessing.file_to_md")
        file_mod.file_to_md(input_source, output_name)

        # Resolve markdown path relative to this module's file at runtime so
        # tests can monkeypatch `__file__` or use tmp directories.
        base = Path(__file__).resolve().parents[2]
        md_path = base / "data" / "markdown" / output_name
        try:
            with open(md_path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            _log(f"extract_text: markdown file not found: {md_path}")
            raise FileNotFoundError(
                f"Expected markdown output at {md_path}. `file_to_md` may have failed to create it."
            )

    # Image inputs: OCR functionality removed. Log and raise to make the
    # behavior explicit for callers/tests instead of attempting heavy OCR.
    elif lower_source.endswith(image_extensions):
        _log("extract_text: image OCR is disabled in this build")
        raise ValueError(f"Image OCR is disabled: {input_source}")

    #Plain text file
    elif input_source.lower().endswith(".txt"):
        _log("extract_text: reading plain text file")
        with open(input_source, "r", encoding="utf-8") as f:
            return f.read()

    else:
        _log(f"extract_text: unsupported input type: {input_source}")
        raise ValueError(f"Unsupported input type: {input_source}")


def _make_output_name(input_source: str, parsed: Optional[object]) -> str:
    """Create a safe markdown filename for outputs produced by file_to_md.

    Prefer a basename from the URL/path; fall back to netloc; finally use a sanitized version.
    """
    name = None
    try:
        if parsed and parsed.path:
            basename = os.path.basename(parsed.path)
            if basename:
                name = basename
        if not name:
            # use network location
            name = parsed.netloc if parsed and parsed.netloc else os.path.basename(input_source)
    except Exception:
        name = os.path.basename(input_source)

    # sanitize and ensure md extension
    name = name.replace("/", "_").replace("\\", "_")

    # Preserve the last dot (for readability) and only replace dots in the base part
    if "." in name:
        last_dot = name.rfind(".")
        base = name[:last_dot]
        ext = name[last_dot:]
        base = base.replace(".", "_")
        name = f"{base}{ext}"
    else:
        name = name.replace(".", "_")

    # finally, ensure the output filename ends with .md
    if not name.endswith(".md"):
        name = f"{name}.md"
    return name

# if __name__ == "__main__":
#     # file_path = extract_text("https://www.youtube.com/watch?v=Gx5qb1uHss4")
#     file_path = extract_text("./data/raw/Arbeidsmiljøloven.pdf")
#     print("Transcript saved at:", file_path)

