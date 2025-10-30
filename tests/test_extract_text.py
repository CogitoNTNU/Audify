import importlib.util
import sys
import types
from pathlib import Path


def _load_module(tmp_path, fake_file_modules=None):
    """Load the extract_text.py module from src/preprocessing in isolation.

    We inject simple dummy modules for top-level imports (file_to_md, video_to_md)
    so the file executes without trying to import real heavy deps.
    """
    repo_root = Path(__file__).resolve().parents[1]
    mod_path = repo_root / "src" / "preprocessing" / "extract_text.py"

    # Prepare minimal fake modules for top-level imports used by the file
    fake_file_modules = fake_file_modules or {}
    saved = {}
    for name, module in fake_file_modules.items():
        saved[name] = sys.modules.get(name)
        sys.modules[name] = module

    spec = importlib.util.spec_from_file_location("extract_text_module", str(mod_path))
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    finally:
        # Ensure the fake modules remain in sys.modules after loading the
        # module so runtime imports inside `extract_text` pick them up.
        for name, module in fake_file_modules.items():
            sys.modules[name] = module

    return mod


def test_extract_text_youtube(tmp_path):
    # fake youtube_to_markdown to create a markdown file and return its path
    fake_video = types.ModuleType("video_to_md")

    def fake_youtube_to_markdown(url):
        p = tmp_path / "yt.md"
        p.write_text("YouTube extracted text\nLine2")
        return str(p)

    fake_video.youtube_to_markdown = fake_youtube_to_markdown
    # provide a minimal file_to_md module with the expected symbol so import works
    fake_file_mod = types.ModuleType("file_to_md")
    fake_file_mod.file_to_md = lambda src, out: None

    mod = _load_module(tmp_path, {"video_to_md": fake_video, "file_to_md": fake_file_mod})

    text = mod.extract_text("https://youtube.com/watch?v=abc")
    assert "YouTube extracted text" in text


def test_extract_text_txt(tmp_path):
    fake_file = types.ModuleType("file_to_md")
    fake_file.file_to_md = lambda src, out: None
    fake_video = types.ModuleType("video_to_md")
    fake_video.youtube_to_markdown = lambda url: None
    mod = _load_module(tmp_path, {"file_to_md": fake_file, "video_to_md": fake_video})

    t = tmp_path / "sample.txt"
    t.write_text("hello world\nsecond line")

    out = mod.extract_text(str(t))
    assert "hello world" in out


def test_extract_text_http_calls_file_to_md(tmp_path):
    # Ensure that for http(s) inputs we call file_to_md and then read the generated markdown
    fake_file = types.ModuleType("file_to_md")

    # write the generated markdown into the pytest tmp_path sandbox instead
    def fake_file_to_md(src, out_name):
        md_dir = tmp_path / "fake" / "data" / "markdown"
        md_dir.mkdir(parents=True, exist_ok=True)
        p = md_dir / out_name
        p.write_text("web page content here")

    fake_file.file_to_md = fake_file_to_md
    fake_video = types.ModuleType("video_to_md")
    fake_video.youtube_to_markdown = lambda url: None

    mod = _load_module(tmp_path, {"file_to_md": fake_file, "video_to_md": fake_video})
    # Make extract_text look for data/markdown under our tmp_path fake tree
    fake_mod_file = tmp_path / "fake" / "src" / "preprocessing" / "extract_text.py"
    fake_mod_file.parent.mkdir(parents=True, exist_ok=True)
    mod.__file__ = str(fake_mod_file)

    txt = mod.extract_text("https://example.com/page")
    assert "web page content here" in txt


def test_extract_text_unsupported_raises(tmp_path):
    fake_file = types.ModuleType("file_to_md")
    fake_file.file_to_md = lambda src, out: None
    fake_video = types.ModuleType("video_to_md")
    fake_video.youtube_to_markdown = lambda url: None
    mod = _load_module(tmp_path, {"file_to_md": fake_file, "video_to_md": fake_video})

    import pytest

    with pytest.raises(ValueError):
        mod.extract_text("somefile.unknown")


def test_pdf_selectable_text(tmp_path):
    """If PDF pages contain selectable text of sufficient length, it's returned directly."""
    # fake modules
    fake_file = types.ModuleType("file_to_md")
    fake_file.file_to_md = lambda src, out: None
    fake_video = types.ModuleType("video_to_md")
    fake_video.youtube_to_markdown = lambda url: None

    class Page:
        def __init__(self, text):
            self._text = text

        def get_text(self):
            return self._text

    class FakePDF:
        def __init__(self, pages):
            self._pages = pages

        def __iter__(self):
            return iter(self._pages)

    fake_fitz = types.ModuleType("fitz")
    fake_fitz.open = lambda path: FakePDF([Page("X" * 100), Page("Y" * 60)])

    # load module with fakes
    mod = _load_module(tmp_path, {"file_to_md": fake_file, "video_to_md": fake_video, "fitz": fake_fitz})

    pdf_path = str(tmp_path / "doc.pdf")
    (tmp_path / "doc.pdf").write_bytes(b"%PDF-1.4")

    out = mod.extract_text(pdf_path)
    assert "XXXXX" in out or len(out) > 50


def test_pdf_ocr_fallback(tmp_path):
    """If selectable text is tiny, extract_text should fall back to OCR per-page."""
    fake_file = types.ModuleType("file_to_md")
    fake_file.file_to_md = lambda src, out: None
    fake_video = types.ModuleType("video_to_md")
    fake_video.youtube_to_markdown = lambda url: None

    class PageSmall:
        def get_text(self):
            return ""  # no selectable text

        def get_pixmap(self):
            class Pix:
                width = 2
                height = 2
                samples = b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"

            return Pix()

    class FakePDF:
        def __init__(self, pages):
            self._pages = pages

        def __iter__(self):
            return iter(self._pages)

    fake_fitz = types.ModuleType("fitz")
    fake_fitz.open = lambda path: FakePDF([PageSmall(), PageSmall()])

    # pytesseract and PIL.Image
    fake_pyt = types.ModuleType("pytesseract")
    fake_pyt.image_to_string = lambda img: "OCR PAGE TEXT"

    fake_PIL = types.ModuleType("PIL")
    fake_Image = types.SimpleNamespace()

    def frombytes(mode, size, samples):
        return object()

    fake_Image.frombytes = frombytes
    fake_PIL.Image = fake_Image

    mod = _load_module(tmp_path, {
        "file_to_md": fake_file,
        "video_to_md": fake_video,
        "fitz": fake_fitz,
        "pytesseract": fake_pyt,
        "PIL": fake_PIL,
    })

    pdf_path = str(tmp_path / "ocr.pdf")
    (tmp_path / "ocr.pdf").write_bytes(b"%PDF-1.4")

    # ensure runtime imports use our fakes
    sys.modules["fitz"] = fake_fitz
    sys.modules["pytesseract"] = fake_pyt
    sys.modules["PIL"] = fake_PIL

    # track that get_text was called to confirm fallback decision
    called = {"get_text": 0}

    def counted_get_text(self):
        called["get_text"] += 1
        return ""

    # monkeypatch PageSmall.get_text
    PageSmall.get_text = counted_get_text

    out = mod.extract_text(pdf_path)
    assert "OCR PAGE TEXT" in out
    assert called["get_text"] > 0


def test_image_ocr_and_logger(tmp_path):
    """Image OCR path should call pytesseract.image_to_string and the logger should receive messages."""
    fake_file = types.ModuleType("file_to_md")
    fake_file.file_to_md = lambda src, out: None
    fake_video = types.ModuleType("video_to_md")
    fake_video.youtube_to_markdown = lambda url: None

    fake_pyt = types.ModuleType("pytesseract")
    fake_pyt.image_to_string = lambda img: "IMAGE OCR TEXT"

    fake_PIL = types.ModuleType("PIL")
    fake_Image = types.SimpleNamespace()

    def open_fn(path):
        return object()

    fake_Image.open = open_fn
    fake_PIL.Image = fake_Image

    messages = []

    def logger(msg):
        messages.append(msg)

    mod = _load_module(tmp_path, {
        "file_to_md": fake_file,
        "video_to_md": fake_video,
        "pytesseract": fake_pyt,
        "PIL": fake_PIL,
    })

    img_path = tmp_path / "pic.jpg"
    img_path.write_bytes(b"\x89PNG\r\n")

    # ensure runtime imports use our fakes
    sys.modules["pytesseract"] = fake_pyt
    sys.modules["PIL"] = fake_PIL

    out = mod.extract_text(str(img_path), logger=logger)
    assert "IMAGE OCR TEXT" in out
    # logger should have been called at least once and contain 'image OCR' message
    assert any("image OCR" in (m.lower()) or "using image ocr" in (m.lower()) for m in messages)
