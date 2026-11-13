"""The paste-a-screenshot component: only image data URLs come back as bytes; the page speaks Streamlit's protocol."""
import base64
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "app"))

PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")


def test_decode_accepts_only_image_data_urls():
    import paste_image as PI
    ok = {"data": "data:image/png;base64," + base64.b64encode(PNG).decode(), "mime": "image/png"}
    raw, mime = PI.decode(ok)
    assert raw == PNG and mime == "image/png"
    assert PI.decode(None) is None and PI.decode("x") is None
    assert PI.decode({"data": "data:text/html;base64,PGI+", "mime": "text/html"}) is None          # not an image
    assert PI.decode({"data": "data:image/png;base64,%%%", "mime": "image/png"}) is None           # bad base64
    assert PI.decode({"data": "data:image/jpeg;base64," + base64.b64encode(PNG).decode(), "mime": "image/png"}) is None


def test_component_page_implements_the_protocol():
    page = (REPO / "app" / "components" / "paste_image" / "index.html").read_text(encoding="utf-8")
    for needle in ("streamlit:componentReady", "streamlit:setComponentValue", "streamlit:setFrameHeight",
                   "streamlit:render", 'addEventListener("paste"', "MAX_BYTES"):
        assert needle in page
    assert "http://" not in page and "https://" not in page          # self-contained: no external script or font
