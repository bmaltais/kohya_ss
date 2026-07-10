"""Offline-safe Gradio UI helpers (issue #3508).

Removes third-party browser/server asset dependencies from the local GUI shell:
Google Fonts, About-tab remote badges, Gradio analytics, and cdnjs iframe-resizer.
"""

from __future__ import annotations

import re
from pathlib import Path

import gradio as gr
from gradio.themes.utils.fonts import Font, LocalFont

import kohya_gui.localization as localization

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_IFRAME_RESIZER_JS = (
    _PROJECT_ROOT / "assets" / "js" / "iframeResizer.contentWindow.min.js"
)

# Gradio frontend index.html / share.html remote host markers (5.x/6.x).
_CDNJS_IFRAME_RESIZER = (
    b"https://cdnjs.cloudflare.com/ajax/libs/iframe-resizer/"
    b"4.3.1/iframeResizer.contentWindow.min.js"
)
_FONTS_GOOGLEAPIS = b"https://fonts.googleapis.com"
_FONTS_GSTATIC = b"https://fonts.gstatic.com"

_REMOTE_MD_IMAGE = re.compile(r"!\[[^\]]*\]\(https?://[^)]+\)")
_REMOTE_HTML_IMAGE = re.compile(
    r"<img\b[^>]*\bsrc\s*=\s*[\"']https?://[^\"']+[\"'][^>]*/?>",
    re.IGNORECASE,
)

# Multiline <link rel="preconnect" … fonts.googleapis|gstatic …>
_PRECONNECT_FONTS = re.compile(
    rb'<link\b[^>]*rel=["\']preconnect["\'][^>]*>',
    re.IGNORECASE,
)
# Multiline script tag that loads iframe-resizer from cdnjs
_CDNJS_IFRAME_SCRIPT = re.compile(
    rb"<script\b[^>]*src=[\"']https://cdnjs\.cloudflare\.com/ajax/libs/"
    rb"iframe-resizer/[^\"']+iframeResizer\.contentWindow(?:\.min)?\.js[\"']"
    rb"[^>]*>\s*</script>",
    re.IGNORECASE,
)

_OFFLINE_PATCH_INSTALLED = False


def build_offline_theme() -> gr.themes.Base:
    """Default theme using only local/system fonts (no Google Fonts CDN)."""
    return gr.themes.Default(
        font=[
            LocalFont("IBM Plex Sans"),
            Font("ui-sans-serif"),
            Font("system-ui"),
            Font("sans-serif"),
        ],
        font_mono=[
            LocalFont("IBM Plex Mono"),
            Font("ui-monospace"),
            Font("Consolas"),
            Font("monospace"),
        ],
    )


def theme_stylesheet_urls(theme: gr.themes.Base) -> list[str]:
    """Collect external stylesheet URLs emitted by a theme's font stack.

    Gradio 6 may store plain strings (e.g. ``\"sans-serif\"``) alongside
    Font/LocalFont/GoogleFont objects in ``_font`` / ``_font_mono``; skip
    anything that does not expose ``stylesheet()``.
    """
    urls: list[str] = []
    for font in list(getattr(theme, "_font", []) or []) + list(
        getattr(theme, "_font_mono", []) or []
    ):
        stylesheet = getattr(font, "stylesheet", None)
        if not callable(stylesheet):
            continue
        sheet = stylesheet()
        url = sheet.get("url") if isinstance(sheet, dict) else None
        if url:
            urls.append(url)
    return urls


def sanitize_readme_for_about(content: str) -> str:
    """Strip remote images so the About tab does not fetch third-party hosts."""
    if not content:
        return content
    cleaned = _REMOTE_MD_IMAGE.sub("", content)
    cleaned = _REMOTE_HTML_IMAGE.sub("", cleaned)
    return cleaned


def load_iframe_resizer_js() -> str:
    """Return vendored iframe-resizer contentWindow script text."""
    return _IFRAME_RESIZER_JS.read_text(encoding="utf-8")


def rewrite_html_for_offline(body: bytes) -> bytes:
    """Rewrite Gradio frontend HTML to drop CDN font preconnects and cdnjs scripts."""
    if not body:
        return body

    def _drop_font_preconnect(match: re.Match[bytes]) -> bytes:
        tag = match.group(0)
        if _FONTS_GOOGLEAPIS in tag or _FONTS_GSTATIC in tag:
            return b""
        return tag

    out = _PRECONNECT_FONTS.sub(_drop_font_preconnect, body)
    out = _CDNJS_IFRAME_SCRIPT.sub(b"", out)

    # If a bare CDN URL remains (partial template variants), strip it.
    out = out.replace(_CDNJS_IFRAME_RESIZER, b"")

    if b"iframeResizer.contentWindow" not in out and _IFRAME_RESIZER_JS.is_file():
        script = load_iframe_resizer_js().encode("utf-8")
        injection = b'<script type="text/javascript">' + script + b"</script>"
        if b"</head>" in out:
            out = out.replace(b"</head>", injection + b"</head>", 1)
        else:
            out = injection + out

    return out


def install_offline_template_patches() -> None:
    """Monkey-patch Gradio TemplateResponse so served HTML is offline-safe.

    Idempotent. Chains under ``localization.GrRoutesTemplateResponse`` so the
    optional localization JS patch can still wrap this base.
    """
    global _OFFLINE_PATCH_INSTALLED
    if _OFFLINE_PATCH_INSTALLED:
        return

    if not hasattr(localization, "GrRoutesTemplateResponse"):
        localization.GrRoutesTemplateResponse = gr.routes.templates.TemplateResponse

    base = localization.GrRoutesTemplateResponse

    def offline_template_response(*args, **kwargs):
        res = base(*args, **kwargs)
        res.body = rewrite_html_for_offline(res.body)
        res.init_headers()
        return res

    localization.GrRoutesTemplateResponse = offline_template_response
    gr.routes.templates.TemplateResponse = offline_template_response
    _OFFLINE_PATCH_INSTALLED = True
