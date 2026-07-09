"""Offline-safe Gradio GUI shell (issue #3508).

Covers theme fonts, About README sanitization, analytics default, and
CDN HTML rewriting for iframe-resizer / Google Fonts preconnects.
"""

import importlib.util
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import gradio as gr
import pytest

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from kohya_gui.offline_assets import (  # noqa: E402
    build_offline_theme,
    load_iframe_resizer_js,
    rewrite_html_for_offline,
    sanitize_readme_for_about,
    theme_stylesheet_urls,
)

_spec = importlib.util.spec_from_file_location(
    "kohya_gui_launcher", os.path.join(_repo_root, "kohya_gui.py")
)
kohya_gui_launcher = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(kohya_gui_launcher)


# --- Cycle 1: system / local fonts (no Google Fonts) ---


def test_ui_theme_does_not_emit_google_fonts_stylesheet():
    theme = build_offline_theme()
    urls = theme_stylesheet_urls(theme)
    for url in urls:
        assert "fonts.googleapis.com" not in url
        assert "fonts.gstatic.com" not in url
    assert urls == []


def test_default_gradio_theme_still_uses_google_fonts_for_contrast():
    """Guard: stock Default() still hits Google Fonts; our helper must diverge."""
    stock_urls = theme_stylesheet_urls(gr.themes.Default())
    assert any("fonts.googleapis.com" in u for u in stock_urls)


# --- Cycle 2: About / README offline-safe content ---


def test_about_readme_content_has_no_remote_image_hosts():
    sample = "\n".join(
        [
            "# Kohya's GUI",
            "[![GitHub stars](https://img.shields.io/github/stars/bmaltais/kohya_ss?style=social)](https://github.com/bmaltais/kohya_ss/stargazers)",
            "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/camenduru/kohya_ss-colab)",
            "![local](./assets/style.css)",
            '<img src="https://img.shields.io/github/license/bmaltais/kohya_ss" />',
            "Body text stays.",
        ]
    )
    cleaned = sanitize_readme_for_about(sample)
    assert "img.shields.io" not in cleaned
    assert "colab.research.google.com/assets/colab-badge.svg" not in cleaned
    assert "colab-badge.svg" not in cleaned
    assert "![local](./assets/style.css)" in cleaned
    assert "Body text stays." in cleaned


def test_sanitize_readme_for_about_on_real_readme():
    readme_path = Path(_repo_root) / "README.md"
    if not readme_path.is_file():
        pytest.skip("README.md missing")
    cleaned = sanitize_readme_for_about(readme_path.read_text(encoding="utf-8"))
    assert "img.shields.io" not in cleaned
    assert "colab-badge.svg" not in cleaned
    assert "colab.research.google.com/assets/" not in cleaned


# --- Cycle 3: Gradio analytics disabled ---


def test_blocks_created_with_analytics_disabled():
    mock_config = MagicMock()
    mock_config.get.return_value = True

    tab_patches = [
        patch.object(
            kohya_gui_launcher,
            "dreambooth_tab",
            return_value=(MagicMock(), MagicMock(), MagicMock(), MagicMock()),
        ),
        patch.object(kohya_gui_launcher, "lora_tab"),
        patch.object(kohya_gui_launcher, "leco_tab"),
        patch.object(kohya_gui_launcher, "anima_lllite_tab"),
        patch.object(kohya_gui_launcher, "ti_tab"),
        patch.object(kohya_gui_launcher, "finetune_tab"),
        patch.object(kohya_gui_launcher, "utilities_tab"),
        patch.object(kohya_gui_launcher, "settings_tab"),
        patch.object(kohya_gui_launcher, "LoRATools"),
    ]

    with (
        patch.object(kohya_gui_launcher.gr, "Blocks") as mock_blocks,
        patch.object(kohya_gui_launcher.gr, "Tab") as mock_tab,
        patch.object(kohya_gui_launcher.gr, "Markdown"),
        patch.object(
            kohya_gui_launcher, "build_offline_theme", wraps=build_offline_theme
        ) as mock_theme,
    ):
        blocks_cm = MagicMock()
        mock_blocks.return_value = blocks_cm
        blocks_cm.__enter__.return_value = blocks_cm
        blocks_cm.__exit__.return_value = None

        tab_cm = MagicMock()
        mock_tab.return_value = tab_cm
        tab_cm.__enter__.return_value = tab_cm
        tab_cm.__exit__.return_value = None

        for p in tab_patches:
            p.start()
        try:
            kohya_gui_launcher.initialize_ui_interface(
                mock_config,
                "./config.toml",
                True,
                True,
                "test-release",
                "# readme",
            )
        finally:
            for p in tab_patches:
                p.stop()

    mock_blocks.assert_called_once()
    kwargs = mock_blocks.call_args.kwargs
    assert kwargs.get("analytics_enabled") is False
    mock_theme.assert_called_once()
    theme = kwargs.get("theme")
    assert theme is not None
    for url in theme_stylesheet_urls(theme):
        assert "fonts.googleapis.com" not in url


# --- Cycle 4: iframe-resizer / Gradio template CDN ---


_SAMPLE_INDEX_HTML = b"""<!doctype html>
<html>
  <head>
    <link rel="preconnect" href="https://fonts.googleapis.com" />
    <link
      rel="preconnect"
      href="https://fonts.gstatic.com"
      crossorigin="anonymous"
    />
    <script
      src="https://cdnjs.cloudflare.com/ajax/libs/iframe-resizer/4.3.1/iframeResizer.contentWindow.min.js"
      async
    ></script>
    <title>app</title>
  </head>
  <body></body>
</html>
"""


def test_served_index_html_has_no_cdnjs_iframe_resizer():
    rewritten = rewrite_html_for_offline(_SAMPLE_INDEX_HTML)
    assert b"cdnjs.cloudflare.com" not in rewritten
    assert b"fonts.googleapis.com" not in rewritten
    assert b"fonts.gstatic.com" not in rewritten
    # Vendored script is inlined
    assert b"iFrame Resizer" in rewritten or b"iframeResizer" in rewritten
    assert b"</head>" in rewritten


def test_vendored_iframe_resizer_file_exists_and_loads():
    js = load_iframe_resizer_js()
    assert "iframe" in js.lower() or "iFrame" in js
    assert len(js) > 1000
