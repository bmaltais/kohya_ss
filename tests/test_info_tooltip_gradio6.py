"""Info-tooltip selectors must use Gradio 6's stable .info-text class.

Gradio 5 used version-hashed .svelte-* wrappers; Gradio 6.x renders
info= text as <div class="info-text …"> for Dropdown, Textbox, and
Checkbox. assets/js/info_tooltip.js and assets/style.css rely on that
stable class — pin this so a Gradio bump cannot silently re-break hover.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[1]
_js = (_repo_root / "assets" / "js" / "info_tooltip.js").read_text(encoding="utf-8")
_css = (_repo_root / "assets" / "style.css").read_text(encoding="utf-8")


class TestInfoTooltipGradio6(unittest.TestCase):
    def test_js_getInfoDiv_uses_stable_info_text_class(self) -> None:
        self.assertIn('querySelector(".info-text")', _js)
        # Must not fall back to Gradio-5 hashed scope classes.
        self.assertNotRegex(_js, r"svelte-[a-z0-9]+")

    def test_css_targets_info_text_not_svelte_hash(self) -> None:
        self.assertRegex(_css, r"(?m)^\.info-text\s*\{")
        self.assertRegex(_css, r"(?m)^\.info-text\.info-tooltip-visible\s*\{")
        self.assertNotRegex(_css, r"\.svelte-[a-z0-9]+:has\(>\s*\.prose\)")

    def test_installed_gradio_info_component_exposes_info_text(self) -> None:
        """Guard: the pinned Gradio build really ships .info-text on Info."""
        import gradio

        assets = (
            Path(gradio.__file__).resolve().parent / "templates" / "frontend" / "assets"
        )
        info_js = sorted(assets.glob("Info-*.js"))
        self.assertTrue(info_js, f"no Info-*.js under {assets}")
        body = info_js[0].read_text(encoding="utf-8", errors="replace")
        self.assertRegex(
            body,
            r'class="info-text',
            'Gradio Info component must render class="info-text" '
            "(required by assets/js/info_tooltip.js)",
        )
        # Sanity: no longer the old prose-wrapper pattern as the only selector.
        self.assertIsNotNone(re.search(r"info-text", body))


if __name__ == "__main__":
    unittest.main()
