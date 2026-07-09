// Positions revealed `info=` hint text as a viewport-fixed tooltip anchored
// to its field, so hovering/focusing one field never grows in place and
// pushes down neighboring rows. See the comment above the matching CSS rule
// in assets/style.css for why this needs to be `position: fixed` computed
// in JS rather than pure CSS `position: absolute` (Accordion overflow:hidden
// clips absolutely-positioned descendants too).
//
// Mouse hover only triggers over the field's *name* text, not its input
// control, so the reveal area matches what the user reads as the field
// label. Keyboard focus triggers on the whole field instead: a keyboard
// user tabs directly into the input control, never the label text, so
// restricting focus the same way as hover would make the tooltip
// unreachable without a mouse.
(function () {
    const VISIBLE_CLASS = "info-tooltip-visible";
    const BLOCK_SELECTOR = ".block";
    // Textbox/Number/Slider/Radio and Dropdown both render the field name
    // in a span carrying this testid (Gradio's naming is misleading here --
    // it's the title text, not the info paragraph). Checkbox has no such
    // span; its own <label> (checkbox + text) doubles as the field name.
    const NAME_SELECTOR = '[data-testid="block-info"], label';

    function getInfoDiv(block) {
        return (
            block.querySelector(":scope > label > .svelte-j9uq24:has(> .prose)") ||
            block.querySelector(
                ":scope > .svelte-1hfxrpf.container > .svelte-j9uq24:has(> .prose)"
            ) ||
            block.querySelector(":scope > .svelte-j9uq24:has(> .prose)")
        );
    }

    function resolveFromName(target) {
        const nameEl = target.closest(NAME_SELECTOR);
        if (!nameEl) return null;
        const block = nameEl.closest(BLOCK_SELECTOR);
        if (!block) return null;
        const infoDiv = getInfoDiv(block);
        if (!infoDiv) return null;
        // A bare <label> match only counts as the field name when it has no
        // dedicated block-info title span of its own (the Checkbox case);
        // otherwise this would fire while hovering the input control area
        // that a label-wrapped field's <label> also happens to contain.
        if (nameEl.matches("label") && nameEl.querySelector('[data-testid="block-info"]')) {
            return null;
        }
        return { anchorEl: nameEl, infoDiv };
    }

    function resolveFromBlock(target) {
        const block = target.closest(BLOCK_SELECTOR);
        if (!block) return null;
        const infoDiv = getInfoDiv(block);
        if (!infoDiv) return null;
        return { anchorEl: block, infoDiv };
    }

    function showTooltip(anchorEl, infoDiv) {
        const rect = anchorEl.getBoundingClientRect();
        infoDiv.style.top = rect.bottom + 4 + "px";
        infoDiv.style.left = rect.left + "px";
        infoDiv.classList.add(VISIBLE_CLASS);
    }

    function hideTooltip(infoDiv) {
        infoDiv.classList.remove(VISIBLE_CLASS);
    }

    document.addEventListener(
        "mouseover",
        function (e) {
            const resolved = resolveFromName(e.target);
            if (resolved) showTooltip(resolved.anchorEl, resolved.infoDiv);
        },
        true
    );
    document.addEventListener(
        "mouseout",
        function (e) {
            const resolved = resolveFromName(e.target);
            if (resolved && !resolved.anchorEl.contains(e.relatedTarget)) {
                hideTooltip(resolved.infoDiv);
            }
        },
        true
    );
    document.addEventListener(
        "focusin",
        function (e) {
            const resolved = resolveFromBlock(e.target);
            if (resolved) showTooltip(resolved.anchorEl, resolved.infoDiv);
        },
        true
    );
    document.addEventListener(
        "focusout",
        function (e) {
            const resolved = resolveFromBlock(e.target);
            if (resolved && !resolved.anchorEl.contains(e.relatedTarget)) {
                hideTooltip(resolved.infoDiv);
            }
        },
        true
    );
})();
