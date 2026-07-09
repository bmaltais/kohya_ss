// Sets a native `title` attribute (browser tooltip) on the 📂/🔄/💾/📄
// tool buttons based on their icon, since gr.Button has no `info=`-style
// description and these buttons only ever show an emoji as their label.
// A MutationObserver is needed rather than a single pass on load because
// Gradio mounts each tab's content lazily/on demand.
(function () {
    const ICON_TITLES = {
        "\u{1F4C2}": "Browse for a folder", // 📂 folder_symbol
        "\u{1F4C4}": "Browse for a file", // 📄 document_symbol
        "\u{1F504}": "Refresh the list of choices", // 🔄 refresh_symbol
        "\u{1F4BE}": "Choose where to save the file", // 💾 save_style_symbol
    };

    function applyTitle(btn) {
        if (btn.title) return;
        const title = ICON_TITLES[btn.textContent.trim()];
        if (title) btn.title = title;
    }

    function applyTitles(root) {
        if (root.id === "open_folder_small") applyTitle(root);
        root.querySelectorAll("#open_folder_small").forEach(applyTitle);
    }

    applyTitles(document.body);

    new MutationObserver(function (mutations) {
        for (const mutation of mutations) {
            for (const node of mutation.addedNodes) {
                if (node.nodeType === Node.ELEMENT_NODE) applyTitles(node);
            }
        }
    }).observe(document.body, { childList: true, subtree: true });
})();
