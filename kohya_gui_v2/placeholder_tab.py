import gradio as gr


def placeholder_tab() -> None:
    """Temporary landing tab for kohya_gui_v2 (Move 2 scaffold).

    Replaced by real per-training-type tabs (kohya_gui_v2/tabs/*) as the
    Phase B+ moves in wargame/2026-07-11-musubi-style-gui-v2.md land.
    """
    gr.Markdown(
        "# kohya_gui_v2 (preview)\n\n"
        "This is the parallel, architecture-registry-based rewrite of the "
        "kohya_ss training GUI. It is under active development alongside "
        "the existing GUI (`kohya_gui.py`) and does not yet train models.\n\n"
        "See `wargame/2026-07-11-musubi-style-gui-v2.md` for the plan."
    )
