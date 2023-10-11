import os
import gradio as gr
import library.localization as localization


def file_path(fn):
    return f'file={os.path.abspath(fn)}?{os.path.getmtime(fn)}'


def js_html_str(language):
    head = f'<script type="text/javascript">{localization.load_language_js(language)}</script>\n'
    head += f'<script type="text/javascript" src="{file_path("js/script.js")}"></script>\n'
    head += f'<script type="text/javascript" src="{file_path("js/localization.js")}"></script>\n'
    return head


def add_javascript(language):
    if language is None:
        # print('no language')
        return
    jsStr = js_html_str(language)

    def template_response(*args, **kwargs):
        res = localization.GrRoutesTemplateResponse(*args, **kwargs)
        res.body = res.body.replace(b'</head>', f'{jsStr}</head>'.encode("utf8"))
        res.init_headers()
        return res

    gr.routes.templates.TemplateResponse = template_response


if not hasattr(localization, 'GrRoutesTemplateResponse'):
    localization.GrRoutesTemplateResponse = gr.routes.templates.TemplateResponse