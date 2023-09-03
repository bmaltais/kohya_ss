import json
import logging
import os

localizationMap = {}


def load_localizations():
    localizationMap.clear()
    dirname = './localizations'
    for file in os.listdir(dirname):
        fn, ext = os.path.splitext(file)
        if ext.lower() != ".json":
            continue
        localizationMap[fn] = os.path.join(dirname, file)


def load_language_js(language_name: str) -> str:
    fn = localizationMap.get(language_name, None)
    data = {}
    if fn is not None:
        try:
            with open(fn, "r", encoding="utf8") as file:
                data = json.load(file)
        except Exception:
            logging.ERROR(f"Error loading localization from {fn}")

    return f"window.localization = {json.dumps(data)}"


load_localizations()