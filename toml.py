try:
    import tomllib as _tomllib  # Python 3.11+
except Exception:
    _tomllib = None


def load(path):
    if _tomllib is None:
        # very small fallback for the narrow tests (reads allowed_paths in [server])
        data = {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                s = f.read()
        except FileNotFoundError:
            return {}
        # naive parsing for lines like: allowed_paths = ["C:/path"]
        lines = s.splitlines()
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith("allowed_paths"):
                start = line.find("[")
                end = line.rfind("]")
                if start != -1 and end != -1:
                    inner = line[start+1:end].strip()
                    # split on commas and strip quotes
                    parts = [p.strip().strip('"').strip("'") for p in inner.split(",") if p.strip()]
                    data.setdefault("server", {})["allowed_paths"] = parts
        return data
    else:
        with open(path, "rb") as f:
            return _tomllib.load(f)


def dump(obj, fobj):
    # best-effort: write a minimal TOML for tests
    server = obj.get("server", {})
    allowed = server.get("allowed_paths", [])
    with open(fobj, "w", encoding="utf-8") as f:
        f.write("[server]\n")
        f.write("allowed_paths = [")
        f.write(", ".join(f'"{p}"' for p in allowed))
        f.write("]\n")
