from pathlib import Path


def ensure_dirs(root):
    for name in ("tables", "figures", "report"):
        path = root / name
        path.mkdir(parents=True, exist_ok=True)
        for file in path.glob("*"):
            if file.is_file():
                file.unlink()


def write_csv(path, headers, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join(headers)]
    for row in rows:
        lines.append(",".join(str(value) for value in row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def markdown_table(headers, rows):
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in rows]
    return "\n".join([head, sep] + body)


def write_markdown(root, content):
    path = root / "report" / "hw3_report.md"
    path.write_text(content, encoding="utf-8")
    return path


def write_markdown_named(root, filename, content):
    path = root / "report" / filename
    path.write_text(content, encoding="utf-8")
    return path
