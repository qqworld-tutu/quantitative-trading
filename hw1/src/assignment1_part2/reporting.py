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


def _scale_points(values, width, height, pad):
    minimum = min(values)
    maximum = max(values)
    if maximum == minimum:
        maximum = minimum + 1.0
    points = []
    for idx, value in enumerate(values):
        x = pad + idx * (width - 2 * pad) / max(len(values) - 1, 1)
        y = height - pad - (value - minimum) * (height - 2 * pad) / (maximum - minimum)
        points.append((x, y))
    return points


def line_svg(path, title, values):
    width, height, pad = 900, 360, 30
    points = _scale_points(values, width, height, pad) if values else []
    polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{pad}" y="20" font-size="16">{title}</text>
<polyline fill="none" stroke="#1f77b4" stroke-width="2" points="{polyline}"/>
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def histogram_svg(path, title, counts, edges, normal_x=None, normal_y=None):
    width, height, pad = 900, 360, 30
    max_count = max(counts) if counts else 1
    bars = []
    for idx, count in enumerate(counts):
        bar_width = (width - 2 * pad) / max(len(counts), 1)
        bar_height = (count / max_count) * (height - 2 * pad)
        x = pad + idx * bar_width
        y = height - pad - bar_height
        bars.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width - 1:.1f}" height="{bar_height:.1f}" fill="#9ecae1"/>')
    line = ""
    if normal_x and normal_y and max(normal_y) > 0:
        points = []
        left = edges[0]
        right = edges[-1]
        for x, y in zip(normal_x, normal_y):
            scaled_x = pad + (x - left) * (width - 2 * pad) / max(right - left, 1e-9)
            scaled_y = height - pad - y * (height - 2 * pad) / max(normal_y)
            points.append((scaled_x, scaled_y))
        polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        line = f'<polyline fill="none" stroke="#d62728" stroke-width="2" points="{polyline}"/>'
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{pad}" y="20" font-size="16">{title}</text>
{''.join(bars)}
{line}
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def markdown_table(headers, rows):
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(str(value) for value in row) + " |" for row in rows]
    return "\n".join([head, sep] + body)


def write_markdown(root, content):
    report_path = root / "report" / "assignment1_part2_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(content, encoding="utf-8")
    return report_path
