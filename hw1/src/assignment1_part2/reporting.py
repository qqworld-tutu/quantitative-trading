from pathlib import Path


def ensure_dirs(root):
    for name in ("tables", "figures", "report"):
        path = root / name
        path.mkdir(parents=True, exist_ok=True)
        for file in path.glob("*"):
            if file.is_file():
                file.unlink()


def _axis_svg(width, height, pad, x_label, y_label, x_ticks, y_ticks):
    parts = [
        f'<line class="axis-x" x1="{pad}" y1="{height - pad}" x2="{width - pad}" y2="{height - pad}" stroke="black" stroke-width="1"/>',
        f'<line class="axis-y" x1="{pad}" y1="{pad}" x2="{pad}" y2="{height - pad}" stroke="black" stroke-width="1"/>',
        f'<text x="{width / 2:.1f}" y="{height - 5}" font-size="12" text-anchor="middle">{x_label}</text>',
        f'<text x="14" y="{height / 2:.1f}" font-size="12" text-anchor="middle" transform="rotate(-90 14 {height / 2:.1f})">{y_label}</text>',
    ]
    for x, label in x_ticks:
        parts.append(f'<line x1="{x:.1f}" y1="{height - pad}" x2="{x:.1f}" y2="{height - pad + 5}" stroke="black" stroke-width="1"/>')
        parts.append(f'<text x="{x:.1f}" y="{height - pad + 18}" font-size="10" text-anchor="middle">{label}</text>')
    for y, label in y_ticks:
        parts.append(f'<line x1="{pad - 5}" y1="{y:.1f}" x2="{pad}" y2="{y:.1f}" stroke="black" stroke-width="1"/>')
        parts.append(f'<text x="{pad - 8}" y="{y + 3:.1f}" font-size="10" text-anchor="end">{label}</text>')
    return "\n".join(parts)


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
    width, height, pad = 900, 360, 45
    points = _scale_points(values, width, height, pad) if values else []
    polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    x_ticks = []
    if values:
        for ratio in (0.0, 0.5, 1.0):
            idx = int((len(values) - 1) * ratio)
            x = pad + idx * (width - 2 * pad) / max(len(values) - 1, 1)
            x_ticks.append((x, str(idx + 1)))
        min_v = min(values)
        max_v = max(values)
        mid_v = (min_v + max_v) / 2
        y_ticks = [
            (height - pad, f"{min_v:.2f}"),
            (height - pad - (mid_v - min_v) * (height - 2 * pad) / max(max_v - min_v, 1.0), f"{mid_v:.2f}"),
            (pad, f"{max_v:.2f}"),
        ]
    else:
        y_ticks = []
    axes = _axis_svg(width, height, pad, "样本点", "价格", x_ticks, y_ticks)
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{pad}" y="20" font-size="16">{title}</text>
{axes}
<polyline fill="none" stroke="#1f77b4" stroke-width="2" points="{polyline}"/>
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def histogram_svg(path, title, counts, edges, normal_x=None, normal_y=None):
    width, height, pad = 900, 360, 45
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
    x_ticks = []
    if edges:
        left = edges[0]
        right = edges[-1]
        for ratio in (0.0, 0.5, 1.0):
            value = left + (right - left) * ratio
            x = pad + (value - left) * (width - 2 * pad) / max(right - left, 1e-9)
            x_ticks.append((x, f"{value:.3f}"))
    y_ticks = [
        (height - pad, "0"),
        (height - pad - (height - 2 * pad) / 2, str(max_count // 2)),
        (pad, str(max_count)),
    ]
    axes = _axis_svg(width, height, pad, "收益率", "频数", x_ticks, y_ticks)
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{pad}" y="20" font-size="16">{title}</text>
{axes}
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
