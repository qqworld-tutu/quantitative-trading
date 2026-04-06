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
    body = ["| " + " | ".join(str(value) for value in row) + " |" for row in rows]
    return "\n".join([head, sep] + body)


def write_markdown_named(root, filename, content):
    path = root / "report" / filename
    path.write_text(content, encoding="utf-8")
    return path


def _scale_points(values, left_pad, top_pad, plot_width, plot_height):
    minimum = min(values)
    maximum = max(values)
    spread = maximum - minimum
    if spread == 0:
        spread = 1.0
    points = []
    for idx, value in enumerate(values):
        x = left_pad + idx * plot_width / max(len(values) - 1, 1)
        y = top_pad + plot_height - (value - minimum) * plot_height / spread
        points.append((x, y))
    return points


def _format_tick(value):
    if abs(value) >= 1000:
        return f"{value:.0f}"
    if abs(value) >= 1:
        return f"{value:.2f}"
    return f"{value:.3f}"


def multi_line_svg(path, title, series_map, y_label, x_ticks=None):
    width, height = 1040, 460
    left_pad, right_pad, top_pad, bottom_pad = 120, 250, 55, 55
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]
    all_values = [value for values in series_map.values() for value in values]
    plot_width = width - left_pad - right_pad
    plot_height = height - top_pad - bottom_pad
    points_map = {
        name: _scale_points(values, left_pad, top_pad, plot_width, plot_height)
        for name, values in series_map.items()
    }
    minimum = min(all_values)
    maximum = max(all_values)
    middle = (minimum + maximum) / 2.0
    spread = maximum - minimum
    if spread == 0:
        spread = 1.0
    sample_len = len(next(iter(series_map.values())))
    if x_ticks is None:
        tick_points = [(0, "start"), (sample_len // 2, "mid"), (sample_len - 1, "end")]
    else:
        tick_points = x_ticks
    x_tick_positions = []
    for idx, label in tick_points:
        x = left_pad + idx * plot_width / max(sample_len - 1, 1)
        x_tick_positions.append((x, label))
    y_ticks = [
        (height - bottom_pad, _format_tick(minimum)),
        (height - bottom_pad - (middle - minimum) * plot_height / spread, _format_tick(middle)),
        (top_pad, _format_tick(maximum)),
    ]
    parts = [
        f'<line x1="{left_pad}" y1="{height - bottom_pad}" x2="{left_pad + plot_width}" y2="{height - bottom_pad}" stroke="black" stroke-width="1"/>',
        f'<line x1="{left_pad}" y1="{top_pad}" x2="{left_pad}" y2="{height - bottom_pad}" stroke="black" stroke-width="1"/>',
        f'<text x="{left_pad + plot_width / 2:.1f}" y="{height - 12}" font-size="12" text-anchor="middle">时间</text>',
        f'<text x="32" y="{top_pad + plot_height / 2:.1f}" font-size="12" text-anchor="middle" transform="rotate(-90 32 {top_pad + plot_height / 2:.1f})">{y_label}</text>',
        f'<rect x="{width - right_pad + 20}" y="{top_pad}" width="{right_pad - 35}" height="{max(70, 24 * len(series_map))}" fill="white" stroke="#cccccc" stroke-width="1"/>',
    ]
    for x, label in x_tick_positions:
        parts.append(f'<line x1="{x:.1f}" y1="{height - bottom_pad}" x2="{x:.1f}" y2="{height - bottom_pad + 5}" stroke="black" stroke-width="1"/>')
        parts.append(f'<text x="{x:.1f}" y="{height - bottom_pad + 18}" font-size="10" text-anchor="middle">{label}</text>')
    for y, label in y_ticks:
        parts.append(f'<line x1="{left_pad - 5}" y1="{y:.1f}" x2="{left_pad}" y2="{y:.1f}" stroke="black" stroke-width="1"/>')
        parts.append(f'<text x="{left_pad - 10}" y="{y + 3:.1f}" font-size="10" text-anchor="end">{label}</text>')
    legend_y = top_pad + 20
    for idx, (name, points) in enumerate(points_map.items()):
        color = colors[idx % len(colors)]
        polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{polyline}"/>')
        lx = width - right_pad + 35
        ly = legend_y + idx * 18
        parts.append(f'<line x1="{lx}" y1="{ly}" x2="{lx + 24}" y2="{ly}" stroke="{color}" stroke-width="2"/>')
        parts.append(f'<text x="{lx + 30}" y="{ly + 4}" font-size="11">{name}</text>')
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{left_pad}" y="24" font-size="16">{title}</text>
{''.join(parts)}
</svg>
"""
    path.write_text(svg, encoding="utf-8")
