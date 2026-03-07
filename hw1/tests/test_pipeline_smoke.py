from pathlib import Path

from assignment1_part2.reporting import ensure_dirs, histogram_svg, line_svg, write_markdown


def test_reporting_writes_markdown(tmp_path: Path):
    ensure_dirs(tmp_path)
    report_path = write_markdown(tmp_path, "# Demo\n\nok\n")
    assert report_path.exists()
    assert report_path.read_text(encoding="utf-8").startswith("# Demo")


def test_svg_charts_include_axes(tmp_path: Path):
    line_path = tmp_path / "line.svg"
    hist_path = tmp_path / "hist.svg"

    line_svg(line_path, "Line", [1.0, 2.0, 3.0])
    histogram_svg(hist_path, "Hist", [2, 3, 1], [0.0, 1.0, 2.0, 3.0], [0.0, 1.5, 3.0], [0.1, 0.3, 0.1])

    line_text = line_path.read_text(encoding="utf-8")
    hist_text = hist_path.read_text(encoding="utf-8")

    assert 'class="axis-x"' in line_text
    assert 'class="axis-y"' in line_text
    assert "样本点" in line_text
    assert 'class="axis-x"' in hist_text
    assert 'class="axis-y"' in hist_text
    assert "收益率" in hist_text
