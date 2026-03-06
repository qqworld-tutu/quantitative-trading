from pathlib import Path

from assignment1_part2.reporting import ensure_dirs, write_markdown


def test_reporting_writes_markdown(tmp_path: Path):
    ensure_dirs(tmp_path)
    report_path = write_markdown(tmp_path, "# Demo\n\nok\n")
    assert report_path.exists()
    assert report_path.read_text(encoding="utf-8").startswith("# Demo")
