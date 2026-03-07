from pathlib import Path

from assignment1_part2.config import DATA_ZIP, FIGURES_DIR, REPORT_DIR, ROOT, TABLES_DIR


def test_output_paths_are_under_outputs():
    assert DATA_ZIP.name.endswith(".zip")
    assert ROOT.name == "hw1"
    assert TABLES_DIR == ROOT / "outputs" / "tables"
    assert FIGURES_DIR == ROOT / "outputs" / "figures"
    assert REPORT_DIR == ROOT / "outputs" / "report"
