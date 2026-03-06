from pathlib import Path

from assignment1_part2.config import DATA_ZIP, FIGURES_DIR, REPORT_DIR, TABLES_DIR


def test_output_paths_are_under_outputs():
    assert DATA_ZIP.name.endswith(".zip")
    assert TABLES_DIR == Path("outputs/tables")
    assert FIGURES_DIR == Path("outputs/figures")
    assert REPORT_DIR == Path("outputs/report")

