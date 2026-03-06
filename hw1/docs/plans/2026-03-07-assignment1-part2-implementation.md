# Assignment 1 Part 2 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a readable Python pipeline that completes assignment 1 part 2 basic analyses and broad advanced analyses, then exports figures, tables, and a markdown report draft.

**Architecture:** Use a small package with separate files for loading data, computing statistics, advanced analyses, plotting, and report writing. Keep functions plain and data-oriented so the code stays easy to read.

**Tech Stack:** Python, pandas, openpyxl, matplotlib, scipy, pytest

---

### Task 1: Create project skeleton

**Files:**
- Create: `pyproject.toml`
- Create: `src/assignment1_part2/__init__.py`
- Create: `src/assignment1_part2/config.py`
- Create: `tests/test_config.py`

**Step 1: Write the failing test**

Write a test that imports output and data paths from `config.py`.

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python3 -m pytest tests/test_config.py -q`
Expected: FAIL because module does not exist.

**Step 3: Write minimal implementation**

Add fixed `Path` objects for zip file and output directories.

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python3 -m pytest tests/test_config.py -q`
Expected: PASS.

### Task 2: Implement data loading

**Files:**
- Create: `src/assignment1_part2/data.py`
- Create: `tests/test_data.py`

**Step 1: Write the failing test**

Test date conversion from Excel serial numbers and loading one representative sheet.

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python3 -m pytest tests/test_data.py -q`
Expected: FAIL because loader is missing.

**Step 3: Write minimal implementation**

Implement zip extraction, workbook loading, date parsing, and basic normalization.

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python3 -m pytest tests/test_data.py -q`
Expected: PASS.

### Task 3: Implement core metrics

**Files:**
- Create: `src/assignment1_part2/metrics.py`
- Create: `tests/test_metrics.py`

**Step 1: Write the failing test**

Test log-return calculation, summary statistics, and lag-1 autocorrelation.

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python3 -m pytest tests/test_metrics.py -q`
Expected: FAIL.

**Step 3: Write minimal implementation**

Add return, distribution, extrema, and correlation helpers.

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python3 -m pytest tests/test_metrics.py -q`
Expected: PASS.

### Task 4: Implement advanced analyses

**Files:**
- Create: `src/assignment1_part2/advanced.py`
- Create: `tests/test_advanced.py`

**Step 1: Write the failing test**

Test weekly/monthly resampling, intraday session split, and equal-volume binning.

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python3 -m pytest tests/test_advanced.py -q`
Expected: FAIL.

**Step 3: Write minimal implementation**

Implement reusable advanced-analysis helpers.

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python3 -m pytest tests/test_advanced.py -q`
Expected: PASS.

### Task 5: Implement plotting and report pipeline

**Files:**
- Create: `src/assignment1_part2/reporting.py`
- Create: `src/assignment1_part2/pipeline.py`
- Create: `run_assignment1_part2.py`
- Create: `tests/test_pipeline_smoke.py`

**Step 1: Write the failing test**

Test that the pipeline can write at least one table, one figure placeholder path, and one markdown file into a temporary output directory.

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python3 -m pytest tests/test_pipeline_smoke.py -q`
Expected: FAIL.

**Step 3: Write minimal implementation**

Implement the end-to-end runner and markdown report writer.

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python3 -m pytest tests/test_pipeline_smoke.py -q`
Expected: PASS.

### Task 6: Run full verification

**Files:**
- Modify: `outputs/**`

**Step 1: Run test suite**

Run: `PYTHONPATH=src python3 -m pytest tests -q`

**Step 2: Run full pipeline**

Run: `PYTHONPATH=src python3 run_assignment1_part2.py`

**Step 3: Inspect deliverables**

Check markdown report, figures, and tables under `outputs/`.
