# HW10

This assignment analyzes A-share order-by-order data.

Put the course data file at `hw10/tickdata.zip` first. The data file is not
tracked by Git because it is larger than GitHub's normal single-file limit.

Run the analysis from the repository root:

```bash
python hw10/run_hw10.py
```

The script reads `tickdata.zip`, counts limit and market orders, estimates
stock-level order activity features, fits inter-arrival-time distributions, and
writes outputs to:

- `hw10/outputs/tables/`
- `hw10/outputs/figures/`
- `hw10/submit/hw10_report.md`
