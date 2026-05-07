# ED318 Statistical Machine Learning Lab Exam

## Files

| File | Description |
|---|---|
| `q1.py` | Q1 — Wine Dataset (LDA, QDA, Logistic Regression) |
| `q2.py` | Q2 — Breast Cancer Dataset (NB, LDA, LR, SVM, PCA) |
| `report.md` | Lab report (convert to PDF before submitting) |
| `output1.txt` | Output of q1.py |
| `output2.txt` | Output of q2.py |

## Requirements

Only **numpy** is needed. No other library.

```bash
pip install numpy
```

## Running

**Question 1:**
```bash
python3 q1.py
```

**Question 2:**
```bash
python3 q2.py
```

**Save output to files:**
```bash
python3 q1.py > output1.txt
python3 q2.py > output2.txt
```

## Report

Convert `report.md` to PDF using pandoc:

```bash
pandoc report.md -o report.pdf
```

Or open `report.md` in VS Code / Cursor and use **Export to PDF**.

## Notes

- Both scripts fetch data automatically from UCI repository (internet required).
- Fill in your **Roll number** and **Name** at the top of `q1.py`, `q2.py`, and `report.md` before submitting.
