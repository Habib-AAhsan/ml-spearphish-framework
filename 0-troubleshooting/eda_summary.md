# 📊 EDA Plotly Issue - Fix Summary

### Problem
- Plotly `fig.show()` raised `ValueError: nbformat>=4.2.0 not installed`, despite `nbformat` being present.

### Root Cause
- `ipython` or `ipykernel` was missing, so Plotly couldn’t render interactive charts inside Jupyter Notebook.

### Fix Steps
```bash
pip install ipython ipykernel nbformat jupyterlab plotly --upgrade
python -m ipykernel install --user --name=venv --display-name "venv"
