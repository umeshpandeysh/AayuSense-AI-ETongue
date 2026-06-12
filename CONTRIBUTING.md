# Contributing to AayuSense-AI-ETongue

Thank you for your interest! This guide explains how to contribute.

---

## 🐛 Reporting Bugs
Found a bug? [Open a bug report](../../issues/new?template=bug_report.md). Include:
- A minimal reproducible example
- Your Python version and OS
- The full error traceback

---

## 💡 Suggesting Features
[Open a feature request](../../issues/new?template=feature_request.md).

---

## 🛠️ Development Setup

### 1. Clone
```bash
git clone https://github.com/umeshpandeysh/AayuSense-AI-ETongue.git
cd AayuSense-AI-ETongue
```

### 2. Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Dashboard
```bash
streamlit run dashboard/app.py
```

---

## 🎨 Code Style
- Follow **PEP 8**
- Add **type hints** to all public functions
- Write **NumPy-style docstrings**
- Keep lines under **100 characters**
- Use `pathlib.Path` over `os.path`

---

## 🧪 Running Tests
```bash
pytest tests/ -v
```
All new features must have unit tests. Aim for >80% coverage on new code.

---

## 📬 Pull Request Checklist
- [ ] Code passes `flake8 src/ tests/ --ignore=E501`
- [ ] All public functions have type hints and docstrings
- [ ] New code has unit tests
- [ ] `pytest tests/ -v` passes locally
- [ ] Branch is up to date with `main`
- [ ] PR description explains **what** and **why**
