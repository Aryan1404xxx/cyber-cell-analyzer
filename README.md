# 🔍 Cyber Cell Subject Analyzer

An AI-powered web app to automatically detect **duplicate and similar subjects** across LEA (Law Enforcement Agency) cyber crime cases — built to help investigators identify overlapping cases faster.

🔗 **Live App:** [cyber-cell-analyzer-7kvit3y9dvva6suqconjkl.streamlit.app](https://cyber-cell-analyzer-7kvit3y9dvva6suqconjkl.streamlit.app)

---

## 📌 What it does

- Upload any Excel file of cyber cell cases
- Automatically detects **exact duplicate subjects** (word-for-word same)
- Detects **similar subjects** using fuzzy matching (same topic, slightly different wording)
- **Search** any case by keyword or Case ID — instantly see if it's a duplicate
- Download a full **Excel report** with 3 sheets: Exact Duplicates, Similar Groups, Summary

---

## 🖥️ Screenshots

| Upload & Analyze | Exact Duplicates | Similar Groups |
|---|---|---|
| Upload your Excel file and get instant results | Cases with identical subjects grouped together | Fuzzy matched cases grouped by similarity |

---

## 📋 Expected Input Format

Your Excel file should have these columns:

| Case ID | LEA Name | LEA Email | Subject | Mail Date | Status |
|---|---|---|---|---|---|
| 31626 | Cyber Crime Cell | acp@delhi.gov.in | Court order compliance FIR 18/26 | 01/01/2026 | Pending |

---

## ⚙️ How it works

### Exact Matching
Groups cases where the `Subject` field is 100% identical using pandas groupby deduplication.

### Fuzzy Matching
Uses **RapidFuzz** (`token_sort_ratio`) to compare all unique subjects pairwise. Subjects scoring above the similarity threshold (default 65%) are grouped together. This catches cases like:
- `"Fwd: Re: Court order FIR 18/26"` and `"Court order compliance FIR 18/26"` being the same underlying case

### Search
Full-text search across all Case IDs and subjects with instant badge labeling (Exact Duplicate / Similar Match / Unique).

---

## 🚀 Run Locally

```bash
git clone https://github.com/Aryan1404xxx/cyber-cell-analyzer.git
cd cyber-cell-analyzer
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

---

## 📦 Tech Stack

- **Streamlit** — web app framework
- **Pandas** — data processing
- **RapidFuzz** — fast fuzzy string matching
- **OpenPyXL** — Excel report generation

---

## 👨‍💻 Developer

**Aryan Sinha**  
Computer Science Student  
Built as a utility tool for cyber cell case management.
