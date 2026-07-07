import streamlit as st
import pandas as pd
import numpy as np
from rapidfuzz import fuzz
import openpyxl
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
import io

st.set_page_config(page_title="Cyber Cell Subject Analyzer", page_icon="🔍", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background-color: #F7F8FA; }

.main-header {
    background: linear-gradient(135deg, #1a3a5c, #2563a8);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    color: white;
}
.main-header h1 { color: white; font-size: 2rem; font-weight: 700; margin: 0; }
.main-header p  { color: rgba(255,255,255,0.75); margin: 0.3rem 0 0; font-size: 0.95rem; }

.metric-card {
    background: white;
    border: 1px solid #E5E7EB;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.metric-num  { font-size: 2rem; font-weight: 700; color: #1a3a5c; }
.metric-label{ font-size: 0.8rem; color: #6B7280; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.05em; }

.section-header {
    font-size: 1rem;
    font-weight: 600;
    color: #1a3a5c;
    border-left: 4px solid #2563a8;
    padding-left: 10px;
    margin: 1.5rem 0 1rem;
}

.badge-exact  { background:#FEE2E2; color:#DC2626; padding:3px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; }
.badge-similar{ background:#FEF3C7; color:#D97706; padding:3px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; }
.badge-unique { background:#D1FAE5; color:#059669; padding:3px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; }

.result-card {
    background: white;
    border: 1px solid #E5E7EB;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.result-card:hover { border-color: #2563a8; }

section[data-testid="stSidebar"] {
    background: #1a3a5c;
}
section[data-testid="stSidebar"] * { color: white !important; }
section[data-testid="stSidebar"] .stSlider > div { color: white !important; }

.stButton > button {
    background: #2563a8;
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.5rem 1.5rem;
}
.stButton > button:hover { background: #1a3a5c; color: white; }

div[data-testid="stMetricValue"] { color: #1a3a5c !important; font-weight: 700 !important; }

.stTabs [aria-selected="true"] {
    color: #2563a8 !important;
    border-bottom: 3px solid #2563a8 !important;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🔍 Cyber Cell Subject Analyzer</h1>
    <p>Upload LEA case data to automatically detect duplicate and similar subjects across cases</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    similarity_threshold = st.slider("Similarity threshold (%)", 50, 95, 65,
        help="How similar two subjects need to be to be grouped together")
    st.markdown("---")
    st.markdown("**How it works**")
    st.markdown("🔴 **Exact** — word for word same subject")
    st.markdown("🟡 **Similar** — same topic, slightly different wording")
    st.markdown("🟢 **Unique** — no duplicates found")
    st.markdown("---")
    st.markdown("**Tips**")
    st.markdown("• Lower threshold = more groups")
    st.markdown("• Higher threshold = stricter matching")
    st.markdown("• 65% works well for legal subjects")

@st.cache_data
def process_file(file_bytes, threshold):
    df = pd.read_excel(io.BytesIO(file_bytes))
    df.columns = [c.strip() for c in df.columns]

    required = ['Case ID', 'Subject']
    missing = [c for c in required if c not in df.columns]
    if missing:
        return None, None, None, None, f"Missing columns: {missing}"

    df['Case ID'] = df['Case ID'].astype(str).str.strip()
    df['Subject'] = df['Subject'].astype(str).str.strip()

    # Exact duplicates
    exact_mask = df.duplicated(subset='Subject', keep=False)
    exact_df = df[exact_mask].copy().sort_values('Subject').reset_index(drop=True)

    exact_group_ids = {}
    for subj, grp in exact_df.groupby('Subject'):
        ids = ', '.join(grp['Case ID'].tolist())
        for i in grp.index:
            exact_group_ids[i] = ids
    exact_df['Matching Case IDs'] = exact_df.index.map(exact_group_ids)

    # Fuzzy matching on unique subjects
    unique_df = df.drop_duplicates(subset='Subject').reset_index(drop=True)
    subjects = unique_df['Subject'].tolist()

    fuzzy_groups = []
    visited = set()
    for i in range(len(subjects)):
        if i in visited:
            continue
        group = [i]
        for j in range(i+1, len(subjects)):
            if j in visited:
                continue
            score = fuzz.token_sort_ratio(subjects[i], subjects[j])
            if score >= threshold:
                group.append(j)
                visited.add(j)
        if len(group) > 1:
            fuzzy_groups.append(group)
        visited.add(i)

    fuzzy_rows = []
    for g_num, group in enumerate(fuzzy_groups, 1):
        for idx in group:
            subj = subjects[idx]
            matches = df[df['Subject'] == subj]
            for _, row in matches.iterrows():
                fuzzy_rows.append({
                    'Group': g_num,
                    'Case ID': row['Case ID'],
                    'LEA Name': row.get('LEA Name', ''),
                    'Subject': row['Subject'],
                })
    fuzzy_df = pd.DataFrame(fuzzy_rows) if fuzzy_rows else pd.DataFrame()

    return df, exact_df, fuzzy_df, fuzzy_groups, None

def build_excel(df, exact_df, fuzzy_df, fuzzy_groups):
    wb = openpyxl.Workbook()

    RED    = PatternFill("solid", fgColor="DC2626")
    ORANGE = PatternFill("solid", fgColor="D97706")
    BLUE   = PatternFill("solid", fgColor="1a3a5c")
    ALT1   = PatternFill("solid", fgColor="FEF2F2")
    ALT2   = PatternFill("solid", fgColor="FFFBEB")
    WHITE  = PatternFill("solid", fgColor="FFFFFF")
    GRAY   = PatternFill("solid", fgColor="F9FAFB")

    hfont  = Font(name='Arial', bold=True, color='FFFFFF', size=10)
    dfont  = Font(name='Arial', size=10)
    tfont  = Font(name='Arial', bold=True, color='FFFFFF', size=13)
    border = Border(*[Side(style='thin', color='E5E7EB')]*4)

    def hrow(ws, r, fill, n):
        for c in range(1, n+1):
            cell = ws.cell(row=r, column=c)
            cell.fill = fill; cell.font = hfont
            cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
            cell.border = border

    def drow(ws, r, fill, n):
        for c in range(1, n+1):
            cell = ws.cell(row=r, column=c)
            cell.fill = fill; cell.font = dfont
            cell.alignment = Alignment(vertical='center', wrap_text=True)
            cell.border = border

    # Sheet 1 - Exact
    ws1 = wb.active; ws1.title = "Exact Duplicates"
    ws1.merge_cells('A1:G1')
    ws1['A1'] = f'EXACT DUPLICATE SUBJECTS — {len(exact_df)} cases'
    ws1['A1'].fill = RED; ws1['A1'].font = tfont
    ws1['A1'].alignment = Alignment(horizontal='center', vertical='center')
    ws1.row_dimensions[1].height = 28

    cols1 = ['Case ID','LEA Name','LEA Email','Subject','Mail Date','Status','Matching Case IDs']
    for c, h in enumerate(cols1, 1):
        ws1.cell(row=2, column=c, value=h)
    hrow(ws1, 2, RED, len(cols1))

    prev = None; tog = True
    for i, (_, row) in enumerate(exact_df.iterrows(), 3):
        if row['Subject'] != prev:
            tog = not tog; prev = row['Subject']
        fill = ALT1 if tog else WHITE
        for c, col in enumerate(cols1, 1):
            ws1.cell(row=i, column=c, value=row.get(col, ''))
        drow(ws1, i, fill, len(cols1))
        ws1.row_dimensions[i].height = 32

    for col, w in zip(['A','B','C','D','E','F','G'], [12,26,30,58,14,12,28]):
        ws1.column_dimensions[col].width = w
    ws1.freeze_panes = 'A3'

    # Sheet 2 - Fuzzy
    ws2 = wb.create_sheet("Similar Subjects")
    ws2.merge_cells('A1:D1')
    ws2['A1'] = f'SIMILAR SUBJECTS — {len(fuzzy_groups)} groups found'
    ws2['A1'].fill = ORANGE; ws2['A1'].font = tfont
    ws2['A1'].alignment = Alignment(horizontal='center', vertical='center')
    ws2.row_dimensions[1].height = 28

    cols2 = ['Group','Case ID','LEA Name','Subject']
    for c, h in enumerate(cols2, 1):
        ws2.cell(row=2, column=c, value=h)
    hrow(ws2, 2, ORANGE, len(cols2))

    if not fuzzy_df.empty:
        prev2 = None; tog2 = True
        for i, (_, row) in enumerate(fuzzy_df.iterrows(), 3):
            if row['Group'] != prev2:
                tog2 = not tog2; prev2 = row['Group']
            fill = ALT2 if tog2 else WHITE
            vals = [f"Group {int(row['Group'])}", row['Case ID'], row['LEA Name'], row['Subject']]
            for c, v in enumerate(vals, 1):
                ws2.cell(row=i, column=c, value=v)
            drow(ws2, i, fill, len(cols2))
            ws2.row_dimensions[i].height = 32

    for col, w in zip(['A','B','C','D'], [14,12,26,70]):
        ws2.column_dimensions[col].width = w
    ws2.freeze_panes = 'A3'

    # Sheet 3 - Summary
    ws3 = wb.create_sheet("Summary")
    ws3.merge_cells('A1:B1')
    ws3['A1'] = 'ANALYSIS SUMMARY'
    ws3['A1'].fill = BLUE; ws3['A1'].font = tfont
    ws3['A1'].alignment = Alignment(horizontal='center', vertical='center')
    ws3.row_dimensions[1].height = 28

    rows = [
        ('Total Cases Analyzed', len(df)),
        ('Total Unique Subjects', df['Subject'].nunique()),
        ('Cases with Exact Duplicate Subjects', len(exact_df)),
        ('Unique Subjects Duplicated', exact_df['Subject'].nunique() if not exact_df.empty else 0),
        ('Similar Subject Groups Found', len(fuzzy_groups)),
        ('Cases in Similar Groups', len(fuzzy_df) if not fuzzy_df.empty else 0),
    ]
    for i, (label, val) in enumerate(rows, 2):
        ws3.cell(row=i, column=1, value=label).font = Font(name='Arial', bold=True, size=11)
        ws3.cell(row=i, column=2, value=val).font = Font(name='Arial', size=11)
        fill = GRAY if i % 2 == 0 else WHITE
        for c in [1,2]:
            ws3.cell(row=i, column=c).fill = fill
            ws3.cell(row=i, column=c).border = border
            ws3.cell(row=i, column=c).alignment = Alignment(vertical='center')
        ws3.row_dimensions[i].height = 22

    ws3.column_dimensions['A'].width = 40
    ws3.column_dimensions['B'].width = 16

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf

# ---- MAIN UI ----
uploaded = st.file_uploader("📂 Upload Excel file (.xlsx)", type=["xlsx"],
    help="File should have Case ID, LEA Name, LEA Email, Subject columns")

if uploaded is None:
    st.info("👆 Upload an Excel file to get started")
    st.markdown("### 📋 Required Columns")
    sample = pd.DataFrame({
        'Case ID': ['31626', '194560', '287803'],
        'LEA Name': ['Cyber Crime Cell', 'CBI', 'Cyber Crime Cell'],
        'LEA Email': ['acp@delhi.gov.in', 'acp@delhi.gov.in', 'acp@delhi.gov.in'],
        'Subject': ['Court order compliance FIR 18/26', 'Court order compliance FIR 18/26', 'Release of frozen funds'],
    })
    st.dataframe(sample, use_container_width=True)

else:
    file_bytes = uploaded.read()
    with st.spinner("🔍 Analyzing subjects..."):
        df, exact_df, fuzzy_df, fuzzy_groups, error = process_file(file_bytes, similarity_threshold)

    if error:
        st.error(f"❌ {error}")
    else:
        high = len(exact_df)
        sim  = len(fuzzy_df) if not fuzzy_df.empty else 0
        total = len(df)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="metric-card"><div class="metric-num">{total}</div><div class="metric-label">Total Cases</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card"><div class="metric-num">{df["Subject"].nunique()}</div><div class="metric-label">Unique Subjects</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-card"><div class="metric-num" style="color:#DC2626">{high}</div><div class="metric-label">Exact Duplicates</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="metric-card"><div class="metric-num" style="color:#D97706">{len(fuzzy_groups)}</div><div class="metric-label">Similar Groups</div></div>', unsafe_allow_html=True)

        st.markdown("---")

        tab1, tab2, tab3 = st.tabs(["🔴 Exact Duplicates", "🟡 Similar Subjects", "🔎 Search"])

        with tab1:
            st.markdown(f'<div class="section-header">Exact Duplicate Subjects — {len(exact_df)} cases</div>', unsafe_allow_html=True)
            if exact_df.empty:
                st.success("No exact duplicates found!")
            else:
                prev = None
                for subj, grp in exact_df.groupby('Subject'):
                    case_ids = ', '.join(grp['Case ID'].tolist())
                    with st.expander(f"🔴 [{len(grp)} cases] {subj[:80]}{'...' if len(subj)>80 else ''}"):
                        st.markdown(f"**Full Subject:** {subj}")
                        st.markdown(f"**Case IDs:** `{case_ids}`")
                        show_cols = [c for c in ['Case ID','LEA Name','LEA Email','Mail Date','Status'] if c in grp.columns]
                        st.dataframe(grp[show_cols].reset_index(drop=True), use_container_width=True)

        with tab2:
            st.markdown(f'<div class="section-header">Similar Subject Groups — {len(fuzzy_groups)} groups</div>', unsafe_allow_html=True)
            if fuzzy_df.empty:
                st.success("No similar subjects found!")
            else:
                for g_num in fuzzy_df['Group'].unique():
                    grp = fuzzy_df[fuzzy_df['Group'] == g_num]
                    preview = grp['Subject'].iloc[0]
                    with st.expander(f"🟡 Group {int(g_num)} — {len(grp)} cases — {preview[:70]}{'...' if len(preview)>70 else ''}"):
                        st.markdown(f"**{len(grp)} cases share similar subjects:**")
                        for _, row in grp.iterrows():
                            st.markdown(f"• `{row['Case ID']}` — {row['LEA Name']} — {row['Subject'][:100]}")

        with tab3:
            st.markdown('<div class="section-header">Search by Subject or Case ID</div>', unsafe_allow_html=True)
            query = st.text_input("🔎 Type a subject keyword or Case ID", placeholder="e.g. FIR 18/26 or 287803")

            if query:
                q = query.strip().lower()
                results = df[
                    df['Subject'].str.lower().str.contains(q, na=False) |
                    df['Case ID'].str.lower().str.contains(q, na=False)
                ]

                if results.empty:
                    st.warning("No matching cases found.")
                else:
                    st.success(f"Found {len(results)} matching cases")
                    for _, row in results.iterrows():
                        is_exact = row['Case ID'] in exact_df['Case ID'].values if not exact_df.empty else False
                        is_sim   = not fuzzy_df.empty and row['Case ID'] in fuzzy_df['Case ID'].values
                        badge = '<span class="badge-exact">Exact Duplicate</span>' if is_exact else \
                                '<span class="badge-similar">Similar Match</span>' if is_sim else \
                                '<span class="badge-unique">Unique</span>'
                        st.markdown(f"""
                        <div class="result-card">
                            <b>Case ID:</b> {row['Case ID']} &nbsp; {badge}<br>
                            <b>LEA:</b> {row.get('LEA Name','')} &nbsp;|&nbsp; <b>Email:</b> {row.get('LEA Email','')}<br>
                            <b>Subject:</b> {row['Subject']}
                        </div>
                        """, unsafe_allow_html=True)

        st.markdown("---")
        excel_buf = build_excel(df, exact_df, fuzzy_df, fuzzy_groups)
        st.download_button(
            "📥 Download Full Report (Excel)",
            excel_buf,
            "cyber_cell_duplicate_analysis.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary"
        )
