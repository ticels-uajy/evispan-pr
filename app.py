from __future__ import annotations

import hashlib
import html
import io
import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import streamlit as st

from evispan_model import LABELS, load_artifacts, predict_text


st.set_page_config(
    page_title="EviSpan-PR: Evidence-Grounded Peer Review Feedback Analysis",
    page_icon="🧾",
    layout="wide",
)


LABEL_COLORS = {
    "Problem": "#ffd6d6",
    "Suggestion": "#d9ecff",
    "Neutral": "#eeeeee",
    "Appreciation": "#d9f7df",
}
LABEL_BORDERS = {
    "Problem": "#b42318",
    "Suggestion": "#175cd3",
    "Neutral": "#475467",
    "Appreciation": "#067647",
}
LABEL_DESCRIPTIONS = {
    "Appreciation": (
        "Penilaian evaluatif positif yang secara eksplisit diberikan oleh rekan sejawat "
        "terhadap karya yang dikumpulkan, tanpa mengusulkan perubahan. Tujuannya adalah "
        "memberikan pengakuan atau pujian, misalnya apresiasi, ungkapan kepuasan, atau persetujuan."
    ),
    "Problem": (
        "Identifikasi terhadap kekurangan, kesalahan, ketidakkonsistenan, ambiguitas, "
        "atau kelemahan lain dalam karya yang memerlukan perbaikan atau klarifikasi."
    ),
    "Suggestion": (
        "Rekomendasi yang mengusulkan tindakan konkret dan dapat ditindaklanjuti untuk "
        "memperbaiki karya. Saran disampaikan sebagai nasihat atau usulan, misalnya "
        "‘pertimbangkan…’, ‘Anda dapat…’, atau ‘akan lebih jelas jika…’, bukan sekadar "
        "melaporkan adanya kekurangan."
    ),
    "Neutral": (
        "Teks peer review yang bersifat informatif, tidak menyatakan penilaian evaluatif—"
        "baik positif maupun negatif—dan tidak menyarankan perubahan."
    ),
}

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_ARTIFACT_DIR = Path(
    os.getenv("EVISPAN_ARTIFACT_DIR", str(PROJECT_DIR / "artifacts" / "evispan_pr"))
)
RESPONSE_DB_PATH = Path(
    os.getenv(
        "EVISPAN_RESPONSE_DB",
        str(PROJECT_DIR / "data" / "evispan_lecturer_responses.sqlite3"),
    )
)
TTF_QUESTIONNAIRE_URL = os.getenv(
    "EVISPAN_TTF_URL",
    "https://forms.gle/REPLACE_WITH_YOUR_TTF_FORM_ID",
).strip()
DEFAULT_REVIEW_TARGET = max(1, int(os.getenv("EVISPAN_REVIEW_TARGET", "10")))

TEXT_COLUMN_CANDIDATES = [
    "text",
    "comment",
    "feedback",
    "review",
    "sentence",
    "content",
    "peer_feedback",
]
ID_COLUMN_CANDIDATES = ["id", "sample_id", "row_id", "comment_id", "index"]

STUDY_STATE_KEYS = [
    "study_session_id",
    "study_phase",
    "review_indices",
    "review_position",
    "review_page_jump",
    "study_responses",
    "respondent_code",
    "respondent_name",
    "respondent_unit",
    "review_target",
    "sampling_method",
    "final_response",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@st.cache_data(show_spinner=False)
def read_test_dataset(data: bytes, filename: str) -> pd.DataFrame:
    """Read an uploaded CSV, JSON, or JSONL test dataset."""
    suffix = Path(filename).suffix.lower()

    if suffix == ".csv":
        frame = pd.read_csv(io.BytesIO(data))
    elif suffix == ".json":
        parsed = json.loads(data.decode("utf-8-sig"))
        if isinstance(parsed, list):
            records = parsed
        elif isinstance(parsed, dict):
            records = next(
                (
                    parsed[key]
                    for key in ("records", "data", "items")
                    if isinstance(parsed.get(key), list)
                ),
                [parsed],
            )
        else:
            raise ValueError("JSON harus berisi objek atau daftar objek.")
        frame = pd.json_normalize(records)
    elif suffix in {".jsonl", ".ndjson"}:
        records = [
            json.loads(line)
            for line in data.decode("utf-8-sig").splitlines()
            if line.strip()
        ]
        frame = pd.json_normalize(records)
    else:
        raise ValueError("Format tidak didukung. Gunakan CSV, JSON, JSONL, atau NDJSON.")

    if frame.empty:
        raise ValueError("Test dataset tidak memiliki baris data.")
    if frame.columns.duplicated().any():
        duplicated = frame.columns[frame.columns.duplicated()].tolist()
        raise ValueError(f"Nama kolom duplikat tidak didukung: {duplicated}")

    return frame.reset_index(drop=True)


def preferred_column(
    columns: Iterable[str],
    candidates: List[str],
    fallback_to_first: bool = True,
) -> Optional[str]:
    columns = list(columns)
    lower_to_original = {str(column).lower(): str(column) for column in columns}
    for candidate in candidates:
        if candidate in lower_to_original:
            return lower_to_original[candidate]
    return str(columns[0]) if columns and fallback_to_first else None


def artifact_is_complete(path: Path) -> bool:
    required = [
        path / "artifact_config.json",
        path / "thresholds.json",
        path / "model_state.pt",
        path / "encoder_config" / "config.json",
        path / "tokenizer",
    ]
    return all(item.exists() for item in required)


@st.cache_resource(show_spinner="Memuat model EviSpan-PR...")
def get_live_model(artifact_dir: str):
    return load_artifacts(artifact_dir)


def render_label_guide() -> str:
    cards = []
    for label in ["Appreciation", "Problem", "Suggestion", "Neutral"]:
        color = LABEL_COLORS[label]
        border = LABEL_BORDERS[label]
        description = LABEL_DESCRIPTIONS[label]
        cards.append(
            '<div class="label-guide-card" '
            f'style="border-left:5px solid {border};background:{color};">'
            f'<div class="label-guide-title" style="color:{border};">'
            f"{html.escape(label)}</div>"
            f'<div class="label-guide-description">{html.escape(description)}</div>'
            "</div>"
        )
    return '<div class="label-guide-grid">' + "".join(cards) + "</div>"


def label_badges(labels: Iterable[str]) -> str:
    labels = list(labels)
    if not labels:
        return '<span class="empty-badge">Tidak ada label</span>'

    badges = []
    for label in labels:
        color = LABEL_COLORS.get(label, "#f2f4f7")
        border = LABEL_BORDERS.get(label, "#475467")
        badges.append(
            f'<span class="label-badge" style="background:{color};'
            f'border-color:{border};color:{border}">{html.escape(label)}</span>'
        )
    return " ".join(badges)


def render_plain_text(text: str) -> str:
    escaped = html.escape(text).replace("\n", "<br>")
    return f'<div class="comment-box plain-text">{escaped}</div>'


def render_highlighted_text(text: str, spans: List[Dict[str, Any]]) -> str:
    """Highlight valid, non-overlapping evidence spans using character offsets."""
    safe_spans: List[Dict[str, Any]] = []
    cursor = 0

    for span in sorted(
        spans,
        key=lambda item: (
            int(item.get("start", 0)),
            -(int(item.get("end", 0)) - int(item.get("start", 0))),
        ),
    ):
        try:
            start = max(0, min(int(span["start"]), len(text)))
            end = max(0, min(int(span["end"]), len(text)))
        except (KeyError, TypeError, ValueError):
            continue

        label = str(span.get("label", ""))
        if label not in LABELS or start < cursor or start >= end:
            continue

        safe_spans.append({**span, "start": start, "end": end, "label": label})
        cursor = end

    if not safe_spans:
        return render_plain_text(text)

    parts: List[str] = []
    cursor = 0
    for span in safe_spans:
        start = int(span["start"])
        end = int(span["end"])
        label = str(span["label"])
        parts.append(html.escape(text[cursor:start]))
        parts.append(
            f'<span class="evidence" style="background:{LABEL_COLORS[label]};'
            f'border-bottom:3px solid {LABEL_BORDERS[label]};" '
            f'title="{html.escape(label)} [{start}:{end}]">'
            f"{html.escape(text[start:end])}"
            f"<small>{html.escape(label)}</small></span>"
        )
        cursor = end

    parts.append(html.escape(text[cursor:]))
    return (
        '<div class="comment-box evidence-text">'
        + "".join(parts).replace("\n", "<br>")
        + "</div>"
    )


def spans_dataframe(spans: List[Dict[str, Any]]) -> pd.DataFrame:
    columns = ["Label", "Evidence", "Start", "End"]
    if not spans:
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(
        [
            {
                "Label": span.get("label", ""),
                "Evidence": span.get("text", ""),
                "Start": span.get("start", ""),
                "End": span.get("end", ""),
            }
            for span in spans
        ],
        columns=columns,
    )


def probabilities_dataframe(result: Dict[str, Any]) -> pd.DataFrame:
    probabilities = result.get("label_probabilities", {})
    predicted = set(result.get("predicted_labels", []))
    return pd.DataFrame(
        [
            {
                "Label": label,
                "Probability": float(probabilities.get(label, 0.0)),
                "Predicted": label in predicted,
            }
            for label in LABELS
        ]
    )


def initialise_database(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path, timeout=30) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS study_sessions (
                session_id TEXT PRIMARY KEY,
                respondent_code TEXT NOT NULL,
                respondent_name TEXT,
                respondent_unit TEXT,
                dataset_name TEXT NOT NULL,
                dataset_signature TEXT NOT NULL,
                text_column TEXT NOT NULL,
                id_column TEXT,
                review_target INTEGER NOT NULL,
                sampling_method TEXT NOT NULL,
                review_indices_json TEXT NOT NULL,
                started_at TEXT NOT NULL,
                completed_at TEXT
            );

            CREATE TABLE IF NOT EXISTS comment_responses (
                session_id TEXT NOT NULL,
                source_row_index INTEGER NOT NULL,
                review_order INTEGER NOT NULL,
                row_id TEXT,
                comment_text TEXT NOT NULL,
                predicted_labels_json TEXT NOT NULL,
                label_probabilities_json TEXT NOT NULL,
                predicted_spans_json TEXT NOT NULL,
                label_assessment TEXT NOT NULL,
                expected_labels_json TEXT NOT NULL,
                evidence_support TEXT NOT NULL,
                evidence_boundary TEXT NOT NULL,
                identified_patterns_json TEXT NOT NULL,
                interpretation TEXT,
                confidence INTEGER NOT NULL,
                notes TEXT,
                reviewed_at TEXT NOT NULL,
                PRIMARY KEY (session_id, source_row_index),
                FOREIGN KEY (session_id) REFERENCES study_sessions(session_id)
            );

            CREATE TABLE IF NOT EXISTS final_responses (
                session_id TEXT PRIMARY KEY,
                easiest_label TEXT NOT NULL,
                hardest_label TEXT NOT NULL,
                recurring_patterns TEXT NOT NULL,
                recurring_mismatches TEXT,
                usefulness_rating INTEGER NOT NULL,
                intended_use TEXT,
                overall_notes TEXT,
                submitted_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES study_sessions(session_id)
            );
            """
        )
        connection.commit()


def create_study_session(path: Path, payload: Dict[str, Any]) -> None:
    with sqlite3.connect(path, timeout=30) as connection:
        connection.execute(
            """
            INSERT INTO study_sessions (
                session_id, respondent_code, respondent_name, respondent_unit,
                dataset_name, dataset_signature, text_column, id_column,
                review_target, sampling_method, review_indices_json, started_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["session_id"],
                payload["respondent_code"],
                payload.get("respondent_name", ""),
                payload.get("respondent_unit", ""),
                payload["dataset_name"],
                payload["dataset_signature"],
                payload["text_column"],
                payload.get("id_column", ""),
                int(payload["review_target"]),
                payload["sampling_method"],
                json.dumps(payload["review_indices"]),
                payload["started_at"],
            ),
        )
        connection.commit()


def save_comment_response(path: Path, payload: Dict[str, Any]) -> None:
    with sqlite3.connect(path, timeout=30) as connection:
        connection.execute(
            """
            INSERT INTO comment_responses (
                session_id, source_row_index, review_order, row_id, comment_text,
                predicted_labels_json, label_probabilities_json, predicted_spans_json,
                label_assessment, expected_labels_json, evidence_support,
                evidence_boundary, identified_patterns_json, interpretation,
                confidence, notes, reviewed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id, source_row_index) DO UPDATE SET
                review_order=excluded.review_order,
                row_id=excluded.row_id,
                comment_text=excluded.comment_text,
                predicted_labels_json=excluded.predicted_labels_json,
                label_probabilities_json=excluded.label_probabilities_json,
                predicted_spans_json=excluded.predicted_spans_json,
                label_assessment=excluded.label_assessment,
                expected_labels_json=excluded.expected_labels_json,
                evidence_support=excluded.evidence_support,
                evidence_boundary=excluded.evidence_boundary,
                identified_patterns_json=excluded.identified_patterns_json,
                interpretation=excluded.interpretation,
                confidence=excluded.confidence,
                notes=excluded.notes,
                reviewed_at=excluded.reviewed_at
            """,
            (
                payload["session_id"],
                int(payload["source_row_index"]),
                int(payload["review_order"]),
                str(payload.get("row_id", "")),
                payload["comment_text"],
                json.dumps(payload["predicted_labels"], ensure_ascii=False),
                json.dumps(payload["label_probabilities"], ensure_ascii=False),
                json.dumps(payload["predicted_spans"], ensure_ascii=False),
                payload["label_assessment"],
                json.dumps(payload["expected_labels"], ensure_ascii=False),
                payload["evidence_support"],
                payload["evidence_boundary"],
                json.dumps(payload["identified_patterns"], ensure_ascii=False),
                payload.get("interpretation", ""),
                int(payload["confidence"]),
                payload.get("notes", ""),
                payload["reviewed_at"],
            ),
        )
        connection.commit()


def save_final_response(path: Path, payload: Dict[str, Any]) -> None:
    submitted_at = payload["submitted_at"]
    with sqlite3.connect(path, timeout=30) as connection:
        connection.execute(
            """
            INSERT INTO final_responses (
                session_id, easiest_label, hardest_label, recurring_patterns,
                recurring_mismatches, usefulness_rating, intended_use,
                overall_notes, submitted_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                easiest_label=excluded.easiest_label,
                hardest_label=excluded.hardest_label,
                recurring_patterns=excluded.recurring_patterns,
                recurring_mismatches=excluded.recurring_mismatches,
                usefulness_rating=excluded.usefulness_rating,
                intended_use=excluded.intended_use,
                overall_notes=excluded.overall_notes,
                submitted_at=excluded.submitted_at
            """,
            (
                payload["session_id"],
                payload["easiest_label"],
                payload["hardest_label"],
                payload["recurring_patterns"],
                payload.get("recurring_mismatches", ""),
                int(payload["usefulness_rating"]),
                payload.get("intended_use", ""),
                payload.get("overall_notes", ""),
                submitted_at,
            ),
        )
        connection.execute(
            "UPDATE study_sessions SET completed_at=? WHERE session_id=?",
            (submitted_at, payload["session_id"]),
        )
        connection.commit()


def session_responses_dataframe(path: Path, session_id: str) -> pd.DataFrame:
    with sqlite3.connect(path, timeout=30) as connection:
        return pd.read_sql_query(
            """
            SELECT
                s.session_id,
                s.respondent_code,
                s.respondent_name,
                s.respondent_unit,
                s.dataset_name,
                c.review_order,
                c.source_row_index,
                c.row_id,
                c.comment_text,
                c.predicted_labels_json,
                c.label_probabilities_json,
                c.predicted_spans_json,
                c.label_assessment,
                c.expected_labels_json,
                c.evidence_support,
                c.evidence_boundary,
                c.identified_patterns_json,
                c.interpretation,
                c.confidence,
                c.notes,
                c.reviewed_at
            FROM comment_responses c
            JOIN study_sessions s ON s.session_id = c.session_id
            WHERE c.session_id = ?
            ORDER BY c.review_order
            """,
            connection,
            params=(session_id,),
        )


def session_final_dataframe(path: Path, session_id: str) -> pd.DataFrame:
    with sqlite3.connect(path, timeout=30) as connection:
        return pd.read_sql_query(
            """
            SELECT
                s.session_id,
                s.respondent_code,
                s.respondent_name,
                s.respondent_unit,
                f.easiest_label,
                f.hardest_label,
                f.recurring_patterns,
                f.recurring_mismatches,
                f.usefulness_rating,
                f.intended_use,
                f.overall_notes,
                f.submitted_at
            FROM final_responses f
            JOIN study_sessions s ON s.session_id = f.session_id
            WHERE f.session_id = ?
            """,
            connection,
            params=(session_id,),
        )


def render_workflow(active_phase: str, completed: int, target: int) -> None:
    phases = [
        ("1", "Pahami label", active_phase in {"setup", "review", "final", "complete"}),
        ("2", "Gunakan EviSpan-PR", active_phase in {"review", "final", "complete"}),
        ("3", "Telaah komentar", active_phase in {"review", "final", "complete"}),
        ("4", "Interpretasi & pola", active_phase in {"final", "complete"}),
        ("5", "Kuesioner TTF", active_phase == "complete"),
    ]
    cards = []
    for number, label, reached in phases:
        status_class = "workflow-active" if reached else "workflow-pending"
        cards.append(
            f'<div class="workflow-card {status_class}"><span>{number}</span>{html.escape(label)}</div>'
        )
    st.markdown('<div class="workflow-grid">' + "".join(cards) + "</div>", unsafe_allow_html=True)
    if target:
        st.progress(min(completed / target, 1.0), text=f"Respons komentar tersimpan: {completed}/{target}")


def reset_study_state() -> None:
    for key in STUDY_STATE_KEYS:
        st.session_state.pop(key, None)


def _set_review_position(index: int, total: int) -> None:
    bounded = max(0, min(index, total - 1))
    st.session_state.review_position = bounded
    st.session_state.review_page_jump = bounded + 1


def _previous_review(total: int) -> None:
    _set_review_position(int(st.session_state.review_position) - 1, total)


def _next_review(total: int) -> None:
    _set_review_position(int(st.session_state.review_position) + 1, total)


def _jump_review(total: int) -> None:
    requested = int(st.session_state.review_page_jump) - 1
    st.session_state.review_position = max(0, min(requested, total - 1))


def render_review_navigation(total: int, key_prefix: str) -> None:
    current = int(st.session_state.review_position)
    previous_col, page_col, next_col = st.columns([1, 2, 1])
    previous_col.button(
        "← Previous",
        key=f"{key_prefix}_previous",
        use_container_width=True,
        disabled=current <= 0,
        on_click=_previous_review,
        args=(total,),
    )
    page_col.number_input(
        "Komentar",
        min_value=1,
        max_value=total,
        step=1,
        key="review_page_jump",
        on_change=_jump_review,
        args=(total,),
        label_visibility="collapsed",
    )
    next_col.button(
        "Next →",
        key=f"{key_prefix}_next",
        use_container_width=True,
        disabled=current >= total - 1,
        on_click=_next_review,
        args=(total,),
    )
    st.caption(f"Komentar {current + 1} dari {total}")


def valid_questionnaire_url(url: str) -> bool:
    return url.startswith(("https://", "http://")) and "REPLACE_WITH" not in url


st.markdown(
    """
<style>
.block-container {max-width: 1180px; padding-top: 2rem; padding-bottom: 3rem;}
.label-badge, .empty-badge {
    display:inline-block; padding:0.30rem 0.68rem; margin:0.10rem 0.14rem 0.10rem 0;
    border:1px solid #98a2b3; border-radius:999px; font-weight:700;
}
.empty-badge {background:#f2f4f7; color:#475467;}
.comment-box {
    border:1px solid #d0d5dd; border-radius:12px; padding:1.15rem 1.25rem;
    line-height:2.05; font-size:1.05rem; background:#ffffff; color:#101828;
}
.plain-text {line-height:1.75;}
.evidence {padding:0.12rem 0.20rem; border-radius:5px; position:relative;}
.evidence small {
    font-size:0.62rem; margin-left:0.28rem; padding:0.08rem 0.22rem;
    border-radius:3px; background:rgba(255,255,255,0.80); font-weight:750;
}
.section-label {
    font-size:0.82rem; color:#667085; font-weight:750; text-transform:uppercase;
    letter-spacing:0.035em; margin-bottom:0.35rem;
}
.label-guide-grid {
    display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); gap:0.75rem;
    margin-top:0.35rem; margin-bottom:0.35rem;
}
.label-guide-card {
    border:1px solid rgba(16,24,40,0.10); border-radius:10px;
    padding:0.85rem 0.95rem; min-height:132px;
}
.label-guide-title {font-weight:800; font-size:1rem; margin-bottom:0.35rem;}
.label-guide-description {color:#344054; line-height:1.55; font-size:0.92rem;}
.workflow-grid {
    display:grid; grid-template-columns:repeat(5, minmax(0, 1fr)); gap:0.55rem;
    margin:1rem 0 0.8rem 0;
}
.workflow-card {
    border-radius:10px; padding:0.70rem 0.55rem; text-align:center; font-size:0.84rem;
    border:1px solid #d0d5dd; font-weight:650;
}
.workflow-card span {
    display:inline-flex; width:1.45rem; height:1.45rem; align-items:center; justify-content:center;
    border-radius:999px; margin-right:0.35rem; font-weight:800;
}
.workflow-active {background:#eff8ff; border-color:#84caff; color:#175cd3;}
.workflow-active span {background:#175cd3; color:white;}
.workflow-pending {background:#f9fafb; color:#667085;}
.workflow-pending span {background:#eaecf0; color:#667085;}
.response-panel {
    border:1px solid #d0d5dd; border-radius:12px; padding:1rem 1.1rem; background:#f9fafb;
}
.completion-box {
    border:1px solid #75e0a7; background:#ecfdf3; border-radius:14px;
    padding:1.25rem 1.35rem; margin:1rem 0;
}
@media (max-width: 820px) {
    .label-guide-grid {grid-template-columns:1fr;}
    .label-guide-card {min-height:auto;}
    .workflow-grid {grid-template-columns:1fr;}
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("EviSpan-PR: Evidence-Grounded Multi-Label Peer Review Feedback Analysis")
st.caption(
    "EviSpan-PR membantu dosen menelaah kategori feedback pada tingkat komentar dan "
    "evidence span yang mendukung setiap prediksi. Pada sesi ini, dosen diminta memeriksa "
    "label, probabilitas, serta bukti tekstual, kemudian menyelesaikan tugas interpretasi "
    "dan identifikasi pola sebelum mengisi kuesioner TTF."
)

try:
    initialise_database(RESPONSE_DB_PATH)
    database_ready = True
    database_error = ""
except Exception as exc:
    database_ready = False
    database_error = str(exc)

with st.sidebar:
    st.header("Konfigurasi studi")
    uploaded = st.file_uploader(
        "Upload test dataset",
        type=["csv", "json", "jsonl", "ndjson"],
        help="Dataset harus memiliki minimal satu kolom teks.",
    )
    artifact_dir_text = st.text_input(
        "Model artifact directory",
        value=str(DEFAULT_ARTIFACT_DIR),
        help="Folder berisi model_state.pt, thresholds.json, tokenizer, dan encoder_config.",
    )
    st.caption(f"Penyimpanan respons: `{RESPONSE_DB_PATH}`")
    if not database_ready:
        st.error(f"Database respons tidak dapat digunakan: {database_error}")

if uploaded is None:
    render_workflow("setup", 0, 0)
    with st.expander("1. Pelajari kategori feedback", expanded=True):
        st.markdown(
            "Satu komentar dapat memiliki lebih dari satu label. Baca definisi berikut sebelum "
            "memulai evaluasi."
        )
        st.markdown(render_label_guide(), unsafe_allow_html=True)
    st.info("Upload test dataset melalui panel sebelah kiri untuk memulai sesi evaluasi.")
    st.stop()

try:
    uploaded_bytes = uploaded.getvalue()
    test_df = read_test_dataset(uploaded_bytes, uploaded.name)
except Exception as exc:
    st.error(f"Test dataset tidak dapat dibaca: {exc}")
    st.stop()

available_columns = [str(column) for column in test_df.columns]
default_text_column = preferred_column(available_columns, TEXT_COLUMN_CANDIDATES)
default_text_index = available_columns.index(default_text_column) if default_text_column in available_columns else 0

with st.sidebar:
    text_column = st.selectbox("Kolom teks", options=available_columns, index=default_text_index)

    id_options = ["(gunakan nomor baris)"] + available_columns
    preferred_id = preferred_column(available_columns, ID_COLUMN_CANDIDATES, fallback_to_first=False)
    default_id_index = id_options.index(preferred_id) if preferred_id in id_options else 0
    id_column_selection = st.selectbox("Kolom ID", options=id_options, index=default_id_index)

working_df = test_df.copy()
working_df[text_column] = working_df[text_column].fillna("").astype(str)
working_df = working_df[working_df[text_column].str.strip().ne("")].reset_index(drop=True)
if working_df.empty:
    st.error(f"Kolom `{text_column}` tidak memiliki teks yang dapat diprediksi.")
    st.stop()

artifact_dir = Path(artifact_dir_text).expanduser().resolve()
if not artifact_is_complete(artifact_dir):
    st.error(
        "Model artifact belum lengkap. Pastikan folder berisi `artifact_config.json`, "
        "`thresholds.json`, `model_state.pt`, `encoder_config/config.json`, dan folder `tokenizer`."
    )
    st.code(str(artifact_dir), language="text")
    st.stop()

file_signature = hashlib.sha256(uploaded_bytes).hexdigest()
dataset_signature = f"{file_signature}:{text_column}:{id_column_selection}:{len(working_df)}"
model_state_path = artifact_dir / "model_state.pt"
model_signature = f"{artifact_dir}:{model_state_path.stat().st_mtime_ns}"
study_signature = f"{dataset_signature}:{model_signature}"

if st.session_state.get("active_study_signature") != study_signature:
    reset_study_state()
    st.session_state.active_study_signature = study_signature
    st.session_state.prediction_cache = {}

if "prediction_cache" not in st.session_state:
    st.session_state.prediction_cache = {}
if "study_responses" not in st.session_state:
    st.session_state.study_responses = {}

active_phase = st.session_state.get("study_phase", "setup")
review_target_for_progress = int(st.session_state.get("review_target", 0) or 0)
completed_count = len(st.session_state.get("study_responses", {}))
render_workflow(active_phase, completed_count, review_target_for_progress)

with st.expander("1. Penjelasan kategori Problem, Suggestion, Neutral, dan Appreciation", expanded=active_phase == "setup"):
    st.markdown(
        "Satu komentar dapat memperoleh lebih dari satu label. Gunakan definisi berikut sebagai "
        "acuan saat menilai kesesuaian prediksi dan evidence span."
    )
    st.markdown(render_label_guide(), unsafe_allow_html=True)

if not database_ready:
    st.warning(
        "Respons masih dapat diisi dalam sesi browser, tetapi tidak dapat disimpan permanen karena "
        "database tidak tersedia. Perbaiki lokasi `EVISPAN_RESPONSE_DB` sebelum studi dijalankan."
    )

if "study_session_id" not in st.session_state:
    st.subheader("Mulai sesi evaluasi dosen")
    st.markdown(
        "Isi identitas responden, tentukan jumlah komentar yang akan ditelaah, lalu mulai sesi. "
        "Nama bersifat opsional; kode responden digunakan untuk membedakan data respons."
    )

    max_target = len(working_df)
    default_target = min(DEFAULT_REVIEW_TARGET, max_target)
    with st.form("study_setup_form"):
        respondent_code = st.text_input(
            "Kode responden *",
            placeholder="Contoh: D-001",
            help="Gunakan kode anonim bila penelitian tidak memerlukan nama dosen.",
        )
        respondent_name = st.text_input("Nama dosen (opsional)")
        respondent_unit = st.text_input("Program studi/unit (opsional)")
        review_target = st.number_input(
            "Jumlah komentar yang ditelaah",
            min_value=1,
            max_value=max_target,
            value=default_target,
            step=1,
        )
        sampling_method = st.radio(
            "Pemilihan komentar",
            options=["Acak", "Berurutan dari awal dataset"],
            horizontal=True,
            help="Pemilihan acak menggunakan seed tetap untuk sesi ini sehingga urutan tidak berubah.",
        )
        begin = st.form_submit_button("Mulai menggunakan EviSpan-PR", type="primary")

    if begin:
        if not respondent_code.strip():
            st.error("Kode responden wajib diisi.")
            st.stop()
        if not database_ready:
            st.error("Sesi tidak dapat dimulai sebelum database respons tersedia.")
            st.stop()

        session_id = uuid.uuid4().hex
        target = int(review_target)
        if sampling_method == "Acak":
            seed_material = f"{study_signature}:{session_id}"
            seed = int(hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:8], 16)
            review_indices = working_df.sample(n=target, random_state=seed).index.tolist()
        else:
            review_indices = list(range(target))

        session_payload = {
            "session_id": session_id,
            "respondent_code": respondent_code.strip(),
            "respondent_name": respondent_name.strip(),
            "respondent_unit": respondent_unit.strip(),
            "dataset_name": uploaded.name,
            "dataset_signature": dataset_signature,
            "text_column": text_column,
            "id_column": "" if id_column_selection == "(gunakan nomor baris)" else id_column_selection,
            "review_target": target,
            "sampling_method": sampling_method,
            "review_indices": review_indices,
            "started_at": utc_now_iso(),
        }
        try:
            create_study_session(RESPONSE_DB_PATH, session_payload)
        except Exception as exc:
            st.error(f"Sesi tidak dapat disimpan: {exc}")
            st.stop()

        st.session_state.study_session_id = session_id
        st.session_state.respondent_code = respondent_code.strip()
        st.session_state.respondent_name = respondent_name.strip()
        st.session_state.respondent_unit = respondent_unit.strip()
        st.session_state.review_target = target
        st.session_state.sampling_method = sampling_method
        st.session_state.review_indices = review_indices
        st.session_state.review_position = 0
        st.session_state.review_page_jump = 1
        st.session_state.study_responses = {}
        st.session_state.study_phase = "review"
        st.rerun()

    st.stop()

session_id = st.session_state.study_session_id
review_indices = list(st.session_state.review_indices)
review_target = int(st.session_state.review_target)
responses: Dict[str, Dict[str, Any]] = st.session_state.study_responses
completed_count = len(responses)

with st.sidebar:
    st.divider()
    st.subheader("Sesi aktif")
    st.write(f"**Kode responden:** {st.session_state.respondent_code}")
    st.write(f"**Komentar:** {review_target}")
    st.progress(completed_count / review_target, text=f"{completed_count}/{review_target} tersimpan")
    if completed_count:
        try:
            export_df = session_responses_dataframe(RESPONSE_DB_PATH, session_id)
            st.download_button(
                "Download respons sementara",
                data=export_df.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"evispan_responses_{st.session_state.respondent_code}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        except Exception as exc:
            st.caption(f"Export sementara belum tersedia: {exc}")

if st.session_state.get("study_phase") == "complete":
    render_workflow("complete", review_target, review_target)
    st.markdown(
        '<div class="completion-box"><strong>Sesi evaluasi EviSpan-PR telah selesai.</strong><br>'
        "Respons komentar dan tugas identifikasi pola telah tersimpan. Tahap terakhir adalah "
        "mengisi kuesioner Task–Technology Fit (TTF).</div>",
        unsafe_allow_html=True,
    )
    st.write(f"**Kode penyelesaian:** `{session_id[:12].upper()}`")

    if valid_questionnaire_url(TTF_QUESTIONNAIRE_URL):
        st.link_button(
            "Buka kuesioner TTF",
            TTF_QUESTIONNAIRE_URL,
            type="primary",
            use_container_width=True,
        )
        st.caption(
            "Gunakan kode responden dan kode penyelesaian di atas apabila diminta pada kuesioner."
        )
    else:
        st.warning(
            "Link kuesioner TTF belum dikonfigurasi. Tetapkan environment variable "
            "`EVISPAN_TTF_URL` dengan URL Google Forms atau platform survei yang digunakan."
        )

    try:
        export_df = session_responses_dataframe(RESPONSE_DB_PATH, session_id)
        st.download_button(
            "Download respons per komentar",
            data=export_df.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"evispan_comment_responses_{st.session_state.respondent_code}.csv",
            mime="text/csv",
        )
        final_df = session_final_dataframe(RESPONSE_DB_PATH, session_id)
        st.download_button(
            "Download hasil interpretasi dan pola",
            data=final_df.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"evispan_final_response_{st.session_state.respondent_code}.csv",
            mime="text/csv",
        )
    except Exception as exc:
        st.error(f"Salinan respons tidak dapat dibuat: {exc}")
    st.stop()

if st.session_state.get("study_phase") == "final":
    render_workflow("final", completed_count, review_target)
    st.subheader("Tugas interpretasi dan identifikasi pola")
    st.markdown(
        "Setelah menelaah seluruh komentar, rangkum pola yang Anda temukan pada prediksi label, "
        "probabilitas, dan evidence span EviSpan-PR."
    )

    with st.form("final_interpretation_form"):
        easiest_label = st.selectbox(
            "Label yang paling mudah diinterpretasikan *",
            options=LABELS,
        )
        hardest_label = st.selectbox(
            "Label yang paling sulit atau ambigu *",
            options=LABELS,
            index=LABELS.index("Neutral") if "Neutral" in LABELS else 0,
        )
        recurring_patterns = st.text_area(
            "Pola umum apa yang Anda identifikasi dari komentar dan hasil model? *",
            placeholder=(
                "Contoh: Appreciation cenderung memiliki ungkapan positif yang eksplisit, "
                "sedangkan Suggestion sering muncul setelah pernyataan Problem."
            ),
            height=130,
        )
        recurring_mismatches = st.text_area(
            "Ketidaksesuaian atau pola kesalahan apa yang berulang?",
            placeholder="Jelaskan label, probabilitas, atau batas evidence span yang sering kurang tepat.",
            height=110,
        )
        usefulness_rating = st.slider(
            "Seberapa membantu label, probabilitas, dan evidence span untuk menelaah peer feedback? *",
            min_value=1,
            max_value=5,
            value=4,
            help="1 = sangat tidak membantu; 5 = sangat membantu.",
        )
        intended_use = st.text_area(
            "Bagaimana EviSpan-PR dapat digunakan dalam kegiatan pembelajaran atau monitoring dosen?",
            height=100,
        )
        overall_notes = st.text_area("Catatan tambahan", height=90)
        finish = st.form_submit_button("Simpan dan selesaikan sesi", type="primary")

    if finish:
        if not recurring_patterns.strip():
            st.error("Ringkasan pola umum wajib diisi.")
        else:
            final_payload = {
                "session_id": session_id,
                "easiest_label": easiest_label,
                "hardest_label": hardest_label,
                "recurring_patterns": recurring_patterns.strip(),
                "recurring_mismatches": recurring_mismatches.strip(),
                "usefulness_rating": int(usefulness_rating),
                "intended_use": intended_use.strip(),
                "overall_notes": overall_notes.strip(),
                "submitted_at": utc_now_iso(),
            }
            try:
                save_final_response(RESPONSE_DB_PATH, final_payload)
            except Exception as exc:
                st.error(f"Tugas akhir tidak dapat disimpan: {exc}")
                st.stop()
            st.session_state.final_response = final_payload
            st.session_state.study_phase = "complete"
            st.rerun()
    st.stop()

# Review phase: run inference and collect a response for each selected comment.
st.session_state.study_phase = "review"
if "review_position" not in st.session_state:
    st.session_state.review_position = 0
if "review_page_jump" not in st.session_state:
    st.session_state.review_page_jump = 1
_set_review_position(int(st.session_state.review_position), review_target)

st.subheader("Telaah komentar peer feedback")
st.markdown(
    "Periksa teks, label prediksi, probabilitas, dan evidence span. Setelah itu, simpan "
    "interpretasi Anda untuk komentar yang sedang ditampilkan."
)
render_review_navigation(review_target, "top")
st.divider()

review_position = int(st.session_state.review_position)
source_row_index = int(review_indices[review_position])
row = working_df.iloc[source_row_index]
text = str(row[text_column]).strip()
row_id = source_row_index + 1 if id_column_selection == "(gunakan nomor baris)" else row[id_column_selection]

cache_key = hashlib.sha256(
    f"{study_signature}:{source_row_index}:{text}".encode("utf-8")
).hexdigest()

if cache_key not in st.session_state.prediction_cache:
    try:
        model, tokenizer, thresholds, config, device = get_live_model(str(artifact_dir))
        with st.spinner(f"Memprediksi komentar {review_position + 1}..."):
            st.session_state.prediction_cache[cache_key] = predict_text(
                text=text,
                model=model,
                tokenizer=tokenizer,
                thresholds=thresholds,
                config=config,
                device=device,
            )
    except Exception as exc:
        st.error(f"Prediksi gagal dijalankan: {exc}")
        st.stop()

result = st.session_state.prediction_cache[cache_key]
predicted_labels = list(result.get("predicted_labels", []))
predicted_spans = list(result.get("predicted_spans", []))

meta_1, meta_2, meta_3, meta_4 = st.columns(4)
meta_1.metric("Urutan telaah", f"{review_position + 1}/{review_target}")
meta_2.metric("ID data", str(row_id))
meta_3.metric("Jumlah label", len(predicted_labels))
meta_4.metric("Evidence span", len(predicted_spans))

st.markdown('<div class="section-label">Teks peer feedback</div>', unsafe_allow_html=True)
st.markdown(render_plain_text(text), unsafe_allow_html=True)

label_col, prob_col = st.columns([1, 1])
with label_col:
    st.markdown('<div class="section-label" style="margin-top:1.2rem">Prediksi label</div>', unsafe_allow_html=True)
    st.markdown(label_badges(predicted_labels), unsafe_allow_html=True)
with prob_col:
    st.markdown('<div class="section-label" style="margin-top:1.2rem">Probabilitas label</div>', unsafe_allow_html=True)
    st.dataframe(
        probabilities_dataframe(result),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Probability": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.3f"),
            "Predicted": st.column_config.CheckboxColumn(),
        },
    )

st.markdown('<div class="section-label" style="margin-top:1.2rem">Evidence span hasil prediksi</div>', unsafe_allow_html=True)
st.markdown(render_highlighted_text(text, predicted_spans), unsafe_allow_html=True)
if predicted_spans:
    st.dataframe(
        spans_dataframe(predicted_spans),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Evidence": st.column_config.TextColumn(width="large"),
            "Start": st.column_config.NumberColumn(format="%d"),
            "End": st.column_config.NumberColumn(format="%d"),
        },
    )
else:
    st.warning("Model tidak menghasilkan evidence span untuk komentar ini.")

if result.get("truncated"):
    st.info("Teks melebihi panjang maksimum model dan dipotong saat inferensi.")

response_key = str(source_row_index)
existing = responses.get(response_key, {})
label_assessment_options = [
    "Sesuai seluruhnya",
    "Sebagian sesuai",
    "Tidak sesuai",
    "Tidak dapat menilai",
]
evidence_support_options = [
    "Seluruh span mendukung label",
    "Sebagian span mendukung label",
    "Span tidak mendukung label",
    "Tidak ada evidence span",
    "Tidak dapat menilai",
]
evidence_boundary_options = [
    "Batas span tepat",
    "Batas span sebagian tepat",
    "Batas span tidak tepat",
    "Tidak ada evidence span",
    "Tidak dapat menilai",
]
pattern_options = [
    "Pujian atau evaluasi positif eksplisit",
    "Identifikasi kekurangan atau masalah",
    "Rekomendasi tindakan yang konkret",
    "Informasi atau deskripsi netral",
    "Gabungan beberapa fungsi feedback",
    "Label atau evidence span ambigu",
]

def option_index(options: List[str], value: Any, default: int = 0) -> int:
    return options.index(value) if value in options else default

st.divider()
st.subheader("Respons dosen untuk komentar ini")
with st.form(f"comment_response_form_{source_row_index}"):
    label_assessment = st.radio(
        "Apakah label prediksi sesuai dengan isi komentar? *",
        options=label_assessment_options,
        index=option_index(label_assessment_options, existing.get("label_assessment"), 0),
        horizontal=True,
    )
    expected_labels = st.multiselect(
        "Menurut Anda, label apa yang seharusnya diberikan? *",
        options=LABELS,
        default=existing.get("expected_labels", predicted_labels),
        help="Boleh memilih lebih dari satu label. Kosongkan hanya bila komentar tidak sesuai dengan semua kategori.",
    )
    evidence_support = st.radio(
        "Apakah evidence span benar-benar mendukung label prediksi? *",
        options=evidence_support_options,
        index=option_index(
            evidence_support_options,
            existing.get("evidence_support"),
            3 if not predicted_spans else 0,
        ),
    )
    evidence_boundary = st.radio(
        "Bagaimana ketepatan batas evidence span? *",
        options=evidence_boundary_options,
        index=option_index(
            evidence_boundary_options,
            existing.get("evidence_boundary"),
            3 if not predicted_spans else 0,
        ),
    )
    identified_patterns = st.multiselect(
        "Pola apa yang tampak pada komentar ini?",
        options=pattern_options,
        default=existing.get("identified_patterns", []),
    )
    interpretation = st.text_area(
        "Jelaskan interpretasi Anda secara singkat *",
        value=existing.get("interpretation", ""),
        placeholder="Jelaskan alasan kesesuaian atau ketidaksesuaian label dan evidence span.",
        height=105,
    )
    confidence = st.slider(
        "Tingkat keyakinan terhadap penilaian Anda *",
        min_value=1,
        max_value=5,
        value=int(existing.get("confidence", 4)),
        help="1 = sangat tidak yakin; 5 = sangat yakin.",
    )
    notes = st.text_area(
        "Catatan tambahan",
        value=existing.get("notes", ""),
        height=75,
    )
    save_only, save_next = st.columns(2)
    save_response = save_only.form_submit_button("Simpan respons", use_container_width=True)
    save_and_next = save_next.form_submit_button("Simpan & lanjut", type="primary", use_container_width=True)

if save_response or save_and_next:
    if not interpretation.strip():
        st.error("Interpretasi singkat wajib diisi sebelum respons disimpan.")
    else:
        response_payload = {
            "session_id": session_id,
            "source_row_index": source_row_index,
            "review_order": review_position + 1,
            "row_id": str(row_id),
            "comment_text": text,
            "predicted_labels": predicted_labels,
            "label_probabilities": result.get("label_probabilities", {}),
            "predicted_spans": predicted_spans,
            "label_assessment": label_assessment,
            "expected_labels": expected_labels,
            "evidence_support": evidence_support,
            "evidence_boundary": evidence_boundary,
            "identified_patterns": identified_patterns,
            "interpretation": interpretation.strip(),
            "confidence": int(confidence),
            "notes": notes.strip(),
            "reviewed_at": utc_now_iso(),
        }
        try:
            save_comment_response(RESPONSE_DB_PATH, response_payload)
        except Exception as exc:
            st.error(f"Respons tidak dapat disimpan: {exc}")
            st.stop()

        st.session_state.study_responses[response_key] = response_payload
        st.success("Respons dosen telah disimpan.")
        if save_and_next and review_position < review_target - 1:
            _set_review_position(review_position + 1, review_target)
        st.rerun()

if response_key in responses:
    st.success("Respons untuk komentar ini sudah tersimpan dan dapat diperbarui.")

render_review_navigation(review_target, "bottom")
completed_count = len(st.session_state.study_responses)
if completed_count == review_target:
    st.success("Seluruh respons komentar telah tersimpan.")
    if st.button("Lanjut ke tugas interpretasi dan identifikasi pola", type="primary", use_container_width=True):
        st.session_state.study_phase = "final"
        st.rerun()
else:
    remaining = review_target - completed_count
    st.info(f"Masih ada {remaining} komentar yang belum memiliki respons tersimpan.")
