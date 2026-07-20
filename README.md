# EviSpan-PR Streamlit

Aplikasi ini menampilkan:

- isi data uji;
- label aktual dan hasil prediksi multi-label;
- probabilitas untuk `Problem`, `Suggestion`, `Neutral`, dan `Appreciation`;
- evidence span hasil CRF-BILUO yang disorot pada teks;
- perbandingan evidence span prediksi dengan anotasi aktual;
- ringkasan performa document-level pada seluruh data uji;
- inferensi komentar baru apabila checkpoint lengkap tersedia.

## Struktur proyek

```text
evispan_streamlit/
├── app.py
├── evispan_model.py
├── export_streamlit_artifacts.py
├── requirements.txt
├── run_app.bat
├── run_app.sh
└── artifacts/
    └── evispan_pr/
        ├── artifact_config.json
        ├── thresholds.json
        ├── model_state.pt
        ├── test_predictions.jsonl
        ├── test_data.jsonl
        ├── encoder_config/
        │   └── config.json
        └── tokenizer/
```

## 1. Menyiapkan model deployment dari notebook

Model pada notebook harus dijalankan dalam mode **single split** agar tersedia satu checkpoint final dan satu test set yang tetap. Sebelum menjalankan eksperimen, gunakan pengaturan berikut:

```python
CFG.run_single_split = True
CFG.run_multilabel_cv = False
CFG.run_ablation_cv = False
```

Jalankan notebook sampai bagian evaluasi test selesai sehingga objek berikut tersedia:

```text
model, tokenizer, best_thresholds, test_features, test_loader, test_pred
```

Notebook versi Streamlit yang disertakan sudah memperoleh satu sel ekspor tambahan di bagian akhir. Sel tersebut membuat folder:

```text
<CFG.output_dir>/streamlit_artifacts/evispan_pr/
```

Sebagai alternatif, jalankan file ekspor dari kernel notebook:

```python
exec(open("export_streamlit_artifacts.py", encoding="utf-8").read(), globals())
```

## 2. Menyalin artifact

Salin seluruh isi folder hasil ekspor ke:

```text
artifacts/evispan_pr/
```

Untuk hanya menampilkan data uji dan prediksi, aplikasi cukup membutuhkan:

```text
test_predictions.jsonl
```

Untuk menjalankan inferensi komentar baru, seluruh checkpoint, konfigurasi encoder, dan tokenizer harus tersedia.

## 3. Instalasi

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

Linux/macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## 4. Menjalankan aplikasi

```bash
streamlit run app.py
```

Atau gunakan `run_app.bat` pada Windows dan `run_app.sh` pada Linux/macOS.

## Format `test_predictions.jsonl`

Setiap baris adalah satu objek JSON:

```json
{
  "id": 1,
  "text": "Komentar peer review...",
  "true_labels": ["Problem", "Suggestion"],
  "predicted_labels": ["Problem", "Suggestion"],
  "label_probabilities": {
    "Problem": 0.91,
    "Suggestion": 0.83,
    "Neutral": 0.12,
    "Appreciation": 0.21
  },
  "true_spans": [
    {"start": 0, "end": 15, "label": "Problem", "text": "..."}
  ],
  "predicted_spans": [
    {"start": 0, "end": 15, "label": "Problem", "text": "..."}
  ]
}
```

## Catatan metodologis

Checkpoint untuk deployment sebaiknya berasal dari model yang dilatih ulang setelah konfigurasi terbaik dipilih melalui cross-validation. Test set yang ditampilkan dalam aplikasi harus tetap terpisah dari data training dan validation. Jangan memilih checkpoint berdasarkan performa test set.
