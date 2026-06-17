# AutomatedRE Model Comparison Experiment

Repositori ini berisi replikasi dan perluasan eksperimen ekstraksi diagram kelas UML dari teks kebutuhan. Pipeline rujukan menggunakan tiga sub-model SVM; eksperimen ini mengganti classifier dengan:

- Logistic Regression
- Random Forest
- XGBoost

Semua model memakai preprocessing, fitur, target, dan protokol evaluasi yang sama agar perbandingan adil.

## Struktur Utama

```text
Dataset/
  Requirements.csv
  Requirements_ProblemAnalysis.csv
  r12_problems.csv.csv
  real_world_studies.csv
  real_world_studies_template.csv

01_experiment_logistic_regression.ipynb
02_experiment_random_forest.ipynb
03_experiment_xgboost.ipynb
04_experiment_real_world_studies.ipynb

create_experiment_notebooks.py
create_real_world_notebook.py
```

Folder `experiment_outputs/`, `paper/`, `model/`, dan file model `.joblib` tidak dimasukkan ke GitHub karena berisi artefak hasil eksperimen, draft paper, atau file berukuran besar.

## Dependensi

Gunakan Python 3.10+ atau 3.11. Install dependensi berikut:

```bash
pip install pandas numpy scikit-learn nltk joblib matplotlib xgboost python-docx
```

Jika memakai Jupyter/VSCode Notebook:

```bash
pip install notebook ipykernel
```

## Alur Eksperimen

Jalankan notebook secara berurutan. Disarankan menjalankan satu notebook sampai selesai, lalu shutdown kernel sebelum lanjut ke notebook berikutnya.

### 1. Logistic Regression

Jalankan:

```text
01_experiment_logistic_regression.ipynb
```

Output utama:

```text
experiment_outputs/logistic_regression/
  logistic_regression_class_attribute.joblib
  logistic_regression_class_attribute_relationship.joblib
  logistic_regression_class_class_relationship.joblib
  table3_metrics.csv
  table3_vs_svm_paper.csv
  table4_r12_model.csv
  table4_r12_vs_svm_domobot.csv
  r12_predictions.csv
  plantuml/
```

### 2. Random Forest

Shutdown kernel notebook pertama, lalu jalankan:

```text
02_experiment_random_forest.ipynb
```

Output utama:

```text
experiment_outputs/random_forest/
  random_forest_class_attribute.joblib
  random_forest_class_attribute_relationship.joblib
  random_forest_class_class_relationship.joblib
  table3_metrics.csv
  table4_r12_model.csv
  table4_r12_vs_svm_domobot.csv
```

Catatan: file model Random Forest bisa sangat besar, terutama sub-model `class_attribute_relationship`.

### 3. XGBoost

Shutdown kernel notebook kedua, lalu jalankan:

```text
03_experiment_xgboost.ipynb
```

Notebook ini memakai wrapper `XGBLabelEncodedClassifier` agar label target dikodekan ulang di setiap fold cross-validation. Ini diperlukan karena XGBoost membutuhkan indeks kelas numerik yang berurutan pada setiap proses `fit`.

Output utama:

```text
experiment_outputs/xgboost/
  xgboost_class_attribute.joblib
  xgboost_class_attribute_relationship.joblib
  xgboost_class_class_relationship.joblib
  table3_metrics.csv
  table4_r12_model.csv
  table4_r12_vs_svm_domobot.csv
```

### 4. Real-World Studies

Setelah tiga model utama selesai, jalankan:

```text
04_experiment_real_world_studies.ipynb
```

Notebook ini membaca:

```text
Dataset/real_world_studies.csv
```

File tersebut berisi dua reconstructed real-world case studies:

- System 1: Stroke recovery assistant
- System 2: Archive space project

Catatan penting: teks asli dua studi kasus dari paper rujukan tidak tersedia eksplisit di dataset publik. Karena itu file ini adalah rekonstruksi berbasis karakteristik yang dilaporkan paper rujukan, bukan salinan input asli.

Output utama:

```text
experiment_outputs/real_world_studies/
  table5_real_world_statistics.csv
  table6_real_world_results.csv
  table6_real_world_results_by_target.csv
  figure5_expert_vs_model_counts.csv
  figure5a_lr_expert_vs_model.png
  figure5b_rf_expert_vs_model.png
  figure5c_xgb_expert_vs_model.png
```

Pada Figure 5, label `Expert` berarti manual reference annotation/proxy expert dari `real_world_studies.csv`, bukan expert asli dari paper rujukan.

## Generate Ulang Notebook

Jika notebook perlu dibuat ulang dari template:

```bash
python create_experiment_notebooks.py
python create_real_world_notebook.py
```

Perintah pertama membuat ulang notebook 01-03. Perintah kedua membuat ulang notebook 04.

## Ringkasan Hasil Final

### Table 3: Dataset Utama 80:20

Weighted F1:

| Model | Weighted F1 |
|---|---:|
| SVM paper | 0.9200 |
| Logistic Regression | 0.9469 |
| Random Forest | 0.8905 |
| XGBoost | 0.9197 |

### Table 4: 12 Requirement Problems

Average F1:

| Model | Average F1 |
|---|---:|
| SVM paper | 85.00% |
| DoMoBOT | 84.33% |
| Logistic Regression | 94.92% |
| Random Forest | 96.19% |
| XGBoost | 90.04% |

### Real-World Reconstructed Studies

| Model | System 1 F1 | System 2 F1 |
|---|---:|---:|
| Logistic Regression | 81.25 | 66.90 |
| Random Forest | 75.44 | 63.75 |
| XGBoost | 80.96 | 64.36 |

## Catatan Validitas

- Eksperimen 01-03 menggunakan dataset publik dari repo rujukan.
- Evaluasi 12 problem memakai dataset `r12_problems.csv.csv`.
- Evaluasi real-world memakai reconstructed case studies karena teks asli System 1 dan System 2 tidak tersedia eksplisit.
- Figure 5 memakai manual reference annotation/proxy expert, bukan expert asli paper rujukan.
- Jangan klaim real-world study sebagai replikasi persis paper rujukan; klaim yang tepat adalah evaluasi tambahan pada studi kasus rekonstruksi.

