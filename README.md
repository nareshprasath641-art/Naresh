# Student Learning Behaviour Clustering using K-Means

## Introduction and problem statement
Students show varied study, attendance, participation, and preparation habits. This college project demonstrates how unsupervised learning can describe similar patterns without using predefined class labels. It is a complete local Python and Streamlit project with generated data, persisted preprocessing and models, report charts, interactive analysis, and CSV export.

**All 2,400 records are synthetic.** Clusters describe this generated population. They do not establish real student types, academic ability, educational effectiveness, or causation. Do not use these labels to grade students or make consequential decisions.

## Objectives and features
- Generate reproducible student data with missing measurements.
- Inspect distributions, relationships, duplicates, and outliers.
- Compare K=2 through 10 and train reproducible K-Means.
- Interpret learned profiles after training rather than preset cluster names.
- Explore eight dashboard pages, including the seven requested analysis pages and CSV upload.
- Assign individual students and uploaded CSV rows using the saved model.
- Export student groups and profiles; regenerate missing models automatically.

## Quick start on Windows / VS Code
Clone, set up, and start everything with this single line in **Command Prompt**, from the folder where you want to download the project:

```bat
git clone https://github.com/nareshprasath641-art/Naresh.git && cd Naresh && python -m venv .venv && .venv\Scripts\python.exe -m pip install -r requirements.txt && .venv\Scripts\python.exe src\data_generator.py && .venv\Scripts\python.exe -m streamlit run app.py
```

For later launches, open Command Prompt in the downloaded `Naresh` folder and run `.venv\Scripts\python.exe -m streamlit run app.py`.

Open this project folder in VS Code. Python 3.11 or newer is recommended. In its terminal:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src/data_generator.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Once your environment is activated, the requested commands also work:

```bash
pip install -r requirements.txt
python src/data_generator.py
streamlit run app.py
```

Open the local URL printed by Streamlit. First run prepares any missing models and outputs. No login or external account is required. Regeneration uses fixed seed 42. To rebuild explicitly, print metrics, or verify:

```bash
python src/clustering.py
python verify_project.py
```

Paths are resolved from source-file locations rather than the terminal working directory. The root CSV is authoritative; `data/` holds a synchronized copy after generation/training. To train on your own data, replace the root CSV and run `python src/clustering.py`. Uploads assign existing groups and do not retrain.

## Dataset description
`student_learning_behavior.csv` contains 2,400 rows and 12 columns. The generator uses a continuous latent engagement variable with independent noise, not preassigned student categories. Approximately 1.5% of behavioural cells are missing. There are no real identities.

| Column | Meaning / supported input range |
|---|---|
| Student_ID | Nonempty row identifier |
| Age | Whole years, 16–35; descriptive only |
| Study_Hours_Per_Day | Hours/day, 0–10 |
| Attendance_Percentage | 0–100% |
| Assignment_Completion_Percentage | 0–100% |
| Online_Learning_Hours | Hours/day, 0–8 |
| Class_Interactions | Whole interactions/week, 0–50 |
| Previous_Semester_Marks | Marks, 0–100 |
| Sleep_Hours | Hours/day, 3–12 |
| Extracurricular_Hours | Hours/day, 0–6 |
| Test_Preparation_Hours | Hours/day, 0–8 |
| Average_Test_Score | Marks, 0–100 |

Online learning and test preparation can overlap with total study time; they are not additive time-budget components. All learning hours are typical daily averages. The configured ranges are demonstration input constraints, not medical or educational standards.

## Technologies
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Plotly, Streamlit, Joblib, and threadpoolctl. No database or complex framework is needed. `requirements.txt` lists compatible version ranges. Artifacts record the training library versions; the app retrains if the relevant installed versions change. Load only the locally generated Joblib artifacts, because pickle-based files are executable data.

## Preprocessing
1. Pandas loads the data. Dataset Analysis shows shape, first five rows, types, missing counts, and descriptive statistics.
2. Required columns, numeric types, finite values, and sensible input bounds are validated. Blank numerical cells are accepted.
3. Exact duplicate training rows are removed. Conflicting duplicate IDs are rejected during training. Uploaded duplicates remain visible and are reported.
4. Student_ID and Age are excluded from the distance calculation. The model uses **10 learning-behaviour features**; age is not itself a habit.
5. A training-fitted `SimpleImputer(strategy='median')` fills behavioural missing values. The original empty cells remain in exported student records for transparency. Cluster averages use the imputed measurements.
6. The 1.5 × IQR rule reports possible outliers. Valid extreme habits are retained.
7. `StandardScaler` gives each feature comparable units. Otherwise percentages could dominate hours in Euclidean distance.

The same saved imputer and scaler transform each new student and CSV upload. They are never refitted on prediction inputs. Age can remain missing because it is excluded from modelling.

## Algorithm and selection of K
K-Means iteratively assigns points to their nearest centroid and updates each centroid to the mean of its assigned points. The objective is to minimize within-cluster squared Euclidean distances (inertia). We use `random_state=42` and `n_init=20`.

**Elbow method:** inspect the bend in inertia as K increases, where added clusters give smaller improvements. This diagnostic is not always decisive.

**Silhouette:** compares within-cluster closeness with distance to the nearest other cluster. Values near 1 indicate stronger separation; values near 0 indicate overlap; negative values can indicate poor assignments. The explicit selection rule takes the highest full-dataset silhouette among K=2…10; a tie uses the smaller K. Four is not forced.

## Evaluation and expected output
The bundled reproducible run selects **K=2**:

| Metric | Value | Interpretation |
|---|---:|---|
| Silhouette | 0.301694 | Higher generally means better separation |
| Davies–Bouldin | 1.233422 | Lower generally means less overlap |
| Calinski–Harabasz | 1484.164092 | Higher generally means stronger between-group dispersion |
| Inertia | 14824.770435 | Within-cluster squared distances; falls as K increases |

Exact values may vary slightly across numerical library versions. Internal metrics assess geometry, not academic success. A train/test accuracy or confusion matrix would not be appropriate here because there is no supervised target or ground-truth clustering.

The elbow curve is available alongside silhouette. This generator produces continuous, overlapping habits, so broad separation wins over dividing students into four categories. PCA is fitted only for visualization; the actual clustering uses all ten standardized features. The two-dimensional view retains about 65.5% of variance.

## Cluster interpretation
After clustering, the program ranks each learned profile by the mean of study hours, attendance, assignment completion, interactions, and preparation after dividing by configured upper bounds. This is an explicit descriptive heuristic, not a validated engagement scale. Neither marks nor sleep nor extracurricular time determines the name.

The current profiles are **Lower Engagement Learners** and **Highly Engaged Learners**. “Highly” is relative to this synthetic population. If four groups were selected, intermediate ranks would receive Moderate Learners and Consistent Learners; other K values get numbered intermediate labels. Cluster IDs are arbitrary and have no inherent ranking.

## Dashboard pages
1. Home: objective, sizes, algorithm, and caveat.
2. Dataset Analysis: preview, missingness, types, statistics, outlier counts, correlation.
3. Exploratory Data Analysis: interactive histograms, box plots, and test-score relationships.
4. K-Means Analysis: elbow, silhouette, selected K, four metrics, distribution.
5. Student Clusters: profiles, explanations, selectable students, CSV downloads.
6. Cluster Visualization: 2D/3D PCA, projected centers, feature comparisons, radar.
7. Student Behaviour Prediction: eleven input measurements and nearest-cluster explanation.
8. CSV Upload: example template, validation, existing-model assignment, distribution, download.

CSV files must contain all 12 dataset columns, with numeric measurements or empty cells. Extra columns are preserved. Uploads are limited to 20 MB and 100,000 rows. Text and infinite/out-of-range numbers are rejected with a message. Export uses UTF-8 with BOM for Excel compatibility; potentially executable spreadsheet text is escaped.

## Folder structure
```text
Student_Learning_Behaviour_Clustering/
├── app.py
├── app_pages/                 # Eight functional pages
├── student_learning_behavior.csv
├── requirements.txt
├── README.md
├── VIVA_GUIDE.md
├── verify_project.py
├── .streamlit/config.toml
├── data/student_learning_behavior.csv
├── models/
│   ├── kmeans_model.pkl
│   ├── scaler.pkl
│   ├── imputer.pkl
│   └── pca.pkl
├── outputs/
│   ├── clustered_students.csv
│   ├── cluster_summary.csv
│   ├── k_evaluation.csv
│   ├── metrics.json
│   ├── diagnostics.json
│   ├── descriptive_statistics.csv
│   ├── pca_students.csv
│   └── charts/               # 17 PNGs plus standalone radar/3D HTML
└── src/
    ├── data_generator.py
    ├── preprocessing.py
    ├── clustering.py
    ├── visualization.py
    └── dashboard.py
```

## Verification
`verify_project.py` checks consistency between saved labels and model reassignment, maximum-silhouette selection, CSV round trips, invalid data, missing-value handling, all eight Streamlit pages, filtering, individual assignment, 3D visualization, upload results and errors, download element availability, and automatic generation in a fresh temporary folder followed by recovery of a missing scaler.

Streamlit AppTest exercises the upload-processing interface with in-memory CSV bytes because file selection is not directly exposed in its widget API. It verifies rendered tables and download controls; operating-system file-picker interaction is outside these automated checks.

## Limitations and future enhancements
Use consented, representative real data before drawing educational conclusions. Study stability across seeds and cohorts, assess alternative algorithms, investigate feature redundancy (several engagement features correlate), and involve educators and students in interpretation. K-Means favours roughly spherical groups, is sensitive to scaling and extreme values, and assigns every input to a cluster even when it differs substantially from training data. Future versions could flag distant observations, compare feature sets, add longitudinal patterns, and evaluate fairness and privacy before real deployment.
