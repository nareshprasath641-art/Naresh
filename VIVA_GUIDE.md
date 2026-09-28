# Viva and demonstration guide

## Two-minute explanation
This project groups similar student learning habits using unsupervised K-Means. I generated 2,400 synthetic records with realistic ranges and missing values. I removed identifiers and age from clustering, filled missing measurements with training medians, and standardized ten behavioural features. I compared K=2 through 10 using the elbow curve and silhouette, selecting two because it achieved the highest silhouette. I then named groups from their measured engagement profiles. The Streamlit dashboard explains the data, model, visualizations, and new-student assignment.

## Questions and answers
**Why unsupervised learning?** There are no known category targets; the model discovers geometric similarity.

**Why standardize?** Percentages and hours have different scales. Scaling prevents large numeric ranges from dominating distances.

**Why median imputation?** It is simple and less affected by extreme values than the mean. Training medians must be reused on new data.

**Why exclude ID and age?** ID has no behavioural meaning. Age describes the student, not a study habit, and can otherwise drive demographic grouping.

**Why not force four clusters?** Four was a suggestion, while measured silhouette favoured two. The generated habits vary continuously, so there need not be four distinct types.

**Are the labels predictions of success?** No. They describe similarity to learned groups. There is no supervised academic-success target.

**Why PCA?** It projects ten-dimensional measurements into two or three dimensions for viewing. It is not used to train K-Means here.

**What does a silhouette of 0.302 mean?** There is some separation and considerable overlap; it is not 30.2% classification accuracy.

**Why no confusion matrix?** There are no ground-truth classes. Internal clustering scores are suitable for this demonstration.

**How are names assigned?** After training, a documented heuristic ranks measured study, attendance, assignment, interaction, and preparation averages. Names remain relative and unvalidated.

**What happens to a CSV with missing cells?** Saved training medians fill model inputs. Original missing cells remain visible in the exported records.

**Main limitations?** Synthetic data, correlated features, assumed distance geometry, relative labels, and no validation of real educational usefulness.

## Five-minute demo
1. Home: show sample count, feature count, K, and synthetic-data limitation.
2. Dataset: show missing cells, statistics, and why preprocessing is needed.
3. EDA: choose attendance and discuss the generated relationship with marks.
4. K-Means: compare the elbow and silhouette curves and explain K=2.
5. Clusters: select each group and inspect its actual averages.
6. Visualization: show PCA and explain lost variance.
7. Prediction: enter a student and explain nearest-center assignment.
8. Upload: use the downloadable template, assign rows, and download results.

Do not claim these synthetic findings prove anything about actual students. Edit any academic cover-page identity details yourself; none are assumed by this project.
