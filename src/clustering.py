"""Train, persist, and reuse K-Means and its fitted preprocessing."""
import hashlib
import json
import sys
from importlib.metadata import version
from pathlib import Path
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from threadpoolctl import threadpool_limits
from src.preprocessing import ROOT, FEATURES, diagnostics, fit_preprocessing, validate_frame
from src.data_generator import generate_dataset

ARTIFACTS = ['models/kmeans_model.pkl', 'models/scaler.pkl', 'models/imputer.pkl',
             'models/pca.pkl', 'outputs/clustered_students.csv', 'outputs/cluster_summary.csv',
             'outputs/k_evaluation.csv', 'outputs/metrics.json', 'outputs/diagnostics.json',
             'outputs/pca_students.csv', 'outputs/charts/pca_2d.png']

def runtime_versions():
    return {name: version(name) for name in ['scikit-learn', 'numpy', 'pandas', 'joblib']}

def fingerprint(root):
    return hashlib.sha256((root / 'student_learning_behavior.csv').read_bytes()).hexdigest()

def interpret_profiles(summary):
    # Labels are assigned AFTER training using measured engagement averages.
    # Sleep and extracurricular time have no assumed good/bad direction.
    columns = ['Study_Hours_Per_Day', 'Attendance_Percentage',
               'Assignment_Completion_Percentage', 'Class_Interactions', 'Test_Preparation_Hours']
    normalized = summary[columns] / pd.Series([10, 100, 100, 50, 8], index=columns)
    order = normalized.mean(axis=1).sort_values().index.tolist()
    labels = {}
    for rank, cluster in enumerate(order):
        if rank == 0:
            label = 'Lower Engagement Learners'
        elif rank == len(order) - 1:
            label = 'Highly Engaged Learners'
        elif len(order) == 4:
            label = ['Moderate Learners', 'Consistent Learners'][rank - 1]
        else:
            label = f'Intermediate Engagement Learners {rank}'
        labels[int(cluster)] = label
    return labels

def train_project(root=ROOT):
    root = Path(root)
    for folder in ['models', 'outputs/charts', 'data']:
        (root / folder).mkdir(parents=True, exist_ok=True)
    if not (root / 'student_learning_behavior.csv').exists():
        generate_dataset(root=root)
    raw = pd.read_csv(root / 'student_learning_behavior.csv')
    report = diagnostics(validate_frame(raw))
    clean, filled, scaled, imputer, scaler = fit_preprocessing(raw)
    if len(clean) < 20:
        raise ValueError('Training needs at least 20 distinct records.')
    candidates, fitted = [], {}
    with threadpool_limits(limits=1):
        for k in range(2, 11):
            model = KMeans(n_clusters=k, random_state=42, n_init=20)
            labels = model.fit_predict(scaled)
            score = silhouette_score(scaled, labels)
            candidates.append({'K': k, 'Inertia': model.inertia_, 'Silhouette': score})
            fitted[k] = model
    evaluation = pd.DataFrame(candidates)
    # The elbow is a visual diagnostic; maximum silhouette is the explicit,
    # reproducible selection rule. Four is never imposed on the observations.
    best_k = int(evaluation.loc[evaluation.Silhouette.idxmax(), 'K'])
    model = fitted[best_k]
    clean['Cluster'] = model.labels_
    imputed = pd.DataFrame(filled, columns=FEATURES)
    imputed['Cluster'] = model.labels_
    summary = imputed.groupby('Cluster')[FEATURES].mean()
    names = interpret_profiles(summary)
    summary['Student_Count'] = imputed.groupby('Cluster').size()
    summary['Learning_Behaviour'] = summary.index.map(names)
    clean['Learning_Behaviour'] = clean.Cluster.map(names)
    pca = PCA(n_components=3, random_state=42)
    coordinates = pd.DataFrame(pca.fit_transform(scaled), columns=['PC1', 'PC2', 'PC3'])
    coordinates['Cluster'] = clean.Cluster
    coordinates['Learning_Behaviour'] = clean.Learning_Behaviour
    metrics = {
        'records': len(clean), 'features': FEATURES, 'selected_k': best_k,
        'silhouette': float(evaluation.Silhouette.max()),
        'davies_bouldin': float(davies_bouldin_score(scaled, model.labels_)),
        'calinski_harabasz': float(calinski_harabasz_score(scaled, model.labels_)),
        'inertia': float(model.inertia_), 'labels': names,
        'pca_explained_variance': pca.explained_variance_ratio_.tolist(),
        'dataset_sha256': fingerprint(root),
        'runtime_versions': runtime_versions(),
        'selection_rule': 'Maximum full-dataset silhouette over K=2..10; elbow reviewed visually.',
    }
    for name, obj in [('kmeans_model', model), ('scaler', scaler), ('imputer', imputer), ('pca', pca)]:
        joblib.dump(obj, root / f'models/{name}.pkl')
    clean.to_csv(root / 'outputs/clustered_students.csv', index=False)
    summary.to_csv(root / 'outputs/cluster_summary.csv')
    evaluation.to_csv(root / 'outputs/k_evaluation.csv', index=False)
    coordinates.to_csv(root / 'outputs/pca_students.csv', index=False)
    raw.to_csv(root / 'data/student_learning_behavior.csv', index=False)
    raw.describe().to_csv(root / 'outputs/descriptive_statistics.csv')
    for name, obj in [('metrics', metrics), ('diagnostics', report)]:
        (root / f'outputs/{name}.json').write_text(json.dumps(obj, indent=2), encoding='utf-8')
    from src.visualization import save_charts
    save_charts(raw, summary, evaluation, coordinates, pca.transform(model.cluster_centers_), root)
    print('K-Means Clustering Results\n' + json.dumps(metrics, indent=2))
    return metrics

def ensure_artifacts(root=ROOT):
    root = Path(root)
    if not (root / 'student_learning_behavior.csv').exists() or any(not (root / name).exists() for name in ARTIFACTS):
        return train_project(root)
    metrics = json.loads((root / 'outputs/metrics.json').read_text())
    if metrics['dataset_sha256'] != fingerprint(root) or metrics.get('runtime_versions') != runtime_versions():
        return train_project(root)
    return metrics

def load_bundle(root=ROOT):
    root = Path(root)
    metrics = ensure_artifacts(root)
    return {**{name: joblib.load(root / f'models/{name}.pkl') for name in ['kmeans_model', 'scaler', 'imputer', 'pca']}, 'metrics': metrics}

def assign_students(frame, bundle):
    result = validate_frame(frame)
    # Reuse training medians and scaling: never refit on uploads or one student.
    values = bundle['imputer'].transform(result[FEATURES])
    scaled = bundle['scaler'].transform(values)
    result['Cluster'] = bundle['kmeans_model'].predict(scaled)
    names = {int(k): v for k, v in bundle['metrics']['labels'].items()}
    result['Learning_Behaviour'] = result.Cluster.map(names)
    return result

def csv_bytes(frame):
    # Escape spreadsheet formula prefixes in user-supplied text on export.
    exported = frame.copy()
    for column in exported.select_dtypes(include=['object', 'string']):
        exported[column] = exported[column].map(lambda x: "'" + x if isinstance(x, str) and x.lstrip().startswith(('=', '+', '-', '@')) else x)
    return exported.to_csv(index=False).encode('utf-8-sig')

if __name__ == '__main__':
    train_project()
