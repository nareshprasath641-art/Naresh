"""Validate inputs and fit preprocessing only on the training dataset."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
BOUNDS = {
    'Age': (16, 35), 'Study_Hours_Per_Day': (0, 10),
    'Attendance_Percentage': (0, 100), 'Assignment_Completion_Percentage': (0, 100),
    'Online_Learning_Hours': (0, 8), 'Class_Interactions': (0, 50),
    'Previous_Semester_Marks': (0, 100), 'Sleep_Hours': (3, 12),
    'Extracurricular_Hours': (0, 6), 'Test_Preparation_Hours': (0, 8),
    'Average_Test_Score': (0, 100),
}
# ID identifies a row; it must never influence distance. Age is descriptive,
# not a learning habit, so it is retained in outputs but excluded from training.
FEATURES = [name for name in BOUNDS if name != 'Age']
REQUIRED = ['Student_ID', *BOUNDS]

def validate_frame(frame):
    missing = sorted(set(REQUIRED) - set(frame.columns))
    if missing:
        raise ValueError('Missing required columns: ' + ', '.join(missing))
    if frame.empty:
        raise ValueError('The CSV has no student records.')
    if len(frame) > 100000:
        raise ValueError('Please upload at most 100,000 records.')
    result = frame.copy()
    if result.Student_ID.isna().any() or result.Student_ID.astype(str).str.strip().eq('').any():
        raise ValueError('Every student needs a non-empty Student_ID.')
    for name, (low, high) in BOUNDS.items():
        values = pd.to_numeric(result[name], errors='coerce')
        if (result[name].notna() & values.isna()).any():
            raise ValueError(f'{name} must contain numbers or empty cells.')
        if (values.notna() & (~np.isfinite(values) | ~values.between(low, high))).any():
            raise ValueError(f'{name} must be between {low} and {high}.')
        if name in ('Age', 'Class_Interactions') and ((values.dropna() % 1) != 0).any():
            raise ValueError(f'{name} must contain whole numbers.')
        result[name] = values
    return result

def diagnostics(frame):
    numeric = frame[list(BOUNDS)]
    q1, q3 = numeric.quantile(.25), numeric.quantile(.75)
    iqr = q3 - q1
    return {
        'shape': list(frame.shape), 'duplicate_rows': int(frame.duplicated().sum()),
        'duplicate_ids': int(frame.Student_ID.duplicated().sum()),
        'missing_values': frame.isna().sum().astype(int).to_dict(),
        'data_types': frame.dtypes.astype(str).to_dict(),
        'iqr_outlier_counts': ((numeric < q1 - 1.5 * iqr) | (numeric > q3 + 1.5 * iqr)).sum().astype(int).to_dict(),
    }

def fit_preprocessing(frame):
    # Exact duplicates are removed before fitting so repeated rows do not gain weight.
    clean = validate_frame(frame).drop_duplicates().reset_index(drop=True)
    if clean.Student_ID.duplicated().any():
        raise ValueError('Training data has conflicting duplicate Student_ID values.')
    if clean[FEATURES].isna().all().any():
        raise ValueError('A training feature is entirely missing; provide observed values.')
    # Median is robust to extreme observations. Keep plausible IQR outliers:
    # unusual study habits are relevant rather than automatically erroneous.
    imputer = SimpleImputer(strategy='median')
    filled = imputer.fit_transform(clean[FEATURES])
    scaler = StandardScaler()
    scaled = scaler.fit_transform(filled)
    return clean, filled, scaled, imputer, scaler
