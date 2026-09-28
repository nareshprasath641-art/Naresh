"""Generate reproducible synthetic records; no real student information is used."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def generate_dataset(n_students=2400, seed=42, root=ROOT):
    if n_students < 2000:
        raise ValueError('Generate at least 2,000 students.')
    rng = np.random.default_rng(seed)
    # A continuous engagement factor creates overlapping habits, not preset labels.
    engagement = rng.beta(2, 2, n_students)
    def noisy(base, slope, noise, low, high):
        return np.clip(base + slope * engagement + rng.normal(0, noise, n_students), low, high).round(2)
    frame = pd.DataFrame({
        'Student_ID': [f'STU{i:05d}' for i in range(1, n_students + 1)],
        'Age': rng.integers(17, 26, n_students),
        'Study_Hours_Per_Day': noisy(.3, 5.4, .75, 0, 8),
        'Attendance_Percentage': noisy(42, 57, 8, 15, 100),
        'Assignment_Completion_Percentage': noisy(22, 78, 11, 0, 100),
        'Online_Learning_Hours': noisy(.2, 2.8, .65, 0, 5),
        'Class_Interactions': noisy(1, 22, 4, 0, 40).round(),
        'Previous_Semester_Marks': noisy(32, 62, 9, 10, 100),
        'Sleep_Hours': noisy(6.6, .7, .85, 4, 10),
        'Extracurricular_Hours': np.clip(rng.gamma(2, .65, n_students), 0, 4).round(2),
        'Test_Preparation_Hours': noisy(.1, 3.7, .7, 0, 6),
        'Average_Test_Score': noisy(27, 69, 8, 5, 100),
    })
    # About 1.5% of behavioural measurements are missing, independently.
    for column in frame.columns[2:]:
        frame.loc[rng.random(n_students) < .015, column] = np.nan
    root = Path(root)
    (root / 'data').mkdir(parents=True, exist_ok=True)
    for path in (root / 'student_learning_behavior.csv', root / 'data/student_learning_behavior.csv'):
        frame.to_csv(path, index=False)
    return frame

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows', type=int, default=2400)
    args = parser.parse_args()
    print(f'Generated {len(generate_dataset(args.rows)):,} synthetic students in {ROOT}')
