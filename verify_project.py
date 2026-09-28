"""Run meaningful pipeline, upload, download, and all-page dashboard checks."""
import io
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest
from src.clustering import load_bundle, assign_students, csv_bytes, ensure_artifacts
from src.data_generator import generate_dataset
from src.preprocessing import ROOT, FEATURES, validate_frame

class ProjectChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_bundle()
        cls.raw = pd.read_csv(ROOT / 'student_learning_behavior.csv')

    def test_artifacts_and_assignment_consistency(self):
        self.assertGreaterEqual(len(self.raw), 2000)
        self.assertEqual(len(FEATURES), 10)
        self.assertGreater(self.raw.isna().sum().sum(), 0)
        assigned = assign_students(self.raw, self.bundle)
        saved = pd.read_csv(ROOT / 'outputs/clustered_students.csv')
        np.testing.assert_array_equal(assigned.Cluster, saved.Cluster)
        summary = pd.read_csv(ROOT / 'outputs/cluster_summary.csv')
        self.assertEqual(summary.Student_Count.sum(), len(saved))
        evaluation = pd.read_csv(ROOT / 'outputs/k_evaluation.csv')
        self.assertEqual(self.bundle['metrics']['selected_k'], int(evaluation.loc[evaluation.Silhouette.idxmax(), 'K']))
        self.assertEqual(len(list((ROOT / 'outputs/charts').glob('*.png'))), 17)
        self.assertTrue(all(path.stat().st_size > 1000 for path in (ROOT / 'outputs/charts').iterdir()))

    def test_upload_validation_and_csv_roundtrip(self):
        sample = self.raw.head(8).copy()
        sample.loc[:, FEATURES] = np.nan
        assigned = assign_students(sample, self.bundle)
        self.assertFalse(assigned.Cluster.isna().any())
        roundtrip = pd.read_csv(io.BytesIO(csv_bytes(assigned)))
        self.assertEqual(roundtrip.Student_ID.tolist(), assigned.Student_ID.tolist())
        self.assertEqual(roundtrip.Learning_Behaviour.tolist(), assigned.Learning_Behaviour.tolist())
        for invalid in [sample.drop(columns='Sleep_Hours'), sample.iloc[:0], sample.assign(Age=99), sample.assign(Sleep_Hours=np.inf), sample.assign(Student_ID='')]:
            with self.assertRaises(ValueError):
                validate_frame(invalid)
        with self.assertRaises(ValueError):
            validate_frame(sample.assign(Average_Test_Score='invalid'))
        malicious = assigned.assign(Student_ID='=1+1')
        self.assertIn("'=1+1", csv_bytes(malicious).decode('utf-8-sig'))

    def test_all_pages_and_interactions(self):
        app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=45).run()
        self.assertFalse(app.exception)
        for page in ['dataset', 'eda', 'kmeans', 'clusters', 'visualization', 'prediction', 'upload']:
            app.switch_page(f'app_pages/{page}.py').run()
            self.assertFalse(app.exception, page)
            if page == 'eda':
                for feature in ['Attendance_Percentage', 'Assignment_Completion_Percentage', 'Previous_Semester_Marks', 'Average_Test_Score']:
                    app.selectbox[0].select(feature).run()
                    self.assertFalse(app.exception)
            if page == 'clusters':
                app.selectbox[0].select(1).run()
                self.assertTrue((app.dataframe[1].value.Cluster == 1).all())
                self.assertEqual(len(app.get('download_button')), 3)
            if page == 'visualization':
                app.checkbox[0].check().run()
                self.assertFalse(app.exception)
            if page == 'prediction':
                app.button[0].click().run()
                self.assertTrue(app.success)
                self.assertFalse(app.exception)

    def test_upload_interface(self):
        # AppTest does not expose file_uploader.set_value. Exercise the same
        # upload panel with in-memory CSV bytes, including the rendered download.
        script = '''
import io
import streamlit as st
from src.dashboard import initialize, upload_panel
from src.preprocessing import ROOT
initialize()
upload_panel(io.BytesIO((ROOT / 'student_learning_behavior.csv').read_bytes()))
'''
        app = AppTest.from_string(script, default_timeout=45).run()
        self.assertFalse(app.exception)
        self.assertTrue(app.success)
        self.assertEqual(len(app.dataframe[0].value), len(self.raw))
        self.assertEqual(len(app.get('download_button')), 1)
        bad = script.replace("(ROOT / 'student_learning_behavior.csv').read_bytes()", "b'Wrong,Columns\\n1,2\\n'")
        app = AppTest.from_string(bad, default_timeout=45).run()
        self.assertTrue(app.error)
        self.assertFalse(app.exception)

    def test_fresh_folder_and_missing_model_recovery(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            metrics = ensure_artifacts(root)
            self.assertEqual(metrics['records'], 2400)
            self.assertEqual(metrics['selected_k'], self.bundle['metrics']['selected_k'])
            self.assertAlmostEqual(metrics['silhouette'], self.bundle['metrics']['silhouette'], places=9)
            (root / 'models/scaler.pkl').unlink()
            ensure_artifacts(root)
            self.assertTrue((root / 'models/scaler.pkl').exists())

if __name__ == '__main__':
    unittest.main(verbosity=2)
