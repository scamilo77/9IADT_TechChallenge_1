from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier 
from sklearn.svm import SVC

def build_diabetes_pipeline():
    pipeline = Pipeline(
        steps=[
        ('scaler', StandardScaler()),
        ('model', 
         LogisticRegression(
            max_iter=1000,
            random_state=42,
            solver='liblinear'
        ))
    ])
    return pipeline

def build_diabetes_pipeline_rf():
    pipeline = Pipeline(
        steps=[
        ('model', 
         RandomForestClassifier(
            n_estimators=200,
            random_state=42,   
            max_depth=None,
            min_samples_split=5
         ))
    ])
    return pipeline


def build_breast_cancer_pipeline():
    pipeline = Pipeline(
        steps=[
        ('scaler', StandardScaler()),
        ('model', 
         LogisticRegression(
            max_iter=2000,
            random_state=42,
            solver='lbfgs'
        ))
    ])
    return pipeline

def build_breast_cancer_pipeline_rf():
    pipeline = Pipeline(
        steps=[
        ('scaler', StandardScaler()),
         ('model', 
          SVC(
            C=1.0,
            kernel='rbf',
            random_state=42,
            gamma='scale',
            probability=True
         ))
    ])
    return pipeline