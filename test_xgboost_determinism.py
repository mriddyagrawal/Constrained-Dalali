import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score

# Dummy data
np.random.seed(42)
X_train = pd.DataFrame(np.random.randn(100, 20))
y_train = pd.Series(np.random.randint(0, 2, 100))
X_test = pd.DataFrame(np.random.randn(20, 20))

RANDOM_STATE = 42

def train_and_eval():
    lr_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf',    LogisticRegression(
            C=0.1,
            max_iter=1000,
            class_weight='balanced',
            random_state=RANDOM_STATE
        ))
    ])
    lr_pipe.fit(X_train, y_train)

    rf_clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=4,
        min_samples_leaf=20,
        max_features='sqrt',
        class_weight='balanced',
        n_jobs=-1,
        random_state=RANDOM_STATE
    )
    rf_clf.fit(X_train, y_train)

    scale_pos = float((y_train == 0).sum()) / float((y_train == 1).sum())
    xgb_clf = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        eval_metric='logloss',
        random_state=RANDOM_STATE,
        verbosity=0
    )
    xgb_clf.fit(X_train, y_train)

    ensemble = VotingClassifier(
        estimators=[
            ('lr',  lr_pipe),
            ('rf',  rf_clf),
            ('xgb', xgb_clf)
        ],
        voting='soft',
        weights=[1, 2, 2]   # downweight LR slightly
    )
    ensemble.fit(X_train, y_train)
    return ensemble.predict_proba(X_test)[:, 1].mean()

p1 = train_and_eval()
p2 = train_and_eval()
print(p1, p2, p1 == p2)
