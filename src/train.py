"""
train.py
Training multi-model sentiment analysis dengan ensemble dan hyperparameter tuning.
Target akurasi: >90%
"""

import os
import joblib
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    VotingClassifier, StackingClassifier, ExtraTreesClassifier
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[INFO] XGBoost tidak tersedia, dilewati.")

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("[INFO] LightGBM tidak tersedia, dilewati.")

from preprocessing import load_and_prepare

# ─── Konfigurasi ────────────────────────────────────────────────────────────
DATASET_PATH = '../data/dataset.csv'
MODEL_DIR    = '../models'
USE_STEMMING = True
TEST_SIZE    = 0.2
RANDOM_STATE = 42
# ────────────────────────────────────────────────────────────────────────────


def get_tfidf(ngram=(1, 2), max_feat=15000):
    return TfidfVectorizer(
        max_features=max_feat,
        ngram_range=ngram,
        min_df=2,
        sublinear_tf=True,
        analyzer='word'
    )


def build_models():
    """Definisi semua model yang akan diuji."""
    models = {}

    # ── 1. Logistic Regression (tuned) ──────────────────────────────────────
    models['Logistic Regression'] = Pipeline([
        ('tfidf', get_tfidf((1, 3), 20000)),
        ('clf', LogisticRegression(
            C=5.0, max_iter=2000,
            solver='lbfgs', random_state=RANDOM_STATE
        ))
    ])

    # ── 2. Complement Naive Bayes (lebih baik dari MultinomialNB untuk imbalanced) ──
    models['Complement Naive Bayes'] = Pipeline([
        ('tfidf', get_tfidf((1, 2), 15000)),
        ('clf', ComplementNB(alpha=0.05))
    ])

    # ── 3. Linear SVM (tuned) ───────────────────────────────────────────────
    models['Linear SVM'] = Pipeline([
        ('tfidf', get_tfidf((1, 3), 20000)),
        ('clf', CalibratedClassifierCV(
            LinearSVC(C=1.5, max_iter=3000, random_state=RANDOM_STATE)
        ))
    ])

    # ── 4. Extra Trees ──────────────────────────────────────────────────────
    models['Extra Trees'] = Pipeline([
        ('tfidf', get_tfidf((1, 2), 10000)),
        ('clf', ExtraTreesClassifier(
            n_estimators=300, max_depth=None,
            min_samples_split=2, random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])

    # ── 5. XGBoost ──────────────────────────────────────────────────────────
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = Pipeline([
            ('tfidf', get_tfidf((1, 2), 10000)),
            ('clf', XGBClassifier(
                n_estimators=300, max_depth=6,
                learning_rate=0.1, subsample=0.8,
                use_label_encoder=False,
                eval_metric='mlogloss',
                random_state=RANDOM_STATE,
                n_jobs=-1
            ))
        ])

    # ── 6. LightGBM ─────────────────────────────────────────────────────────
    if LIGHTGBM_AVAILABLE:
        models['LightGBM'] = Pipeline([
            ('tfidf', get_tfidf((1, 2), 10000)),
            ('clf', LGBMClassifier(
                n_estimators=500, max_depth=-1,
                learning_rate=0.05, num_leaves=63,
                random_state=RANDOM_STATE,
                n_jobs=-1, verbose=-1
            ))
        ])

    # ── 7. Voting Ensemble (LR + SVM + CNB) ─────────────────────────────────
    models['Voting Ensemble'] = Pipeline([
        ('tfidf', get_tfidf((1, 3), 20000)),
        ('clf', VotingClassifier(
            estimators=[
                ('lr', LogisticRegression(C=5.0, max_iter=2000, solver='lbfgs',
                                          random_state=RANDOM_STATE)),
                ('svm', CalibratedClassifierCV(
                    LinearSVC(C=1.5, max_iter=3000, random_state=RANDOM_STATE)
                )),
                ('cnb', ComplementNB(alpha=0.05)),
            ],
            voting='soft'
        ))
    ])

    return models


def evaluate_model(name, pipeline, X_train, X_test, y_train, y_test):
    """Latih dan evaluasi satu model."""
    print(f"\n  Training: {name}...")
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average='weighted')

    print(f"  Akurasi : {acc*100:.2f}%  |  F1-Score: {f1*100:.2f}%")
    return pipeline, acc, f1, y_pred


def plot_model_comparison(results: dict):
    """Bar chart perbandingan semua model."""
    names = list(results.keys())
    accs  = [v['accuracy'] * 100 for v in results.values()]
    f1s   = [v['f1'] * 100 for v in results.values()]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width/2, accs, width, label='Accuracy (%)', color='#3498db', edgecolor='black')
    bars2 = ax.bar(x + width/2, f1s,  width, label='F1-Score (%)', color='#2ecc71', edgecolor='black')

    ax.set_xlabel('Model')
    ax.set_ylabel('Score (%)')
    ax.set_title('Perbandingan Performa Model')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right')
    ax.set_ylim(50, 105)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    for bar in bars1:
        ax.annotate(f'{bar.get_height():.1f}',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8)
    for bar in bars2:
        ax.annotate(f'{bar.get_height():.1f}',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8)

    plt.tight_layout()
    os.makedirs('../reports', exist_ok=True)
    plt.savefig('../reports/perbandingan_model.png', dpi=150)
    plt.show()
    print("Grafik disimpan di ../reports/perbandingan_model.png")


def plot_confusion_matrix(cm, labels, title):
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('Label Asli')
    plt.xlabel('Label Prediksi')
    plt.tight_layout()
    os.makedirs('../reports', exist_ok=True)
    safe_title = title.lower().replace(' ', '_').replace('-', '_')
    plt.savefig(f'../reports/cm_{safe_title}.png', dpi=150)
    plt.close()


def main():
    print("=" * 60)
    print("   SENTIMENT ANALYSIS — MULTI-MODEL TRAINING")
    print("=" * 60)

    if not os.path.exists(DATASET_PATH):
        print(f"[ERROR] Dataset tidak ditemukan: {DATASET_PATH}")
        return

    # ── Load & Preprocessing ─────────────────────────────────────────────────
    df = load_and_prepare(DATASET_PATH, use_stemming=USE_STEMMING)

    X = df['clean_text']
    y = df['sentiment']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nData training : {len(X_train):,} | Data testing: {len(X_test):,}")

    # ── Training Semua Model ─────────────────────────────────────────────────
    models  = build_models()
    results = {}
    trained = {}

    print(f"\n{'='*60}")
    print(f"  Melatih {len(models)} model...")
    print(f"{'='*60}")

    for name, pipeline in models.items():
        try:
            pipe, acc, f1, y_pred = evaluate_model(
                name, pipeline, X_train, X_test, y_train, y_test
            )
            results[name] = {'accuracy': acc, 'f1': f1}
            trained[name] = pipe

            # Confusion matrix per model
            labels = sorted(y.unique())
            cm = confusion_matrix(y_test, y_pred, labels=labels)
            plot_confusion_matrix(cm, labels, f'CM {name}')

        except Exception as e:
            print(f"  [SKIP] {name} gagal: {e}")

    # ── Ringkasan ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  RINGKASAN HASIL")
    print(f"{'='*60}")
    print(f"  {'Model':<28} {'Accuracy':>10} {'F1-Score':>10}")
    print(f"  {'-'*50}")

    best_name = max(results, key=lambda k: results[k]['accuracy'])
    for name, scores in sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
        marker = " ★ TERBAIK" if name == best_name else ""
        print(f"  {name:<28} {scores['accuracy']*100:>9.2f}%  {scores['f1']*100:>9.2f}%{marker}")

    # ── Simpan Model Terbaik ─────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    best_model = trained[best_name]
    joblib.dump(best_model, f'{MODEL_DIR}/best_model.pkl')

    # Simpan semua model
    all_models_path = f'{MODEL_DIR}/all_models.pkl'
    joblib.dump(trained, all_models_path)

    # Simpan info
    info = {
        'best_model_name': best_name,
        'results': results,
        'label_classes': sorted(y.unique()),
        'use_stemming': USE_STEMMING
    }
    joblib.dump(info, f'{MODEL_DIR}/model_info.pkl')

    print(f"\nModel terbaik ({best_name}) disimpan di: {MODEL_DIR}/best_model.pkl")
    print(f"Semua model disimpan di: {all_models_path}")

    # ── Grafik Perbandingan ──────────────────────────────────────────────────
    plot_model_comparison(results)

    # ── Classification Report Model Terbaik ─────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Classification Report — {best_name}")
    print(f"{'='*60}")
    y_pred_best = best_model.predict(X_test)
    print(classification_report(y_test, y_pred_best))

    print("\nTraining selesai!")


if __name__ == "__main__":
    main()
