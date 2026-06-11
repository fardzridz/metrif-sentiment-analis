"""Tab training model per sesi pengguna."""

import warnings

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.data_service import validate_training_data
from src.model_service import clear_session_model, set_session_model
from src.preprocessing import preprocess
from src.ui import section_header


def render_training_page(df) -> None:
    section_header("3. Training dan Evaluasi Model")
    st.info(
        "Model dilatih dari dataset upload saat ini dan hanya disimpan pada sesi pengguna. "
        "Model tidak mengganti file server dan tidak dibagikan ke pengguna lain."
    )

    st.markdown("#### ⚙️ Konfigurasi Training")
    cfg1, cfg2, cfg3 = st.columns(3)
    with cfg1:
        test_size = st.slider("Ukuran Data Test (%)", 10, 40, 20, key="tr_test") / 100
        use_stemming = st.checkbox(
            "Gunakan Stemming",
            value=False,
            key="tr_stem",
            help="Konfigurasi yang sama otomatis digunakan saat prediksi.",
        )
    with cfg2:
        max_feat = st.select_slider("Max Features TF-IDF", options=[5000, 10000, 15000, 20000, 25000], value=15000, key="tr_feat")
        ngram_opt = st.selectbox("N-gram Range", ["(1,1) Unigram", "(1,2) Bigram", "(1,3) Trigram"], index=1, key="tr_ngram")
    with cfg3:
        models_to_train = st.multiselect(
            "Pilih Model:",
            [
                "Logistic Regression",
                "Complement Naive Bayes",
                "Linear SVM",
                "Extra Trees",
                "Voting Ensemble",
            ],
            default=["Logistic Regression", "Complement Naive Bayes", "Linear SVM", "Voting Ensemble"],
            key="tr_models",
        )
    ngram = {"(1,1) Unigram": (1, 1), "(1,2) Bigram": (1, 2), "(1,3) Trigram": (1, 3)}[ngram_opt]
    st.divider()

    readiness = validate_training_data(df, test_size)
    _render_training_readiness(readiness)

    if st.button("🚀 Mulai Training", type="primary", key="tr_start", use_container_width=False):
        if not readiness.is_valid:
            st.error("Training diblokir sampai seluruh syarat kesiapan data terpenuhi.")
            return
        _run_training(df, test_size, max_feat, ngram, models_to_train, use_stemming)


def _render_training_readiness(report) -> None:
    with st.expander("Validasi kesiapan training", expanded=not report.is_valid):
        if report.is_valid:
            st.success("Dataset memenuhi syarat minimum training tiga kelas.")
        for message in report.errors:
            st.error(message)
        for message in report.warnings:
            st.warning(message)
        class_counts = report.stats.get("class_counts", {})
        if class_counts:
            st.dataframe(
                pd.DataFrame(
                    [{"Sentimen": name, "Jumlah": count} for name, count in class_counts.items()]
                ),
                use_container_width=True,
                hide_index=True,
            )


def _run_training(df, test_size, max_feat, ngram, models_to_train, use_stemming) -> None:
    if not models_to_train:
        st.error("Pilih minimal satu model.")
        return

    clear_session_model()
    warnings.filterwarnings("ignore")
    with st.spinner("Menyiapkan teks menggunakan konfigurasi preprocessing training..."):
        X = df["review_text"].astype(str).apply(
            lambda text: preprocess(text, use_stemming=use_stemming)
        )
    empty_texts = X.str.strip().eq("")
    if empty_texts.any():
        st.error(
            f"Training diblokir karena {int(empty_texts.sum())} ulasan menjadi kosong "
            "dengan konfigurasi preprocessing yang dipilih."
        )
        return
    y = df["sentiment"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    st.info(f"Training: **{len(X_train):,}** | Testing: **{len(X_test):,}**")

    defs = _build_model_definitions(max_feat, ngram)
    results_tr = {}
    trained_m = {}
    labels_ord = sorted(y.unique())
    pbar = st.progress(0, text="Memulai...")
    stat_box = st.empty()
    log_box = st.empty()
    log_lines = []

    for idx, name in enumerate(models_to_train):
        if name not in defs:
            log_lines.append(f"⚠️ {name} tidak tersedia")
            log_box.code("\n".join(log_lines))
            continue
        pbar.progress(idx / len(models_to_train), text=f"Training: {name}...")
        stat_box.info(f"⏳ Melatih **{name}**...")
        try:
            pipe = defs[name]
            pipe.fit(X_train, y_train)
            yp = pipe.predict(X_test)
            acc = accuracy_score(y_test, yp)
            f1_macro = f1_score(y_test, yp, average="macro", zero_division=0)
            f1_weighted = f1_score(y_test, yp, average="weighted", zero_division=0)
            results_tr[name] = {
                "accuracy": acc,
                "f1_macro": f1_macro,
                "f1_weighted": f1_weighted,
                "y_pred": yp.tolist(),
            }
            trained_m[name] = pipe
            log_lines.append(
                f"✅ {name}: Akurasi={acc*100:.2f}%  Macro F1={f1_macro*100:.2f}%"
            )
        except Exception as exc:
            log_lines.append(f"❌ {name} gagal: {exc}")
        log_box.code("\n".join(log_lines))

    if results_tr:
        pbar.progress(1.0, text="Selesai!")
        stat_box.success("Training selesai dan model terbaik aktif.")
        _save_and_render_training_results(
            results_tr, trained_m, labels_ord, y_test, use_stemming, test_size
        )
    else:
        pbar.progress(1.0, text="Gagal")
        stat_box.error("Semua model gagal dilatih. Periksa log training.")


def _build_model_definitions(max_feat: int, ngram: tuple[int, int]) -> dict:
    def _tfidf():
        return TfidfVectorizer(max_features=max_feat, ngram_range=ngram, min_df=2, sublinear_tf=True)

    defs = {
        "Logistic Regression": Pipeline(
            [
                ("tfidf", _tfidf()),
                (
                    "clf",
                    LogisticRegression(
                        C=5.0,
                        max_iter=2000,
                        solver="lbfgs",
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "Complement Naive Bayes": Pipeline([("tfidf", _tfidf()), ("clf", ComplementNB(alpha=0.05))]),
        "Linear SVM": Pipeline(
            [
                ("tfidf", _tfidf()),
                (
                    "clf",
                    CalibratedClassifierCV(
                        LinearSVC(
                            C=1.5,
                            max_iter=3000,
                            class_weight="balanced",
                            random_state=42,
                        )
                    ),
                ),
            ]
        ),
        "Extra Trees": Pipeline(
            [
                ("tfidf", _tfidf()),
                (
                    "clf",
                    ExtraTreesClassifier(
                        n_estimators=200,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "Voting Ensemble": Pipeline(
            [
                ("tfidf", _tfidf()),
                (
                    "clf",
                    VotingClassifier(
                        estimators=[
                            (
                                "lr",
                                LogisticRegression(
                                    C=5.0,
                                    max_iter=2000,
                                    solver="lbfgs",
                                    class_weight="balanced",
                                    random_state=42,
                                ),
                            ),
                            (
                                "svm",
                                CalibratedClassifierCV(
                                    LinearSVC(
                                        C=1.5,
                                        max_iter=3000,
                                        class_weight="balanced",
                                        random_state=42,
                                    )
                                ),
                            ),
                            ("cnb", ComplementNB(alpha=0.05)),
                        ],
                        voting="soft",
                    ),
                ),
            ]
        ),
    }
    return defs


def _save_and_render_training_results(
    results_tr, trained_m, labels_ord, y_test, use_stemming, test_size
) -> None:
    best_name = max(results_tr, key=lambda name: results_tr[name]["f1_macro"])
    model_info = {
        "best_model_name": best_name,
        "results": {
            name: {
                "accuracy": result["accuracy"],
                "f1_macro": result["f1_macro"],
                "f1_weighted": result["f1_weighted"],
            }
            for name, result in results_tr.items()
        },
        "label_classes": labels_ord,
        "use_stemming": use_stemming,
        "test_size": test_size,
        "selection_metric": "f1_macro",
    }
    set_session_model(trained_m[best_name], model_info)

    st.markdown("#### 📊 Hasil Evaluasi")
    res_df = pd.DataFrame(
        [
            {
                "Model": k,
                "Akurasi (%)": round(v["accuracy"] * 100, 2),
                "Macro F1 (%)": round(v["f1_macro"] * 100, 2),
                "Weighted F1 (%)": round(v["f1_weighted"] * 100, 2),
                "Terbaik": "★" if k == best_name else "",
            }
            for k, v in sorted(
                results_tr.items(), key=lambda item: item[1]["f1_macro"], reverse=True
            )
        ]
    )
    st.dataframe(
        res_df.style.background_gradient(subset=["Macro F1 (%)"], cmap="Greens"),
        use_container_width=True,
    )

    fig_tr1 = go.Figure()
    fig_tr1.add_trace(go.Bar(name="Akurasi (%)", x=res_df["Model"], y=res_df["Akurasi (%)"], marker_color="#3498db"))
    fig_tr1.add_trace(go.Bar(name="Macro F1 (%)", x=res_df["Model"], y=res_df["Macro F1 (%)"], marker_color="#2ecc71"))
    fig_tr1.update_layout(
        barmode="group",
        title="Perbandingan Performa Model",
        yaxis=dict(range=[50, 105]),
        xaxis_tickangle=-20,
        height=400,
    )
    st.plotly_chart(fig_tr1, use_container_width=True, key="t8_bar")

    st.markdown(f"#### 🔢 Confusion Matrix — {best_name}")
    yp_best = results_tr[best_name]["y_pred"]
    cm = confusion_matrix(y_test, yp_best, labels=labels_ord)
    fig_tr2 = px.imshow(
        cm,
        x=labels_ord,
        y=labels_ord,
        color_continuous_scale="Blues",
        text_auto=True,
        labels={"x": "Prediksi", "y": "Asli"},
        title=f"Confusion Matrix — {best_name}",
    )
    fig_tr2.update_layout(height=400)
    st.plotly_chart(fig_tr2, use_container_width=True, key="t8_cm")

    st.markdown(f"#### 📋 Classification Report — {best_name}")
    rpt = classification_report(y_test, yp_best, output_dict=True, zero_division=0)
    st.dataframe(
        pd.DataFrame(rpt).transpose().round(3).style.background_gradient(
            subset=["precision", "recall", "f1-score"], cmap="Greens"
        ),
        use_container_width=True,
    )

    st.success(
        f"✅ Model **{best_name}** "
        f"(Macro F1: {results_tr[best_name]['f1_macro']*100:.2f}%) "
        "aktif untuk prediksi selama sesi ini."
    )
