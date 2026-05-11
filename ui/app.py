"""
Intelligent Reading Comprehension and Quiz Generation System
Streamlit UI — 4 required screens:
  Screen 1: Article Input
  Screen 2: Question & Answer Quiz View
  Screen 3: Hint Panel
  Screen 4: Developer / Analytics Dashboard
"""

import os
import re
import time
import random
import csv
import io
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, confusion_matrix
)

# ── page config (must be first st call) ─────────────────────
st.set_page_config(
    page_title="RACE Reading Comprehension System",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── paths ────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_A    = os.path.join(BASE_DIR, "models", "model_a")
MODEL_B    = os.path.join(BASE_DIR, "models", "model_b")
DATA_DIR   = os.path.join(BASE_DIR, "data", "processed")

# ── helper: load models with cache ──────────────────────────
@st.cache_resource(show_spinner=False)
def load_model_a():
    """load all model A artefacts once"""
    try:
        vec   = joblib.load(os.path.join(MODEL_A, "vectorizer.pkl"))
        lr    = joblib.load(os.path.join(MODEL_A, "logistic_regression.pkl"))
        svm   = joblib.load(os.path.join(MODEL_A, "svm_classifier.pkl"))
        nb    = joblib.load(os.path.join(MODEL_A, "naive_bayes.pkl"))
        sc    = joblib.load(os.path.join(MODEL_A, "scaler.pkl"))
        return vec, lr, svm, nb, sc
    except Exception as e:
        return None, None, None, None, None


@st.cache_resource(show_spinner=False)
def load_model_b():
    """load all model B artefacts once"""
    try:
        dist  = joblib.load(os.path.join(MODEL_B, "distractor_ranker.pkl"))
        hint  = joblib.load(os.path.join(MODEL_B, "hint_scorer.pkl"))
        return dist, hint
    except Exception as e:
        return None, None


@st.cache_data(show_spinner=False)
def load_sample_data():
    """load test split for random sample feature"""
    try:
        path = os.path.join(DATA_DIR, "test_clean.csv")
        df = pd.read_csv(path)
        # drop rows with any NaN in key columns
        df = df.dropna(subset=["article", "question", "A", "B", "C", "D", "answer"])
        df = df.reset_index(drop=True)
        return df
    except Exception:
        return None


# ── inference helpers ────────────────────────────────────────
FEATURE_COLS = ["overlap", "opt_len", "correct_len", "len_diff", "char_sim"]
HINT_COLS    = ["q_overlap", "ans_overlap", "position", "sent_len"]


def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def verify_answer(article, question, option, vectorizer, lr, svm, nb, scaler):
    """soft-vote LR + NB; SVM has no predict_proba so weight 1/0"""
    combined = f"{article} {article} {question} {option}"
    X        = vectorizer.transform([combined])
    X_sc     = scaler.transform(X)

    p_lr = lr.predict_proba(X)[0][1]
    p_nb = nb.predict_proba(X_sc)[0][1]
    # SVM decision function mapped to [0,1] via sigmoid
    raw_svm = svm.decision_function(X)[0]
    p_svm   = 1 / (1 + np.exp(-raw_svm))

    ensemble = (p_lr + p_nb + p_svm) / 3.0
    return ensemble, p_lr, p_nb, p_svm


def get_phrase_candidates(article, correct_answer, n=20):
    sentences  = [s.strip() for s in article.split(".") if len(s.split()) > 5]
    candidates = []
    correct_words = set(correct_answer.lower().split())
    for sent in sentences:
        words = sent.split()
        for i in range(len(words) - 1):
            phrase = f"{words[i]} {words[i+1]}"
            phrase = re.sub(r"[^a-zA-Z\s]", "", phrase).strip()
            if (len(phrase) > 5
                    and phrase.lower() != correct_answer.lower()
                    and not set(phrase.lower().split()) & correct_words):
                candidates.append(phrase)
        for i in range(len(words) - 2):
            phrase = f"{words[i]} {words[i+1]} {words[i+2]}"
            phrase = re.sub(r"[^a-zA-Z\s]", "", phrase).strip()
            if (len(phrase) > 8
                    and phrase.lower() != correct_answer.lower()
                    and not set(phrase.lower().split()) & correct_words):
                candidates.append(phrase)
    seen, unique = set(), []
    for c in candidates:
        if c.lower() not in seen:
            seen.add(c.lower())
            unique.append(c)
    return unique[:n]


def generate_distractors(article, correct, ranker, n=3):
    candidates = get_phrase_candidates(article, correct, n=30)
    if not candidates:
        return ["option one", "option two", "option three"][:n]
    if len(candidates) < n:
        return (candidates + ["unknown"] * n)[:n]

    features = []
    c_words  = set(correct.lower().split())
    for c in candidates:
        o_words = set(c.lower().split())
        features.append([
            len(c_words & o_words),
            len(c.split()),
            len(correct.split()),
            abs(len(c.split()) - len(correct.split())),
            len(set(correct) & set(c)) / max(len(set(correct)), 1),
        ])
    feat_df = pd.DataFrame(features, columns=FEATURE_COLS)
    scores  = ranker.predict_proba(feat_df)[:, 1]
    top_idx = np.argsort(scores)[-n:][::-1]
    return [candidates[i] for i in top_idx]


def generate_hints(article, question, hint_model, n=3):
    sentences = [s.strip() for s in article.split(".") if len(s.split()) > 4]
    if not sentences:
        return ["Read the passage carefully."] * n
    q_words  = set(question.lower().split())
    features = []
    for i, sent in enumerate(sentences):
        s_words = set(sent.lower().split())
        features.append([
            len(q_words & s_words),
            0,
            i / max(len(sentences), 1),
            len(sent.split()),
        ])
    feat_df = pd.DataFrame(features, columns=HINT_COLS)
    scores  = hint_model.predict_proba(feat_df)[:, 2]
    top_idx = np.argsort(scores)[-n:]
    top_idx = sorted(top_idx)
    hints   = [sentences[i] for i in top_idx if i < len(sentences)]
    hints.sort(key=lambda x: len(x.split()))
    while len(hints) < n:
        hints.append("Focus on the key ideas in the passage.")
    return hints[:n]


def generate_question(article):
    """template-based question generation with Wh-word prefixes"""
    sentences = [s.strip() for s in article.split(".") if len(s.split()) > 6]
    if not sentences:
        return "What is the main idea of the passage?"
    # pick sentence with most content words
    best = max(sentences, key=lambda s: len(s.split()))
    words = best.split()

    # simple keyword-to-Wh mapping
    wh_map = {
        "when": ["year", "century", "ago", "time", "date", "day", "month"],
        "where": ["city", "country", "place", "location", "area", "region", "town"],
        "who": ["he", "she", "they", "his", "her", "person", "man", "woman", "people"],
        "why": ["because", "reason", "cause", "therefore", "since"],
        "how": ["way", "method", "process", "step"],
    }
    chosen_wh = "what"
    for wh, keywords in wh_map.items():
        if any(w.lower() in keywords for w in words):
            chosen_wh = wh
            break

    question = f"{chosen_wh.capitalize()} does the passage mainly discuss regarding {' '.join(words[1:5]).lower()}?"
    return question


# ── session state initialisation ────────────────────────────
def init_state():
    defaults = {
        "screen":         "input",
        "article":        "",
        "question":       "",
        "options":        {},           # {"A": ..., "B": ..., "C": ..., "D": ...}
        "correct_label":  "",           # "A" / "B" / "C" / "D"
        "user_answer":    None,
        "checked":        False,
        "hints_used":     0,
        "hints":          [],
        "inference_log":  [],           # list of dicts for dashboard
        "model_a_loaded": False,
        "model_b_loaded": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()

# ── load models ──────────────────────────────────────────────
vec, lr, svm, nb, scaler      = load_model_a()
dist_ranker, hint_model        = load_model_b()
test_df                        = load_sample_data()

models_a_ok = all(m is not None for m in [vec, lr, svm, nb, scaler])
models_b_ok = all(m is not None for m in [dist_ranker, hint_model])

# ── sidebar navigation ───────────────────────────────────────
with st.sidebar:
    st.title("RACE RC System")
    st.markdown("*Reading Comprehension & Quiz Generation*")
    st.divider()

    nav = st.radio(
        "Navigate to",
        ["Article Input", "Quiz View", "Hint Panel", "Analytics Dashboard"],
        index=["input", "quiz", "hint", "dashboard"].index(st.session_state["screen"])
              if st.session_state["screen"] in ["input","quiz","hint","dashboard"] else 0,
    )
    nav_map = {
        "Article Input":        "input",
        "Quiz View":            "quiz",
        "Hint Panel":           "hint",
        "Analytics Dashboard":  "dashboard",
    }
    if nav_map[nav] != st.session_state["screen"]:
        st.session_state["screen"] = nav_map[nav]
        st.rerun()

    st.divider()
    if models_a_ok:
        st.success("Model A loaded")
    else:
        st.warning("Model A not found — run notebooks first, copy pkl files to models/model_a/")
    if models_b_ok:
        st.success("Model B loaded")
    else:
        st.warning("Model B not found — run notebooks first, copy pkl files to models/model_b/")

    if test_df is not None:
        st.info(f"Dataset: {len(test_df)} test samples available")

    st.divider()
    st.caption("FAST-NUCES | AI Lab Project | Spring 2026")


# ═══════════════════════════════════════════════════════════
#  SCREEN 1 — Article Input
# ═══════════════════════════════════════════════════════════
if st.session_state["screen"] == "input":
    st.title("Article Input")
    st.markdown(
        "Paste a reading passage below, or load a random sample from the RACE test set. "
        "Once you click **Submit**, both Model A and Model B will run inference."
    )
    st.divider()

    col_left, col_right = st.columns([3, 1])
    with col_right:
        load_random = st.button("Load Random RACE Sample", use_container_width=True)
        if load_random:
            if test_df is not None:
                sample = test_df.sample(1, random_state=random.randint(0, 10000)).iloc[0]
                st.session_state["article"]       = sample["article"]
                st.session_state["question"]       = sample["question"]
                st.session_state["correct_label"]  = sample["answer"]
                # store original options (already cleaned text)
                st.session_state["_prefill_opts"] = {
                    "A": sample["A"], "B": sample["B"],
                    "C": sample["C"], "D": sample["D"],
                }
                st.success("RACE sample loaded.")
                st.rerun()
            else:
                st.error("test_clean.csv not found. Place it in data/processed/.")

    with col_left:
        article_input = st.text_area(
            "Reading Passage",
            value=st.session_state.get("article", ""),
            height=300,
            placeholder="Paste the article text here...",
        )

    st.divider()
    submit = st.button("Submit — Run Inference", type="primary", use_container_width=True)

    if submit:
        article = article_input.strip()
        if len(article) < 50:
            st.error("Please enter a passage of at least 50 characters.")
        elif not models_a_ok or not models_b_ok:
            st.error(
                "Models are not loaded. Run the training notebooks on Kaggle/Colab, "
                "download the .pkl files, and place them in models/model_a/ and models/model_b/."
            )
        else:
            with st.spinner("Running Model A and Model B inference…"):
                t0 = time.time()
                article_clean = clean_text(article)
                st.session_state["article"] = article_clean

                # question: use RACE original if loaded, else generate
                if st.session_state.get("question", ""):
                    question = clean_text(st.session_state["question"])
                else:
                    question = generate_question(article_clean)
                st.session_state["question"] = question

                # correct answer: use RACE original if loaded
                correct_label = st.session_state.get("correct_label", "")
                prefill_opts  = st.session_state.get("_prefill_opts", {})

                if correct_label and prefill_opts:
                    correct_text = clean_text(prefill_opts[correct_label])
                    wrong_labels = [l for l in ["A","B","C","D"] if l != correct_label]
                    wrong_texts  = [clean_text(prefill_opts[l]) for l in wrong_labels]
                else:
                    # generate from article: pick a sentence fragment as correct
                    sents = [s.strip() for s in article_clean.split(".") if len(s.split()) > 4]
                    correct_text  = sents[0] if sents else "the main topic"
                    correct_label = "A"
                    wrong_texts   = generate_distractors(article_clean, correct_text, dist_ranker, 3)
                    wrong_labels  = ["B", "C", "D"]

                # run distractor generation using Model B (replaces wrong options)
                distractors = generate_distractors(article_clean, correct_text, dist_ranker, 3)

                # assemble final options
                all_labels   = [correct_label] + wrong_labels
                all_texts_   = [correct_text] + distractors
                combined     = list(zip(all_labels, all_texts_))
                random.shuffle(combined)

                final_labels = ["A", "B", "C", "D"]
                final_opts   = {}
                new_correct  = ""
                for fl, (orig_lbl, txt) in zip(final_labels, combined):
                    final_opts[fl] = txt
                    if orig_lbl == correct_label:
                        new_correct = fl

                st.session_state["options"]       = final_opts
                st.session_state["correct_label"] = new_correct
                st.session_state["user_answer"]   = None
                st.session_state["checked"]        = False
                st.session_state["hints_used"]     = 0

                # generate hints
                hints = generate_hints(article_clean, question, hint_model, 3)
                st.session_state["hints"] = hints

                latency = round(time.time() - t0, 3)

                # log for dashboard
                st.session_state["inference_log"].append({
                    "timestamp":  time.strftime("%H:%M:%S"),
                    "article_len": len(article_clean.split()),
                    "question":    question[:60] + "…",
                    "correct":     new_correct,
                    "latency_s":   latency,
                    "model_a_pred": new_correct,
                })

            st.success(f"Inference done in {latency}s. Navigate to Quiz View to answer.")
            st.session_state["screen"] = "quiz"
            st.rerun()


# ═══════════════════════════════════════════════════════════
#  SCREEN 2 — Quiz View
# ═══════════════════════════════════════════════════════════
elif st.session_state["screen"] == "quiz":
    st.title("Question & Answer Quiz")
    st.divider()

    if not st.session_state.get("question"):
        st.info("No quiz loaded yet. Go to **Article Input** and submit a passage first.")
    else:
        article  = st.session_state["article"]
        question = st.session_state["question"]
        options  = st.session_state["options"]
        correct  = st.session_state["correct_label"]

        # show passage in expander
        with st.expander("Reading Passage", expanded=False):
            st.write(article)

        st.subheader("Question")
        st.markdown(f"**{question.capitalize()}**")
        st.divider()

        if not st.session_state["checked"]:
            chosen = st.radio(
                "Select your answer:",
                list(options.keys()),
                format_func=lambda k: f"**{k}.**  {options[k]}",
                index=None,
                key="answer_radio",
            )
            col_check, col_hint = st.columns([1, 1])
            with col_check:
                check_btn = st.button("Check Answer", type="primary", use_container_width=True)
            with col_hint:
                hint_btn = st.button("Get a Hint", use_container_width=True)

            if hint_btn:
                st.session_state["screen"] = "hint"
                st.rerun()

            if check_btn:
                if chosen is None:
                    st.warning("Please select an option before checking.")
                else:
                    st.session_state["user_answer"] = chosen
                    st.session_state["checked"]     = True

                    # run Model A verifier
                    if models_a_ok:
                        t0 = time.time()
                        ensemble, p_lr, p_nb, p_svm = verify_answer(
                            article, question, options[chosen],
                            vec, lr, svm, nb, scaler
                        )
                        lat = round(time.time() - t0, 4)
                        st.session_state["inference_log"].append({
                            "timestamp":  time.strftime("%H:%M:%S"),
                            "article_len": len(article.split()),
                            "question":    question[:60] + "…",
                            "correct":     correct,
                            "latency_s":   lat,
                            "model_a_pred": chosen,
                            "ensemble_score": round(ensemble, 4),
                            "p_lr":  round(p_lr, 4),
                            "p_nb":  round(p_nb, 4),
                            "p_svm": round(p_svm, 4),
                        })
                    st.rerun()

        else:
            # show result
            user_ans = st.session_state["user_answer"]
            is_right = user_ans == correct

            for label, text in options.items():
                if label == correct and label == user_ans:
                    st.success(f"**{label}. {text}** ← Your answer — Correct!")
                elif label == correct:
                    st.success(f"**{label}. {text}** ← Correct Answer")
                elif label == user_ans:
                    st.error(f"**{label}. {text}** ← Your answer — Incorrect")
                else:
                    st.markdown(f"**{label}.** {text}")

            st.divider()
            if is_right:
                st.balloons()
                st.markdown("### Well done! You got it right.")
            else:
                st.markdown(
                    f"### Not quite. The correct answer was **{correct}**: {options[correct]}"
                )

            # show model A scores if available
            last_log = st.session_state["inference_log"][-1] if st.session_state["inference_log"] else {}
            if "ensemble_score" in last_log:
                st.divider()
                st.markdown("**Model A Verification Scores** (probability that each chosen option is correct)")
                mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                mcol1.metric("Ensemble", f"{last_log['ensemble_score']:.3f}")
                mcol2.metric("LR",       f"{last_log['p_lr']:.3f}")
                mcol3.metric("NB",       f"{last_log['p_nb']:.3f}")
                mcol4.metric("SVM",      f"{last_log['p_svm']:.3f}")

            st.divider()
            col_retry, col_new = st.columns(2)
            with col_retry:
                if st.button("Try Again", use_container_width=True):
                    st.session_state["checked"]     = False
                    st.session_state["user_answer"] = None
                    st.rerun()
            with col_new:
                if st.button("New Article", use_container_width=True):
                    for k in ["article","question","options","correct_label","user_answer","checked",
                              "hints","hints_used","_prefill_opts"]:
                        if k in st.session_state:
                            del st.session_state[k]
                    st.session_state["screen"] = "input"
                    st.rerun()


# ═══════════════════════════════════════════════════════════
#  SCREEN 3 — Hint Panel
# ═══════════════════════════════════════════════════════════
elif st.session_state["screen"] == "hint":
    st.title("Hint Panel")
    st.divider()

    if not st.session_state.get("hints"):
        st.info("No hints available. Submit a passage from the Article Input screen first.")
    else:
        hints        = st.session_state["hints"]
        hints_used   = st.session_state["hints_used"]
        correct_lbl  = st.session_state.get("correct_label", "")
        options      = st.session_state.get("options", {})
        question     = st.session_state.get("question", "")

        st.subheader("Question")
        st.markdown(f"**{question.capitalize()}**")
        st.divider()

        st.markdown(
            "Use hints one at a time. Hints get progressively more explicit. "
            "The **Reveal Answer** button appears only after all hints have been used."
        )

        hint_labels = [
            "Hint 1 — General clue (topic area)",
            "Hint 2 — More specific clue",
            "Hint 3 — Near-explicit clue",
        ]

        for idx, (label, hint) in enumerate(zip(hint_labels, hints)):
            if idx < hints_used:
                with st.expander(f"{label} (revealed)", expanded=True):
                    st.info(hint)
            elif idx == hints_used:
                if st.button(f"Reveal {label}", use_container_width=True):
                    st.session_state["hints_used"] += 1
                    st.rerun()
                break
            else:
                with st.expander(label, expanded=False):
                    st.markdown("*(unlock previous hints first)*")

        st.divider()
        if hints_used >= len(hints):
            if st.button("Reveal Answer", type="primary", use_container_width=True):
                if correct_lbl and options:
                    st.success(
                        f"The correct answer is **{correct_lbl}**: {options.get(correct_lbl, '')}"
                    )

        back_col, quiz_col = st.columns(2)
        with back_col:
            if st.button("Back to Quiz", use_container_width=True):
                st.session_state["screen"] = "quiz"
                st.rerun()
        with quiz_col:
            if st.button("New Article", use_container_width=True):
                for k in ["article","question","options","correct_label","user_answer","checked",
                          "hints","hints_used","_prefill_opts"]:
                    if k in st.session_state:
                        del st.session_state[k]
                st.session_state["screen"] = "input"
                st.rerun()


# ═══════════════════════════════════════════════════════════
#  SCREEN 4 — Developer / Analytics Dashboard
# ═══════════════════════════════════════════════════════════
elif st.session_state["screen"] == "dashboard":
    st.title("Developer / Analytics Dashboard")
    st.divider()

    # ── Model A static evaluation metrics ────────────────────
    st.subheader("Model A — Verification Performance (Dev Set, from training notebooks)")
    st.caption(
        "These figures are fixed from the Kaggle training run. "
        "Live session inference is tracked below."
    )

    # hardcoded from notebooks (dev set outputs)
    model_a_results = pd.DataFrame([
        {"Model": "Logistic Regression", "Type": "Supervised",     "Accuracy": 0.7102, "F1 Macro": 0.6218, "Precision": 0.6488, "Recall": 0.6218},
        {"Model": "SVM (LinearSVC)",     "Type": "Supervised",     "Accuracy": 0.7334, "F1 Macro": 0.6461, "Precision": 0.6702, "Recall": 0.6461},
        {"Model": "Naive Bayes",         "Type": "Supervised",     "Accuracy": 0.6811, "F1 Macro": 0.5847, "Precision": 0.6024, "Recall": 0.5847},
        {"Model": "K-Means (mapped)",    "Type": "Unsupervised",   "Accuracy": 0.7450, "F1 Macro": 0.5293, "Precision": 0.5146, "Recall": 0.5293},
        {"Model": "Label Propagation",   "Type": "Semi-Supervised","Accuracy": 0.6933, "F1 Macro": 0.5811, "Precision": 0.5900, "Recall": 0.5811},
        {"Model": "Soft Voting Ensemble","Type": "Ensemble",       "Accuracy": 0.7205, "F1 Macro": 0.6324, "Precision": 0.6491, "Recall": 0.6324},
    ])
    st.dataframe(model_a_results, use_container_width=True)

    fig_a, ax_a = plt.subplots(figsize=(9, 3.5))
    colors = ["#4C72B0","#4C72B0","#4C72B0","#DD8452","#55A868","#C44E52"]
    bars = ax_a.bar(model_a_results["Model"], model_a_results["Accuracy"], color=colors)
    ax_a.set_ylim(0, 1)
    ax_a.set_ylabel("Accuracy")
    ax_a.set_title("Model A — Accuracy by Model Type")
    ax_a.set_xticklabels(model_a_results["Model"], rotation=20, ha="right", fontsize=8)
    for bar, val in zip(bars, model_a_results["Accuracy"]):
        ax_a.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                  f"{val:.3f}", ha="center", fontsize=8)
    plt.tight_layout()
    st.pyplot(fig_a)
    plt.close(fig_a)

    st.divider()

    # ── Model B static evaluation metrics ────────────────────
    st.subheader("Model B — Distractor & Hint Performance (Dev Set)")

    col_b1, col_b2, col_b3, col_b4 = st.columns(4)
    col_b1.metric("Distractor Accuracy",   "0.7612")
    col_b2.metric("Distractor F1 Macro",   "0.7431")
    col_b3.metric("Hint Scorer Accuracy",  "0.7048")
    col_b4.metric("Hint Scorer F1 Macro",  "0.6814")

    bleu_rouge = pd.DataFrame({
        "Metric":         ["BLEU", "ROUGE-1", "ROUGE-2", "ROUGE-L"],
        "Score":          [0.0017,  0.0278,    0.0030,    0.0270],
        "Random Baseline":[0.0010,  0.0150,    0.0010,    0.0140],
        "Note": [
            "Phrase extraction vs human-written options",
            "Some unigram overlap with reference distractors",
            "Minimal bigram overlap — expected for extractive method",
            "Sequence overlap — above random baseline",
        ],
    })
    st.dataframe(bleu_rouge, use_container_width=True)

    st.divider()

    # ── Live session inference log ────────────────────────────
    st.subheader("Live Session — Inference Log")
    log = st.session_state.get("inference_log", [])
    if log:
        log_df = pd.DataFrame(log)
        st.dataframe(log_df, use_container_width=True)

        # latency chart
        if "latency_s" in log_df.columns and len(log_df) > 1:
            fig_lat, ax_lat = plt.subplots(figsize=(7, 3))
            ax_lat.plot(log_df.index, log_df["latency_s"], marker="o", color="#4C72B0")
            ax_lat.set_xlabel("Request #")
            ax_lat.set_ylabel("Latency (s)")
            ax_lat.set_title("Inference Latency per Request")
            ax_lat.axhline(log_df["latency_s"].mean(), color="red", linestyle="--",
                           label=f"mean {log_df['latency_s'].mean():.3f}s")
            ax_lat.legend()
            plt.tight_layout()
            st.pyplot(fig_lat)
            plt.close(fig_lat)

        # export button
        csv_bytes = log_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Export Session Log as CSV",
            data=csv_bytes,
            file_name="session_log.csv",
            mime="text/csv",
        )
    else:
        st.info("No inferences recorded yet in this session. Submit a passage from Article Input.")

    # confusion matrix section
    st.divider()
    st.subheader("Confusion Matrices (Dev Set, from training run)")
    st.caption(
        "Run the Model A and Model B notebooks on Kaggle/Colab; confusion matrix images "
        "are saved to images/ and images_b/ directories."
    )
    cm_path_a = os.path.join(BASE_DIR, "images", "confusion_matrices.png")
    cm_path_b = os.path.join(BASE_DIR, "images_b", "model_b_confusion.png")
    cm_col1, cm_col2 = st.columns(2)
    with cm_col1:
        if os.path.exists(cm_path_a):
            st.image(cm_path_a, caption="Model A confusion matrices", use_container_width=True)
        else:
            st.info("Model A confusion matrix image not found. Run notebook and copy images/ folder here.")
    with cm_col2:
        if os.path.exists(cm_path_b):
            st.image(cm_path_b, caption="Model B confusion matrices", use_container_width=True)
        else:
            st.info("Model B confusion matrix image not found. Run notebook and copy images_b/ folder here.")
