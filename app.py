import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import KMeans

st.set_page_config(page_title="Loan Default Predictor", page_icon="🏦", layout="wide")
st.markdown("""<style>
.block-container {padding-top: 2rem;}
div[data-testid="stMetric"] {background: rgba(128,128,128,.08); padding: 14px 18px; border-radius: 12px;}
</style>""", unsafe_allow_html=True)

DEFAULT_CSV = "loan_data (1).csv"


# ───────────────────────── Notebook code (unchanged) ─────────────────────────
@st.cache_data(show_spinner="Running preprocessing and training both models…")
def run_pipeline(raw: pd.DataFrame):
    # Step 2: Load the Dataset
    df = raw.copy()
    df.isnull().sum()
    df.dropna(inplace=True)

    # Step 4: Data Preprocessing
    df_copy = df.copy()
    df_copy = df_copy.drop('applicant_id', axis=1)

    cat_col = df_copy.select_dtypes(include='object').columns.drop('loan_status')

    df_copy = df.copy()
    df_copy = pd.get_dummies(df_copy,
                             columns=cat_col,
                             dtype=int,
                             drop_first=True)

    df_copy = df_copy.drop('applicant_id', axis=1)

    Le = LabelEncoder()
    df_copy['loan_status'] = Le.fit_transform(df_copy['loan_status'])

    X = df_copy.drop('loan_status', axis=1)
    y = df_copy['loan_status']

    # Step 5: Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)

    sm = SMOTE(random_state=42)
    X_train, y_train = sm.fit_resample(X_train, y_train)

    # Step 6: Naive Bayes
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)

    # KNN
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit(X_test)

    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X, y)
    y_pred2 = neigh.predict(X_test)

    return dict(df=df, cat_col=list(cat_col), Le=Le, X=X, y=y, X_train=X_train,
                y_train=y_train, X_test=X_test, y_test=y_test, gnb=gnb, neigh=neigh,
                y_pred=y_pred, y_pred2=y_pred2, X_train_scaled=X_train_scaled)


@st.cache_data(show_spinner="Fitting KMeans for K = 1 … 19…")
def run_elbow(X_train_scaled):
    k_values = range(1, 20)
    inertia_values = []
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_train_scaled)
        inertia_values.append(kmeans.inertia_)
    return list(k_values), inertia_values
# ──────────────────────────── End notebook code ────────────────────────────


# ───────────────────────────── UI helpers ─────────────────────────────
def report_table(y_true, y_hat, classes):
    rep = pd.DataFrame(classification_report(y_true, y_hat, output_dict=True)).T
    rep.index = [classes[int(i)] if i.isdigit() else i for i in rep.index]
    return rep.round(3)


def confusion_fig(y_true, y_hat, classes, title):
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    sns.heatmap(confusion_matrix(y_true, y_hat), annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, cbar=False, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title)
    return fig


def build_row(values, cat_col, x_cols):
    """Turn one raw applicant into a row aligned with the one-hot encoded training columns."""
    row = pd.DataFrame(0.0, index=[0], columns=x_cols)
    for col, val in values.items():
        key = f"{col}_{val}"
        if col in cat_col:
            if key in row.columns:      # the dropped baseline category stays all-zero
                row[key] = 1
        else:
            row[col] = val
    return row


# ───────────────────────────── Sidebar / data ─────────────────────────────
st.sidebar.title("🏦 Loan Default Predictor")
st.sidebar.caption("Naïve Bayes vs KNN on applicant data")
upload = st.sidebar.file_uploader("Upload loan CSV", type="csv")

if upload is not None:
    raw = pd.read_csv(upload)
elif Path(DEFAULT_CSV).exists():
    raw = pd.read_csv(DEFAULT_CSV)
    st.sidebar.success(f"Loaded `{DEFAULT_CSV}`")
else:
    st.title("🏦 Loan Default Prediction")
    st.info("Upload your loan dataset in the sidebar to begin. The CSV needs an "
            "`applicant_id` column and a `loan_status` target column.")
    st.stop()

missing = {"applicant_id", "loan_status"} - set(raw.columns)
if missing:
    st.error(f"The notebook's pipeline expects these columns, missing from the CSV: {sorted(missing)}")
    st.stop()

R = run_pipeline(raw)
classes = list(R["Le"].classes_)

st.sidebar.markdown("---")
st.sidebar.markdown("**Pipeline settings (from notebook)**")
st.sidebar.markdown("- Test size: `0.33`, seed `42`\n- Imbalance: `SMOTE` on train set\n"
                    "- Naïve Bayes: `GaussianNB`\n- KNN: `n_neighbors=3`")

st.title("🏦 Loan Default Prediction")
tabs = st.tabs(["📋 Data & EDA", "🛠️ Preprocessing", "🤖 Naïve Bayes", "📍 KNN",
                "📈 Elbow", "🔮 Predict"])

# ───────────────────────────── Tab 1: EDA ─────────────────────────────
with tabs[0]:
    df = R["df"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows (after dropna)", f"{len(df):,}")
    c2.metric("Columns", df.shape[1])
    c3.metric("Rows dropped (nulls)", f"{len(raw) - len(df):,}")
    st.subheader("Preview")
    st.dataframe(df.head(20), use_container_width=True)
    left, right = st.columns(2)
    with left:
        st.subheader("Summary statistics")
        st.dataframe(df.describe(), use_container_width=True)
    with right:
        st.subheader("Loan status distribution")
        fig, ax = plt.subplots(figsize=(5, 3.4))
        sns.countplot(data=df, x='loan_status', ax=ax)
        st.pyplot(fig)
        st.dataframe(df['loan_status'].value_counts().rename("count"))
    st.subheader("Missing values (before dropna)")
    st.dataframe(raw.isnull().sum().rename("nulls").to_frame().T, use_container_width=True)

# ───────────────────────────── Tab 2: Preprocessing ─────────────────────────────
with tabs[1]:
    st.markdown("GaussianNB and KNN need numbers, so text columns are **one-hot encoded** "
                "(`drop_first=True`), the target is **label encoded**, then the training set is "
                "balanced with **SMOTE**.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Categorical columns", len(R["cat_col"]))
    c2.metric("Features after encoding", R["X"].shape[1])
    c3.metric("Train rows after SMOTE", f"{len(R['y_train']):,}")
    st.write("**Categorical columns encoded:**", ", ".join(R["cat_col"]) or "none")
    st.write("**Target encoding:**", {c: i for i, c in enumerate(classes)})
    left, right = st.columns(2)
    with left:
        st.caption("Encoded feature matrix X")
        st.dataframe(R["X"].head(10), use_container_width=True)
    with right:
        st.caption("Class balance in training set, after SMOTE")
        st.bar_chart(pd.Series(R["y_train"]).map(lambda i: classes[i]).value_counts())

# ───────────────────────────── Tab 3 & 4: Models ─────────────────────────────
def model_tab(tab, name, y_hat):
    with tab:
        yt = R["y_test"]
        acc = (yt.values == y_hat).mean()
        st.metric(f"{name} accuracy on test set", f"{acc:.2%}")
        left, right = st.columns([3, 2])
        with left:
            st.subheader("Classification report")
            st.dataframe(report_table(yt, y_hat, classes), use_container_width=True)
        with right:
            st.pyplot(confusion_fig(yt, y_hat, classes, f"{name} confusion matrix"))
        st.subheader("Actual vs predicted (first 10 test rows)")
        st.dataframe(pd.DataFrame({"Actual": yt.values[:10], "Predicted": y_hat[:10]}))

model_tab(tabs[2], "Naïve Bayes", R["y_pred"])
model_tab(tabs[3], "KNN", R["y_pred2"])
with tabs[3]:
    st.caption("Heads-up: as written in the notebook, the KNN is fit on the full unscaled `X, y` "
               "and then scored on `X_test`, so those rows were already seen during training.")

# ───────────────────────────── Tab 5: Elbow ─────────────────────────────
with tabs[4]:
    st.markdown("Inertia (WCSS) from **KMeans** for K = 1 to 19 on `X_train_scaled`.")
    if st.button("Run elbow analysis"):
        ks, inertia = run_elbow(R["X_train_scaled"])
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(ks, inertia, marker='o', linestyle='-')
        ax.set_title('Elbow Method for Optimal K')
        ax.set_xlabel('Number of Clusters (K)')
        ax.set_ylabel('Inertia (WCSS)')
        ax.set_xticks(ks)
        ax.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig)

# ───────────────────────────── Tab 6: Predict ─────────────────────────────
with tabs[5]:
    st.markdown("Enter a new applicant. The values are one-hot encoded the same way as the "
                "training data, then sent to both models.")
    num_cols = [c for c in raw.columns
                if c not in R["cat_col"] + ["applicant_id", "loan_status"]]
    values = {}
    with st.form("applicant"):
        cols = st.columns(3)
        for i, c in enumerate(num_cols):
            s = R["df"][c]
            values[c] = cols[i % 3].number_input(c, value=float(s.median()),
                                                 min_value=float(s.min()), max_value=float(s.max()))
        for j, c in enumerate(R["cat_col"]):
            values[c] = cols[(len(num_cols) + j) % 3].selectbox(c, sorted(R["df"][c].unique()))
        go = st.form_submit_button("Predict", type="primary")
    if go:
        row = build_row(values, R["cat_col"], R["X"].columns)
        c1, c2 = st.columns(2)
        for col, name, model in [(c1, "Naïve Bayes", R["gnb"]), (c2, "KNN", R["neigh"])]:
            pred = model.predict(row)[0]
            proba = model.predict_proba(row)[0]
            col.metric(name, str(classes[pred]))
            col.bar_chart(pd.Series(proba, index=classes))
