import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

# ===============================
# 1. Charger les données
# ===============================
@st.cache_data
def load_data():
    # Remplace par ton chemin CSV ou BigQuery export
    df = pd.read_csv("bquxjob_6467eac5_1985b508c46.csv")
    return df

df = load_data()

st.title("📊 Snowball – Exploratory dashboard")

st.markdown("""
**⚠️ Disclaimer :** The data are heavily imbalanced (only ~55 paying subscribers).  
The models shown here are **experimental and unreliable** – they are for exploratory purposes only.
""")

# ===============================
# 2. Vue d’ensemble
# ===============================
st.header("1️⃣ User overview")

col1, col2 = st.columns(2)
with col1:
    total_users = len(df)
    paid_users = df["is_paid_subscriber"].sum()
    st.metric("Total users", total_users)
    st.metric("Paying users", paid_users)

with col2:
    fig, ax = plt.subplots()
    df["is_paid_subscriber"].value_counts().plot(kind="pie", labels=["Free", "Paid"], autopct="%1.1f%%", ax=ax)
    ax.set_ylabel("")
    ax.set_title("Paid / Free split")
    st.pyplot(fig)

# ===============================
# 3. Exploration interactive
# ===============================
st.header("2️⃣ Data exploration")

numeric_cols = df.select_dtypes(include=['float64','int64']).columns.tolist()
feature = st.selectbox("Choose a variable to explore :", numeric_cols)

# Create KDE plots for paid vs free with explicit labels
fig, ax = plt.subplots()
labels_map = {False: "Free", True: "Paid"}
for status, subset in df.groupby("is_paid_subscriber"):
    subset[feature].plot(kind="kde", ax=ax, label=labels_map.get(status, status))
ax.set_title(f"Distribution of {feature} by status")
ax.legend(title="Subscription status")

# If the variable can't be negative (e.g. rates or days), keep the x‑axis ≥ 0
if df[feature].min() >= 0:
    ax.set_xlim(left=0)

st.pyplot(fig)

# ===============================
# 4. Prototype modèle prédictif
# ===============================
st.header("3️⃣ Predictive model prototype")

# Features de base (à ajuster selon ton dataset)
features = ["open_rate", "click_rate", "time_since_signup"]
df_model = df.dropna(subset=features)  # on retire les NA pour l'exemple

X = df_model[features]
y = df_model["is_paid_subscriber"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Normalisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modèle
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train_scaled, y_train)

# Évaluation
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)

st.write(f"**Accuracy :** {acc:.2f} | **Precision :** {prec:.2f} | **Recall :** {rec:.2f}")
st.caption("⚠️ Warning: These scores are low and should be interpreted with caution.")

# ===============================
# 5. Simulation “What if”
# ===============================
st.subheader("🔮 Simulate a user")

open_rate = st.slider("Open rate (%)", 0, 100, 30) / 100
click_rate = st.slider("Click rate (%)", 0, 100, 5) / 100
time_since_signup = st.slider("Days since signup", 0, 365, 60)

# Créer un DataFrame pour prédire
input_df = pd.DataFrame([[open_rate, click_rate, time_since_signup]], columns=features)
input_scaled = scaler.transform(input_df)

proba = model.predict_proba(input_scaled)[0][1]
st.write(f"👉 **Estimated probability of being a paying subscriber: {proba*100:.1f}%**")
st.caption("⚠️ This result is purely indicative.")