import streamlit as st
import joblib
import re  # 👈 For text validation

# -----------------------------
# 1️⃣ Load saved models & TF-IDF vectorizers
# -----------------------------
model_a = joblib.load("models/Model_A.pkl")      # Balanced model
tfidf_a = joblib.load("models/tfidf_bal.pkl")    # Vectorizer for Model_A

model_b = joblib.load("models/Model_B.pkl")      # Imbalanced model
tfidf_b = joblib.load("models/tfidf_imb.pkl")    # Vectorizer for Model_B

# -----------------------------
# 2️⃣ Streamlit UI
# -----------------------------
st.set_page_config(page_title="Review Rating Predictor", page_icon="⭐", layout="centered")

st.title("📊 Automated Review Rating System")


# Input box
user_review = st.text_area("📝 Enter your review here:", height=150)

# -----------------------------
# 3️⃣ Submit Button
# -----------------------------
if st.button("Submit"):
    if user_review.strip() == "":
        st.warning("⚠️ Please enter a review before submitting.")
    elif not re.search(r"[a-zA-Z]", user_review):  # 👈 checks if text has no letters
        st.error("🚫 Please enter a valid review (text only, not just numbers or symbols).")
    else:
        # -----------------------------
        # 4️⃣ Predict with both models
        # -----------------------------
        X_vec_a = tfidf_a.transform([user_review])
        pred_a = model_a.predict(X_vec_a)[0]

        X_vec_b = tfidf_b.transform([user_review])
        pred_b = model_b.predict(X_vec_b)[0]

        # -----------------------------
        # 5️⃣ Display results
        # -----------------------------
        st.success("✅ Prediction Completed!")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🎯 Model_A (Balanced)")
            st.write(f"**Predicted Rating:** ⭐ {pred_a}")

        with col2:
            st.subheader("⚖️ Model_B (Imbalanced)")
            st.write(f"**Predicted Rating:** ⭐ {pred_b}")

        # Summary table
        st.markdown("---")
        st.markdown("### 📈 Model Comparison")
        st.table({
            "Model": ["Model_A (Balanced)", "Model_B (Imbalanced)"],
            "Predicted Rating": [pred_a, pred_b]
        })
