import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Predictor",
    page_icon="🤖",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
/* Background */
.stApp {
    background: linear-gradient(to right, #1e3c72, #2a5298);
    color: white;
}

/* Title */
.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    animation: fadeIn 1.5s ease-in;
}

/* Card */
.card {
    background: rgba(255, 255, 255, 0.1);
    padding: 25px;
    border-radius: 15px;
    backdrop-filter: blur(8px);
    box-shadow: 0px 4px 20px rgba(0,0,0,0.3);
    animation: slideUp 0.8s ease-in-out;
}

/* Button */
.stButton>button {
    background: linear-gradient(to right, #ff512f, #dd2476);
    color: white;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    border: none;
}
.stButton>button:hover {
    transform: scale(1.05);
    transition: 0.3s;
}

/* Result Box */
.result {
    text-align: center;
    font-size: 26px;
    font-weight: bold;
    padding: 15px;
    border-radius: 10px;
    background: rgba(0,0,0,0.4);
    animation: fadeIn 1s ease-in;
}

/* Animations */
@keyframes fadeIn {
    from {opacity: 0;}
    to {opacity: 1;}
}
@keyframes slideUp {
    from {opacity: 0; transform: translateY(30px);}
    to {opacity: 1; transform: translateY(0);}
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("<div class='title'>🤖 AI Prediction System</div>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Enter your data or upload CSV for predictions</p>", unsafe_allow_html=True)

# ---------------- MODE SELECTION ----------------
mode = st.radio("Select Input Mode", ["Manual Input", "CSV Upload"])

# ---------------- MANUAL INPUT ----------------
if mode == "Manual Input":
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.subheader("🔢 Enter Features")

    # 👉 MODIFY BASED ON YOUR MODEL INPUTS
    col1, col2 = st.columns(2)
    with col1:
        f1 = st.number_input("Feature 1")
        f2 = st.number_input("Feature 2")
    with col2:
        f3 = st.number_input("Feature 3")
        f4 = st.number_input("Feature 4")

    input_data = np.array([[f1, f2, f3, f4]])

    if st.button("🚀 Predict"):
        try:
            prediction = model.predict(input_data)

            st.markdown(
                f"<div class='result'>✅ Prediction: {prediction[0]}</div>",
                unsafe_allow_html=True
            )

        except Exception as e:
            st.error(f"Error: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- CSV UPLOAD ----------------
else:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    file = st.file_uploader("📂 Upload CSV", type=["csv"])

    if file is not None:
        df = pd.read_csv(file)
        st.write("Preview:", df.head())

        if st.button("📊 Predict CSV"):
            try:
                preds = model.predict(df)
                df["Prediction"] = preds

                st.success("✅ Prediction Completed")
                st.dataframe(df)

                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("⬇ Download Results", csv, "output.csv", "text/csv")

            except Exception as e:
                st.error(f"Error: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("<p style='text-align:center;'>Made with ❤️ using Streamlit</p>", unsafe_allow_html=True)
