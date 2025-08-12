import streamlit as st
import pandas as pd
import joblib
import os
import altair as alt

st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

# --- Title & Description ---
st.markdown(
    """
    <h1 style='color:#d7263d;'>❤️ Heart Disease Prediction App</h1>
    <p style='font-size:18px;'>Predict the likelihood of heart disease based on your health data.</p>
    """,
    unsafe_allow_html=True
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'knn_model.pkl')

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at {MODEL_PATH}")
    st.stop()

model = joblib.load(MODEL_PATH)

# --- Sidebar Input ---
st.sidebar.title("Patient Data Input")
st.sidebar.markdown("Fill in the patient's health information:")

def user_input_features():
    col1, col2, col3 = st.sidebar.columns(3)

    with col1:
        age = st.slider('Age (years)', 29, 77, 54)
        sex = st.selectbox('Sex', ['Male', 'Female'])
        cp = st.selectbox('Chest Pain Type', ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'])
        trestbps = st.slider('Resting Blood Pressure (mm Hg)', 94, 180, 131)
        chol = st.slider('Serum Cholesterol (mg/dl)', 126, 500, 246)

    with col2:
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['True', 'False'])
        restecg = st.selectbox('Resting ECG', ['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'])
        thalach = st.slider('Max Heart Rate Achieved', 71, 202, 149)
        exang = st.selectbox('Exercise Induced Angina', ['Yes', 'No'])
        oldpeak = st.slider('ST Depression by Exercise', 0.0, 6.2, 1.0, 0.1)

    with col3:
        slope = st.selectbox('Slope of ST Segment', ['Upsloping', 'Flat', 'Downsloping'])
        ca = st.selectbox('Number of Major Vessels (0-3)', ['0', '1', '2', '3'])  # Keep as string if trained like this
        thal = st.selectbox('Thalassemia', ['Normal', 'Fixed Defect', 'Reversible Defect'])

    # Return raw values — no encoding here!
    data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalch': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }

    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

st.markdown("---")
st.subheader('User Input Parameters')
st.dataframe(input_df, use_container_width=True)

if st.button('Predict'):
    with st.spinner('Predicting...'):
        try:
            prediction = model.predict(input_df)
            prediction_proba = model.predict_proba(input_df)
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.stop()

    with st.container():
        st.subheader('Prediction Result')
        if prediction[0] == 1:
            st.error('⚠️ You are likely to have heart disease.')
        else:
            st.success('✅ You are not likely to have heart disease.')

        st.subheader('Prediction Probability')
        st.write(f'Probability of No Heart Disease: **{prediction_proba[0][0]:.2f}**')
        st.write(f'Probability of Heart Disease: **{prediction_proba[0][1]:.2f}**')

        # Probability Bar Chart
        proba_df = pd.DataFrame({
            'Outcome': ['No Heart Disease', 'Heart Disease'],
            'Probability': prediction_proba[0]
        })
        chart = alt.Chart(proba_df).mark_bar(color='#d7263d').encode(
            x=alt.X('Outcome', sort=None),
            y=alt.Y('Probability', scale=alt.Scale(domain=[0, 1])),
            tooltip=['Outcome', alt.Tooltip('Probability', format='.2f')]
        ).properties(
            width=400,
            height=300,
            title='Prediction Probabilities'
        )
        st.altair_chart(chart, use_container_width=True)
