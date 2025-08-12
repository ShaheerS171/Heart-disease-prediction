# ❤️ Heart Disease Prediction App

This is a **Streamlit-based web application** that predicts the likelihood of heart disease using a pre-trained **K-Nearest Neighbors (KNN)** machine learning model.  
The app provides an intuitive interface for users to enter their health parameters and visualize prediction results along with probabilities.

---

## 📌 Features

- **User-friendly UI** built with Streamlit
- Sidebar-based input form for patient health data
- Instant prediction of heart disease likelihood
- **Probability chart** showing confidence levels
- Model loaded from a `.pkl` file using Joblib
- Clean and responsive layout

---

## 📂 Project Structure

```
project/
│
├── knn_model.pkl          # Trained KNN model
├── app.py                  # Streamlit app source code
├── README.md               # Project documentation
└── requirements.txt        # Python dependencies
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/yourusername/heart-disease-prediction.git
cd heart-disease-prediction
```

### 2️⃣ Create and activate a virtual environment (optional but recommended)
```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Add your trained model
Place your trained model file `knn_model.pkl` in the project directory.  
Ensure the model is trained with the same feature names and preprocessing pipeline expected by the app.

---

## ▶️ Running the App
```bash
streamlit run app.py
```

This will launch the app in your default web browser.

---

## 🧠 Model Information
- **Algorithm:** K-Nearest Neighbors (KNN)
- **Training Data:** Heart disease dataset (e.g., UCI Heart Disease dataset)
- **Input Features:** Age, sex, chest pain type, blood pressure, cholesterol, fasting blood sugar, ECG results, heart rate, exercise-induced angina, ST depression, slope, number of major vessels, thalassemia.

---

## 📊 Example Prediction Output

- **Prediction Result:**  
  ✅ You are not likely to have heart disease.  
  ⚠️ You are likely to have heart disease.

- **Probability Chart:**  
  Displays probabilities for both outcomes in a bar chart format.

---

## 📦 Requirements

Example `requirements.txt`:
```
streamlit
pandas
joblib
altair
```

---

## 💡 Future Improvements
- Add more ML models for comparison
- Allow batch prediction from CSV
- Deploy online via Streamlit Cloud or Hugging Face Spaces
