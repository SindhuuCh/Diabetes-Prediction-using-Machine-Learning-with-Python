# Diabetes Prediction Using Machine Learning

This project aims to predict whether a person has diabetes based on diagnostic measurements. The dataset used is the **Pima Indians Diabetes Database**, which includes medical records for women and various health indicators.

## 📁 Project Structure

- `Diabetes Prediction.ipynb` – Main Jupyter Notebook containing code for data loading, exploration, preprocessing, modeling, and evaluation.

## 📊 Dataset

The dataset contains the following features:

- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age
- Outcome (0: No diabetes, 1: Diabetes)

## ⚙️ Steps Performed

1. **Data Loading** – Load CSV data into a pandas DataFrame.
2. **Exploratory Data Analysis (EDA)** – Visualize missing values, feature distributions, and correlations.
3. **Data Cleaning & Preprocessing** – Handle zeros as missing values, impute missing data, normalize features.
4. **Model Building** – Train and evaluate machine learning models such as:
   - Logistic Regression
   - K-Nearest Neighbors
   - Random Forest
5. **Model Evaluation** – Use metrics like accuracy, confusion matrix, and classification report.

## 📈 Results

The best-performing model was chosen based on accuracy and F1-score, ensuring reliable diabetes predictions on unseen data.

## 🧰 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib & Seaborn
- Scikit-learn
- Jupyter Notebook

## 🚀 How to Run

1. Clone the repository or download the notebook.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open the notebook:
   ```bash
   jupyter notebook Diabetes\ Prediction.ipynb
   ```
4. Run the cells to see data exploration, model training, and evaluation.

## 📝 Future Improvements

- Hyperparameter tuning
- Try additional models like XGBoost or SVM
- Deploy the model using Flask or Streamlit
- Improve missing value handling techniques

## 📬 Contact
- Sindhu
- Master's student in Information Technology
- Passionate about machine learning and data analytics



