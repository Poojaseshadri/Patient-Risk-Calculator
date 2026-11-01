# Patient Risk Calculator

A machine learning-based web application that estimates patient risk scores using healthcare data. The application uses an XGBoost regression model and provides interactive dashboards, preventive measures, and an AI-powered chatbot.

## Project Overview

The Patient Risk Calculator is designed to analyze patient-related information and predict risk scores using machine learning techniques. It combines data preprocessing, NLP techniques, model optimization, and an interactive Streamlit interface.

## Features

- **Risk Score Prediction:** Predicts patient risk scores using a trained XGBoost regression model.
- **Interactive Dashboard:** Displays patient data and risk-related visualizations.
- **Preventive Measures:** Provides preventive suggestions based on patient risk information.
- **AI Chatbot:** Uses Google Gemini API to answer user questions.
- **Data Preprocessing:** Handles text preprocessing, feature scaling, and categorical encoding.
- **Hyperparameter Optimization:** Uses Optuna to optimize XGBoost model parameters.

## Technologies Used

- **Programming Language:** Python
- **Machine Learning:** XGBoost, Scikit-learn
- **Hyperparameter Optimization:** Optuna
- **NLP:** NLTK, TF-IDF, SymSpell
- **Data Processing:** Pandas
- **Data Visualization:** Plotly
- **Web Framework:** Streamlit
- **AI Integration:** Google Gemini API
- **Imbalanced Data Handling:** Imbalanced-learn

## Model Performance

The trained XGBoost regression model achieved the following results on the test dataset:

| Evaluation Metric | Result |
|---|---:|
| R² Score | 0.7486 (74.86%) |
| Mean Absolute Error (MAE) | 0.3024 |
| Mean Squared Error (MSE) | 0.1424 |
| Root Mean Squared Error (RMSE) | 0.3773 |

**Note:** The R² score indicates how much variation in the target values is explained by the model. It should not be interpreted as prediction accuracy.

## Project Structure

```text
Patient-Risk-Calculator/
│
├── app.py
├── generative_ai.py
├── get_patient_data.py
├── text_preprocessing.py
├── requirements.txt
│
├── datasets/
│   └── targetfinaldataset1 (1).csv
│
├── models/
│   └── xgb_pipeline.pkl
│
├── xgboost_with_optuna.ipynb
├── .gitignore
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Poojaseshadri/Patient-Risk-Calculator.git
```

### 2. Navigate to the project directory

```bash
cd Patient-Risk-Calculator
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

**Windows:**

```bash
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Configure the Gemini API key

Create a `.env` file in the project root directory:

```env
GENAI_API_KEY=your_gemini_api_key
```

### 7. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## Model Development

The machine learning workflow includes:

1. Loading and exploring the healthcare dataset.
2. Cleaning and preprocessing structured and textual data.
3. Applying NLP preprocessing and TF-IDF vectorization.
4. Scaling numerical features and encoding categorical features.
5. Training an XGBoost regression model.
6. Optimizing model hyperparameters using Optuna.
7. Evaluating model performance using MAE, MSE, RMSE, and R².
8. Saving the trained pipeline for use in the Streamlit application.

## Future Enhancements

- Improve model performance through further experimentation and validation.
- Add more interactive patient analytics.
- Enhance preventive recommendation features.
- Improve chatbot responses and integration.
- Deploy the application to a cloud platform.

## Disclaimer

This project is an educational machine learning prototype. Its predictions and preventive suggestions are not medically validated and must not be used as a substitute for professional medical advice, diagnosis, or treatment.

## Author

**Pooja Seshadri**

GitHub: https://github.com/Poojaseshadri

Project Repository: https://github.com/Poojaseshadri/Patient-Risk-Calculator
