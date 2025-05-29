# Used Car Price Estimator

This project is a **Streamlit** web application that enables users to explore and predict the price of used cars based on various features such as brand, model, body type, fuel type, mileage, and more.

The project leverages data science tools such as **pandas**, **matplotlib**, **seaborn**, and machine learning models like **XGBoost** to provide accurate price estimations.

---

## Project Objective

To create a simple, interactive web tool that predict car prices using a pre-trained regression model.

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone <repository_url>
cd <project_directory>
```

### 2. Create a virtual environment

Depending on your system configuration, run:

```bash
python3 -m venv .venv       
# or
python -m venv .venv        
```

### 3. Activate the virtual environment


```bash
.venv\Scripts\activate  # For Windows

source .venv/bin/activate # For macOS/Linux
```

### 4. Install the dependencies

All required libraries are listed in `requirements.txt`. Install them with:

```bash
pip install -r requirements.txt
```

---

## Running the Application

After installation, start the Streamlit app by running:

```bash
streamlit run app.py
```

This will open the application in your default web browser.

---

## Key Files

- `app.py`: Main Streamlit application script
- `xgboost_model.pkl`: Pre-trained machine learning model used for predictions
- `Pretraitement_EDA_Prix_Voitures.ipynb`: Jupyter Notebook containing data preprocessing and exploratory data analysis'
- `training_and_compare_models.ipynb`: Jupyter Notebook for training and comparing different machine learning models

---

## Contributors

- El Mamoune **BENMASSAOUD**
- Rida **BELKHIRI**
- Imad **EL HARROUTI**
