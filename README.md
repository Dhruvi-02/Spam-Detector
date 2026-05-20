# 📬 Email Spam Detector

A beginner-friendly Python project that uses Machine Learning and Natural Language Processing (NLP) to automatically classify emails as **Spam** or **Not Spam (Ham)**.

Unlike basic keyword filters, this tool analyzes the context of the entire message to make smart predictions. 

---

## ✨ Features

* **Smart Classification:** Uses TF-IDF vectorization to understand phrase importance rather than just flagging random words.
* **Context-Aware:** Knows the difference between a spam lottery (*"Congratulations! You won a million dollars!"*) and a real email (*"Congratulations! You're hired for the internship!"*).
* **Interactive Test:** Includes a built-in interactive mode where you can type your own messages and see a real-time spam probability score.

---

## 🛠️ Tech Stack

* **Python 3**
* **Pandas** (for loading and handling the CSV data)
* **Scikit-Learn** (for training the Machine Learning model)

---

## 🚀 How to Run It

### 1. Install Dependencies
Open your terminal and run this command to install the required libraries:
```bash
pip install pandas scikit-learn
```

### 2. Prepare the Files
Make sure your folder has these two files saved together:
- Home.py (The main Python script)
- mail_data.csv (The dataset containing spam and ham examples)

### 3. Run the Program
Start the detector by running below line in terminal:
```bash
python home.py
```

## 📊 How it Works

- ​Loads & Cleans Data: Reads the CSV file and converts all text to lowercase so case-sensitivity doesn't mess up predictions.
- Trains the Model: Splits the data into an 80% training set and a 20% testing set, using a Logistic Regression model.
- Tests Live Text: Runs a few built-in sample tests to demonstrate its accuracy.
- User Input: Prompts you to type a message and outputs a definitive answer alongside a "Spam Confidence Score" percentage.
