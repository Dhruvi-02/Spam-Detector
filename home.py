import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. LOAD DATA
try:
    df = pd.read_csv('mail_data.csv')
    print("CSV loaded successfully!")
except FileNotFoundError:
    print("Error: 'mail_data.csv' not found. Please create the file first.")
    exit()

# 2. PRE-PROCESSING
# Handle potential case-sensitivity issues by forcing column names to lowercase
df.columns = [col.lower() for col in df.columns]

# Map labels: spam -> 1, ham -> 0
# Using .strip() to remove any accidental spaces in the CSV
df['category'] = df['category'].str.strip().str.lower().map({'spam': 1, 'ham': 0})

X = df['message']
y = df['category']

# 3. SPLIT DATA (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. FEATURE EXTRACTION (TF-IDF)
# Converts text into numerical scores

tfidf = TfidfVectorizer(stop_words='english', lowercase=True)
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

# 5. TRAIN LOGISTIC REGRESSION
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# 6. EVALUATE
y_pred = model.predict(X_test_vec)
print(f"Initial Training Accuracy: {accuracy_score(y_test, y_pred):.2%}")

# 7. THE "LEARNING" DEMONSTRATION
def analyze_message(text):
    vec_text = tfidf.transform([text])
    
    # Probability of being spam 
    probability = model.predict_proba(vec_text)[0][1] * 100
    
    # Final classification
    prediction = model.predict(vec_text)[0]
    result = "SPAM" if prediction == 1 else "NOT SPAM"
    
    return result, probability

# List of messages to prove the model has learned patterns
test_samples = [
    "Hey, are we still meeting for coffee at 5?",
    "WINNER! Claim your free gift card now by clicking here!",
    "Can you send the report by EOD?",
    "URGENT: Your account was hacked. Send money now.",
    "I'll be a bit late for the meeting, sorry."
]

print("\n" + "="*40)
print("   SPAM DETECTOR LEARNING RESULTS")
print("="*40)

for msg in test_samples:
    label, score = analyze_message(msg)
    print(f"MESSAGE: {msg}")
    print(f"ANALYSIS: {label}")
    print(f"SPAM CONFIDENCE: {score:.2f}%")
    print("-" * 40)

# 8. INTERACTIVE INPUT
print("\nTry it yourself!")
user_input = input("Enter an email message to test: ")
final_label, final_score = analyze_message(user_input)
print(f"\nResult: {final_label} ({final_score:.2f}% spam score)")