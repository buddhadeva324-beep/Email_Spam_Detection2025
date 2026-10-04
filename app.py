import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(
    page_title="Email Spam Detection",
    page_icon="📧",
    layout="centered"
)

st.title("📧 Email Spam Detection")
st.write("Enter an email message to check whether it is Spam or Ham.")

# Dataset
data = {
    "label": [
        "spam","ham","spam","ham","spam","ham","spam","ham","spam","ham",
        "spam","ham","spam","ham","spam","ham","spam","ham","spam","ham",
        "spam","ham","spam","ham","spam","ham","spam","ham","spam","ham"
    ],
    "message": [
        "Congratulations you won a free prize",
        "Hey, how are you doing today?",
        "Claim your free money now",
        "Please call me when you are free",
        "You have won a lottery prize",
        "Can we meet tomorrow?",
        "Free gift waiting for you",
        "Don't forget our meeting",
        "Win cash now click the link",
        "Happy birthday have a great day",
        "You are selected for a free reward",
        "Please send me the assignment",
        "Get free recharge immediately",
        "Are you coming to college today?",
        "Congratulations you won 10000 dollars",
        "Let's have lunch together",
        "Free offer available now",
        "Please check the project file",
        "You won a free vacation",
        "I will call you later",
        "Claim your cash reward",
        "Good morning have a nice day",
        "Limited time free offer",
        "Please attend the class tomorrow",
        "You have won free tickets",
        "Can you send me the notes?",
        "Click now to get your free prize",
        "See you tomorrow",
        "Urgent claim your free reward",
        "Thanks for your message"
    ]
}

df = pd.DataFrame(data)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df["message"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# Convert text to numbers
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Accuracy
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)

st.success(f"Model Accuracy: {accuracy * 100:.2f}%")

# Email input
email = st.text_area(
    "Enter your email message:",
    placeholder="Type your email here..."
)

if st.button("🔍 Check Email"):
    if email.strip():
        email_tfidf = vectorizer.transform([email])
        prediction = model.predict(email_tfidf)[0]

        if prediction == "spam":
            st.error("🚨 This email is SPAM")
        else:
            st.success("✅ This email is HAM (Not Spam)")
    else:
        st.warning("Please enter an email message.")

# Dataset preview
with st.expander("📊 View Dataset"):
    st.dataframe(df)

# Confusion Matrix
with st.expander("📈 Confusion Matrix"):
    cm = confusion_matrix(y_test, y_pred, labels=["ham", "spam"])
    st.write(cm)
