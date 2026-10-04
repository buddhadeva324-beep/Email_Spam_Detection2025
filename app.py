import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Email Spam Detection",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.title {
    font-size: 42px;
    font-weight: 800;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #666;
    margin-bottom: 30px;
}

.card {
    padding: 20px;
    border-radius: 15px;
    background: white;
    border: 1px solid #e5e7eb;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.06);
    margin-bottom: 20px;
}

.metric-title {
    font-size: 15px;
    color: #666;
}

.metric-value {
    font-size: 30px;
    font-weight: 700;
}

.footer {
    text-align: center;
    color: #777;
    margin-top: 40px;
    padding: 20px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="title">📧 Email Spam Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning based Email Classification System'
    '</div>',
    unsafe_allow_html=True
)

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("⚙️ Project Information")

    st.write("### 🤖 Algorithm")
    st.write("Multinomial Naive Bayes")

    st.write("### 🔤 Text Processing")
    st.write("TF-IDF Vectorization")

    st.write("### 📊 Dataset")
    st.write("30 sample emails")

    st.write("### 🎯 Classes")
    st.write("• Spam")
    st.write("• Ham")

    st.divider()

    st.info(
        "This application predicts whether an email "
        "is Spam or Ham using Machine Learning."
    )

# =========================================================
# DATASET
# =========================================================

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

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    df["message"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# =========================================================
# TF-IDF
# =========================================================

vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# =========================================================
# MACHINE LEARNING MODEL
# =========================================================

model = MultinomialNB()

model.fit(
    X_train_tfidf,
    y_train
)

# =========================================================
# MODEL PREDICTION
# =========================================================

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(
    y_test,
    y_pred
)

# =========================================================
# DASHBOARD METRICS
# =========================================================

st.markdown("### 📊 Model Performance")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "🎯 Accuracy",
        f"{accuracy * 100:.2f}%"
    )

with col2:
    st.metric(
        "📧 Total Emails",
        len(df)
    )

with col3:
    st.metric(
        "🟢 Ham Emails",
        len(df[df["label"] == "ham"])
    )

with col4:
    st.metric(
        "🔴 Spam Emails",
        len(df[df["label"] == "spam"])
    )

st.divider()

# =========================================================
# EMAIL PREDICTION
# =========================================================

st.markdown("### 🔍 Check Your Email")

st.write(
    "Enter an email message below and the machine learning "
    "model will classify it."
)

email = st.text_area(
    "📨 Email Message",
    placeholder="Example: Congratulations! You won a free prize...",
    height=150
)

if st.button(
    "🔍 Check Email",
    use_container_width=True
):

    if email.strip():

        email_tfidf = vectorizer.transform([email])

        prediction = model.predict(
            email_tfidf
        )[0]

        probability = model.predict_proba(
            email_tfidf
        )[0]

        if prediction == "spam":

            st.error(
                "🚨 SPAM EMAIL DETECTED"
            )

            st.warning(
                "⚠️ Be careful with this email. "
                "It may contain unwanted or suspicious content."
            )

        else:

            st.success(
                "✅ HAM EMAIL"
            )

            st.info(
                "This email appears to be a normal message."
            )

        # Probability

        st.write("### 📊 Prediction Confidence")

        class_names = model.classes_

        probability_df = pd.DataFrame({
            "Class": class_names,
            "Probability": probability
        })

        probability_df["Probability"] = (
            probability_df["Probability"] * 100
        ).round(2)

        st.bar_chart(
            probability_df.set_index("Class")
        )

    else:

        st.warning(
            "⚠️ Please enter an email message first."
        )

# =========================================================
# CONFUSION MATRIX
# =========================================================

st.divider()

st.markdown("### 📈 Confusion Matrix")

st.write(
    "The confusion matrix shows how correctly the model "
    "classified Spam and Ham emails."
)

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["ham", "spam"]
)

fig, ax = plt.subplots(
    figsize=(7, 5)
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Ham", "Spam"]
)

display.plot(
    ax=ax,
    cmap="Blues",
    values_format="d"
)

ax.set_title(
    "Email Spam Detection - Confusion Matrix",
    fontsize=14,
    fontweight="bold"
)

ax.set_xlabel(
    "Predicted Label"
)

ax.set_ylabel(
    "Actual Label"
)

st.pyplot(
    fig,
    use_container_width=False
)

plt.close(fig)

# =========================================================
# DATASET
# =========================================================

st.divider()

st.markdown("### 📚 Dataset")

with st.expander(
    "📋 View Complete Dataset"
):

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

# =========================================================
# HOW IT WORKS
# =========================================================

st.markdown("### 🧠 How the System Works")

step1, step2, step3, step4 = st.columns(4)

with step1:
    st.markdown("### 1️⃣")
    st.write("Enter Email")
    st.caption("User enters an email message.")

with step2:
    st.markdown("### 2️⃣")
    st.write("TF-IDF")
    st.caption("Text is converted into numerical features.")

with step3:
    st.markdown("### 3️⃣")
    st.write("Naive Bayes")
    st.caption("Machine learning model analyzes the text.")

with step4:
    st.markdown("### 4️⃣")
    st.write("Prediction")
    st.caption("Email is classified as Spam or Ham.")

# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        📧 Email Spam Detection System<br>
        Built using Python • Scikit-learn • Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
