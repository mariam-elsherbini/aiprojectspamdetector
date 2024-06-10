import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics


data = pd.read_csv("sms_spam.csv")

m = data["text"] 
k = data['type'] 

m_train, m_test, k_train, k_test = train_test_split(m, k, test_size=0.25, random_state=1)

vectorizer = CountVectorizer().fit(m_train)
sms_train_vectorized = vectorizer.transform(m_train)
sms_test_vectorized = vectorizer.transform(m_test)

clfr = MultinomialNB()
clfr.fit(sms_train_vectorized, k_train)


predicted = clfr.predict(sms_test_vectorized)
acc = metrics.accuracy_score(k_test, predicted)




st.title("Spam Detector")
st.image("spampic.jpg", use_column_width=True)
st.text("Model Description: This is a Naive Bayes Model that is trained on SMS data")



text = st.text_input("Enter text here", "Type here...")
predict = st.button("Predict")

if predict:
    newTestData = vectorizer.transform([text])
    predicted_label = clfr.predict(newTestData)[0]
    prediction_text = "Spam" if predicted_label == 'spam' else "Human"

    if predicted_label == 'spam':
        st.error(f"'{text}' is classified as {prediction_text}")
    else:
        st.success(f"'{text}' is classified as {prediction_text}")

