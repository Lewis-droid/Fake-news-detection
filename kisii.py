import streamlit as st
import joblib

vectorizer = joblib.load("vectorizerkisii.jb")
model = joblib.load("lr_modelkisii.jb")

st.title("Kisii University Fake News Detector model")
st.write("Enter a news article below to check credibility")

news_input = st.text_area("Enter news article here")

if st.button("Check news article"):
    if news_input.strip():
        transformed_input = vectorizer.transform([news_input])
        prediction = model.predict(transformed_input)
        
        if prediction[0] == 1:
            st.success("This News is real!")
        else:
            st.error("This News is fake!")
    else:
        st.error("Please enter a news article to check")  