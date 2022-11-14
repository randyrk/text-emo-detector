#   #   #  Frontendpart  #  #  #


#core pkgs
import streamlit as st
import altair as alt

# EDA pkgs
import pandas as pd
import numpy as np

# Utils
import joblib

pipe_lr = joblib.load(open("models/emotion_classifier.pkl","rb"))

def  predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results


img = "emoji.jpg"
emotions_dict = {"anger":"😤","disgust":"disgus","fear":"fee","happy":"hurray","joy":img,"neutral":"toto"}


def main():
    st.title("Emotion Classifier App")
    menu = ["Home","Monitor","About"]
    choice = st.sidebar.selectbox("Menu",menu)
    


    if choice == "Home":
        st.subheader("Home-Emotion In Text")

    with st.form(key = 'emotion_clf_form'):
            raw_text = st.text_area("Type Here")
            submit_text = st.form_submit_button(label='Submit')        

    if submit_text:
           col1,col2 = st.columns(2)


           prediction = predict_emotions(raw_text)
           probability = get_prediction_proba(raw_text)


    with col1:
        st.success("Original Text")
        st.write(raw_text)   

        st.success("Prediction")  
        emoji_icon = emotions_dict[prediction]  
        st.write(emoji_icon)
        st.write("Confidence:{}".format(np.max(probability)))

    with col2:
        st.success("Prediction Probability")
        st.write(probability)
        prob_df = pd.DataFrame(probability,columns=pipe_lr.classes_)
        st.write(prob_df.T)
        prob_df_clean = prob_df.T.reset_index()
        prob_df_clean.columns = ["emotions","probability"]
        fig = alt.Chart(prob_df_clean).mark_bar().encode(x='emotions',y='probability',color='emotions')
        st.altair_chart(fig,use_container_width=True)

    
main()

