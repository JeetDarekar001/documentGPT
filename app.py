import streamlit as st


st.title("Document GPT")

with st.form("submit_form"):
    text = st.text_input(label="Enter your query.")
    button=st.form_submit_button("Answer")

if button:
    st.write(text)

