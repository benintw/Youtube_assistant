import streamlit as st
import langchain_helper as lch
import textwrap


st.title("Youtube Assistant")

st.markdown("-" * 30)
st.markdown("##### My answer is less than 100 words, cuz it costs $$$$.")


with st.sidebar:
    with st.form(key="my_form"):
        openai_api_key = st.text_input(
            label="Enter your OpenAI API Key", type="password"
        )
        youtube_url = st.sidebar.text_area(
            label="What is the youtube vid url?", max_chars=60
        )
        query = st.sidebar.text_area(
            label="Ask me about the video", max_chars=50, key="query"
        )

        submit_buttom = st.form_submit_button(label="Submit my question")

if query and youtube_url and openai_api_key and submit_buttom:

    db = lch.create_vector_db_from_youtube_url(youtube_url, openai_api_key)
    response, docs = lch.get_response_from_query(db, query, openai_api_key)
    st.subheader("Answer: ")
    st.text(textwrap.fill(response, width=80))
