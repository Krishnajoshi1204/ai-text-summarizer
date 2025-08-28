# streamlit_app.py
import streamlit as st
from summarizer import AbstractiveSummarizer, ExtractiveSummarizer

st.set_page_config(page_title='AI Text Summarizer')

st.title('AI Text Summarizer')

mode = st.radio('Mode', ['abstractive', 'extractive'])
model_choice = st.selectbox('Abstractive model (only used for abstractive)', ['facebook/bart-large-cnn', 't5-small'])

uploaded_files = st.file_uploader('Upload text files (optional)', accept_multiple_files=True, type=['txt'])

input_text = st.text_area('Or paste text here', height=300)

sentences = st.slider('Extractive: sentences', 1, 10, 3)

if st.button('Summarize'):
    texts = []
    if uploaded_files:
        for uf in uploaded_files:
            texts.append(uf.read().decode('utf-8'))
    if input_text.strip():
        texts.append(input_text)
    if not texts:
        st.warning('Please provide text input or upload files')
    else:
        if mode == 'abstractive':
            with st.spinner('Loading model and summarizing...'):
                s = AbstractiveSummarizer(model_name=model_choice)
                for i, t in enumerate(texts, 1):
                    st.subheader(f'Summary {i} (abstractive)')
                    st.write(s.summarize(t))
        else:
            s = ExtractiveSummarizer()
            for i, t in enumerate(texts, 1):
                st.subheader(f'Summary {i} (extractive)')
                st.write(s.summarize(t, sentences_count=sentences))
