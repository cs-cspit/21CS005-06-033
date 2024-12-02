# import streamlit as st
# from sentence_transformers import SentenceTransformer, util
# import openai
# from utils import extract_text_from_pdf, generate_answer

# # Set OpenAI API key (you'll need to replace this with your actual key)
# openai.api_key = "sk-None-lrq1XtXL3hazF6wTdhrqT3BlbkFJLJTvsstl24C5fwsueFQv"

# # Load the model for semantic search
# model = SentenceTransformer('all-MiniLM-L6-v2')

# # Streamlit UI
# st.title("Multilingual PDF Question Answering Dashboard")
# st.write("Upload PDFs in any language (English, Gujarati, Hindi) and ask questions.")

# # File uploader (supports multiple PDF files)
# uploaded_files = st.file_uploader("Upload PDF Files", type="pdf", accept_multiple_files=True)

# # Initialize empty list for PDF texts
# pdf_texts = []

# if uploaded_files:
#     for uploaded_file in uploaded_files:
#         # Extract text from each uploaded PDF
#         text = extract_text_from_pdf(uploaded_file)
#         pdf_texts.append(text)
    
#     # Combine all PDFs into one big document
#     combined_text = " ".join(pdf_texts)

#     # Display input text box for user to ask questions
#     question = st.text_input("Ask a question based on the uploaded PDFs")

#     if question:
#         with st.spinner("Generating answer..."):
#             answer = generate_answer(combined_text, question, model)
#             st.write(f"**Answer**: {answer}")


import streamlit as st
from sentence_transformers import SentenceTransformer
import openai
from utils import extract_text_from_pdf, generate_answer

# Set OpenAI API key (replace with your actual key)
openai.api_key = ""

# Load the model for semantic search
model = SentenceTransformer('all-MiniLM-L6-v2')

# Streamlit UI
st.title("Multilingual PDF Question Answering Dashboard")
st.write("Upload PDFs in any language (English, Gujarati, Hindi) and ask questions.")

# File uploader (supports multiple PDF files)
uploaded_files = st.file_uploader("Upload PDF Files", type="pdf", accept_multiple_files=True)

# Initialize empty list for PDF texts
pdf_texts = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        # Extract text from each uploaded PDF
        text = extract_text_from_pdf(uploaded_file)
        pdf_texts.append(text)
    
    # Combine all PDFs into one large document
    combined_text = " ".join(pdf_texts)

    # Display input text box for user to ask questions
    question = st.text_input("Ask a question based on the uploaded PDFs")

    if question:
        with st.spinner("Generating answer..."):
            answer = generate_answer(combined_text, question, model)
            st.write(f"**Answer**: {answer}")
