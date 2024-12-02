# import PyPDF2
# import openai
# from sentence_transformers import SentenceTransformer, util
# from langdetect import detect
# from googletrans import Translator

# translator = Translator()

# # Function to extract text from uploaded PDF
# def extract_text_from_pdf(pdf_file):
#     pdf_reader = PyPDF2.PdfReader(pdf_file)
#     text = ""
#     for page_num in range(len(pdf_reader.pages)):
#         page = pdf_reader.pages[page_num]
#         text += page.extract_text()
#     return text

# # Function to detect language
# def detect_language(text):
#     try:
#         lang = detect(text)
#         return lang
#     except Exception as e:
#         return "unknown"

# # Function to translate text to English if it's in another language
# def translate_to_english(text, lang):
#     if lang != "en":
#         translated = translator.translate(text, src=lang, dest='en')
#         return translated.text
#     return text

# # Function to generate answer using OpenAI GPT
# def generate_answer(pdf_text, question, model):
#     # Step 1: Detect and translate the PDF text if necessary
#     lang = detect_language(pdf_text)
#     pdf_text_in_english = translate_to_english(pdf_text, lang)

#     # Step 2: Find the most relevant text chunk using SentenceTransformer
#     sentences = pdf_text_in_english.split(". ")
#     question_embedding = model.encode(question, convert_to_tensor=True)
#     sentence_embeddings = model.encode(sentences, convert_to_tensor=True)

#     # Find the top sentence based on similarity
#     similarities = util.pytorch_cos_sim(question_embedding, sentence_embeddings)
#     top_sentence_idx = similarities.argmax().item()
#     context = sentences[top_sentence_idx]

#     # Step 3: Send the context + question to OpenAI GPT for answering
#     prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
#     response = openai.Completion.create(
#         model="gpt-4o-mini",
#         prompt=prompt,
#         max_tokens=200
#     )
#     return response['choices'][0]['text'].strip()





import PyPDF2
import openai
from sentence_transformers import SentenceTransformer, util
from langdetect import detect
import logging

# Set up logging for accuracy metrics
logging.basicConfig(filename="accuracy_logs.log", level=logging.INFO, format='%(asctime)s - %(message)s')

# Function to extract text from uploaded PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text

# Function to detect language
def detect_language(text):
    try:
        lang = detect(text)
        return lang
    except Exception as e:
        return "unknown"

# Function to translate text to English using OpenAI Chat API
def translate_to_english(text, lang):
    if lang != "en":
        try:
            # Use OpenAI ChatCompletion for translation
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-instruct",
                messages=[
                    {"role": "system", "content": "You are a highly accurate translator."},
                    {"role": "user", "content": f"Translate this text to English:\n\n{text}"}
                ],
                max_tokens=2000
            )
            translated_text = response['choices'][0]['message']['content'].strip()
            # Log translation success
            logging.info(f"Translation successful from {lang} to English.")
            return translated_text
        except Exception as e:
            logging.error(f"Translation failed: {e}")
            return text  # Return original text if translation fails
    return text

# Function to chunk text into paragraphs
def chunk_text(text, chunk_size=500):
    sentences = text.split(". ")
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        current_chunk.append(sentence)
        current_length += len(sentence)
        if current_length >= chunk_size:
            chunks.append(". ".join(current_chunk))
            current_chunk = []
            current_length = 0

    if current_chunk:
        chunks.append(". ".join(current_chunk))
    
    return chunks

# Function to generate an answer using OpenAI GPT (Chat API)
def generate_answer(pdf_text, question, model):
    # Step 1: Detect and translate the PDF text if necessary
    lang = detect_language(pdf_text)
    pdf_text_in_english = translate_to_english(pdf_text, lang)

    # Step 2: Chunk the PDF text into paragraphs for better context retrieval
    chunks = chunk_text(pdf_text_in_english)

    # Step 3: Use SentenceTransformer to find the most relevant chunk
    question_embedding = model.encode(question, convert_to_tensor=True)
    chunk_embeddings = model.encode(chunks, convert_to_tensor=True)

    # Find the most similar chunk based on cosine similarity
    similarities = util.pytorch_cos_sim(question_embedding, chunk_embeddings)
    top_chunk_idx = similarities.argmax().item()
    context = chunks[top_chunk_idx]

    # Log similarity score for evaluation
    similarity_score = similarities[0, top_chunk_idx].item()
    logging.info(f"Top similarity score for answer generation: {similarity_score}")

    # Step 4: Use GPT-4 to generate an answer based on the most relevant chunk
    prompt = f"""
    The following text is an important part of the document that may help answer the question:

    {context}

    Now, answer the following question based on the provided document text:

    Question: {question}
    
    Answer accurately based on the information provided in the text:
    """
    
    # Use OpenAI ChatCompletion for answering
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-instruct",
        messages=[
            {"role": "system", "content": "You are a highly intelligent assistant who can understand document content and provide accurate answers."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500
    )
    
    answer = response['choices'][0]['message']['content'].strip()

    # Log final answer and return it
    logging.info(f"Generated answer: {answer}")
    return answer
