import os
import re
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import defaultdict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle  # Import modul pickle

# Unduh sumber daya yang diperlukan
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

# Inisialisasi stemmer Sastrawi
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Load stopwords
stop_words = set(stopwords.words('indonesian'))


def clean_text(text):
    text = text.lower()  # Konversi ke huruf kecil
    text = re.sub(r'[^\w\s]', '', text)  # Menghapus tanda baca
    text = re.sub(r'\d+', '', text)  # Menghapus angka
    return text


def tokenize(text):
    return word_tokenize(text)


def remove_stopwords(tokens):
    return [word for word in tokens if word not in stop_words]


def stem(tokens):
    return [stemmer.stem(token) for token in tokens]


def process_document(text):
    cleaned_text = clean_text(text)
    tokens = tokenize(cleaned_text)
    tokens_no_stopwords = remove_stopwords(tokens)
    stemmed_tokens = stem(tokens_no_stopwords)
    return stemmed_tokens


def create_index(folder_path):
    index = defaultdict(list)
    documents = {}
    titles = {}

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r', encoding='utf-8') as file:
                doc_text = file.read()

                title_match = re.search(r'judul:\s*(.*)', doc_text)
                if title_match:
                    doc_title = title_match.group(1)
                    titles[file_name] = doc_title
                else:
                    print(f"Judul tidak ditemukan dalam {file_name}.")
                    continue

                content_match = re.search(r'isi:\s*(.*)', doc_text, re.DOTALL)
                if content_match:
                    doc_content = content_match.group(1)
                else:
                    print(f"Isi tidak ditemukan dalam {file_name}.")
                    continue

                processed_tokens = process_document(doc_content)
                documents[file_name] = " ".join(processed_tokens)

                for token in processed_tokens:
                    index[token].append(file_name)

    return index, documents, titles


def save_index(index, documents, titles, filename):
    with open(filename, 'wb') as f:
        pickle.dump((index, documents, titles), f)


def load_index(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def vectorize_documents(documents):
    vectorizer = TfidfVectorizer()
    doc_vectors = vectorizer.fit_transform(documents.values())
    return vectorizer, doc_vectors


def jaccard_similarity(query_tokens, doc_tokens):
    intersection = set(query_tokens).intersection(set(doc_tokens))
    union = set(query_tokens).union(set(doc_tokens))
    return len(intersection) / len(union)


def search_jaccard(query, index, documents, titles):
    query_tokens = process_document(query)
    scores = {}

    for doc_name, doc_text in documents.items():
        doc_tokens = doc_text.split()
        score = jaccard_similarity(query_tokens, doc_tokens)
        scores[doc_name] = score

    sorted_scores = sorted(
        scores.items(), key=lambda item: item[1], reverse=True)
    results = [(titles[doc_name], doc_name, score)
               for doc_name, score in sorted_scores]
    return results


def search_vector_space_model(query, vectorizer, doc_vectors, titles):
    query_processed = " ".join(process_document(query))
    query_vector = vectorizer.transform([query_processed])
    cosine_similarities = cosine_similarity(
        query_vector, doc_vectors).flatten()

    scores = {i: cosine_similarities[i]
              for i in range(len(cosine_similarities))}
    sorted_scores = sorted(
        scores.items(), key=lambda item: item[1], reverse=True)

    results = [(titles[list(documents.keys())[idx]], list(
        documents.keys())[idx], score) for idx, score in sorted_scores]
    return results


# Contoh penggunaan:
folder_path = "C:/Users/LENOVO/Music/Search Engine/0. Crawling/Berita txt"
index_file = "index_data.pkl"

# Cek apakah file indeks sudah ada
if os.path.exists(index_file):
    # Muat indeks dari file
    index, documents, titles = load_index(index_file)
    print("Indeks dimuat dari file.")
else:
    # Buat indeks baru dan simpan ke file
    index, documents, titles = create_index(folder_path)
    save_index(index, documents, titles, index_file)
    print("Indeks baru dibuat dan disimpan ke file.")

vectorizer, doc_vectors = vectorize_documents(documents)
