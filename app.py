from flask import Flask, render_template, request, send_file
from flask import Flask, render_template, request
import os
import pickle
from modulse import load_index, search_jaccard, search_vector_space_model, vectorize_documents
import re

app = Flask(__name__)

# Load index from pickle file
index_file = "index_data.pkl"
index, documents, titles = load_index(index_file)
vectorizer, doc_vectors = vectorize_documents(documents)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/search', methods=['POST'])  # Accepting POST requests
def search_results():
    query = request.form.get('query')
    algorithm = request.form.get('algorithm')
    num_results = int(request.form.get('num_results', 5)
                      )  # Default to 5 if not specified

    if algorithm == 'jaccard':
        results = search_jaccard(query, index, documents, titles)
    else:
        results = search_vector_space_model(
            query, vectorizer, doc_vectors, titles)

    # Limit results to the specified number
    results = results[:num_results]

    # Pass documents to the template
    return render_template('result.html', query=query, algorithm=algorithm, results=results, documents=documents)


# Assuming 'folder_path' contains the path to your txt files
folder_path = "Berita txt"


@app.route('/document/<doc_name>')
def document(doc_name):
    file_path = os.path.join(folder_path, doc_name)

    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        title_match = re.search(r'judul:\s*(.*)', content)
        content_match = re.search(r'isi:\s*(.*)', content, re.DOTALL)

        if title_match and content_match:
            title = title_match.group(1).strip()
            document_content = content_match.group(1).strip()
            # Split the content into paragraphs by line breaks
            document_content = document_content.replace('\n', '</p><p>')
        else:
            title = "Title not found"
            document_content = "Content not found"

        return render_template('document.html', title=title, content=document_content)
    else:
        return "Document not found", 404


if __name__ == '__main__':
    app.run(debug=True)
