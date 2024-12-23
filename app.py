from flask import Flask, request, render_template
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pandas as pd
import os

app = Flask(__name__)

# Load your dataset
dataset_path = "Revenue chatbot dataset.xlsx"
if os.path.exists(dataset_path):
    df = pd.read_excel(dataset_path)  # Adjust path if needed
    questions = df['Questions '].tolist()  # List of questions
    answers = df['Answers'].tolist()  # List of answers
else:
    raise FileNotFoundError(f"The file {dataset_path} was not found.")

# Load the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for the questions
question_embeddings = model.encode(questions)

# Create a FAISS index for similarity search
index = faiss.IndexFlatL2(question_embeddings.shape[1])  # L2 distance for similarity search
index.add(question_embeddings)  # Add the question embeddings to the index

@app.route("/", methods=["GET", "POST"])
def ask():
    if request.method == "POST":
        user_query = request.form["query"]
        
        # Encode the user query
        query_embedding = model.encode([user_query])
        
        # Perform the similarity search
        D, I = index.search(query_embedding, 1)  # D = distances, I = indices of nearest neighbors
        
        # Get the most similar question and its answer
        similar_question = df.iloc[I[0][0]]['Questions ']
        similar_answer = df.iloc[I[0][0]]['Answers']
        
        return render_template("index.html", query=user_query, similar_question=similar_question, answer=similar_answer)
    
    return render_template("index.html")

if __name__ == "__main__":
    # Get the port from environment variable for Render deployment, default to 5000
    port = int(os.environ.get("PORT", 10000))  # Or try 5000 if issues occur
    # Run the app on 0.0.0.0 to make it publicly accessible
    app.run(host='0.0.0.0', port=port, debug=False)  # Disable debug for production
