# MindTickle Semantic Search

This project implements a semantic search system for case study documents using FastAPI and Hugging Face's transformers for embeddings.

## Setup

1. Clone the repository.
2. Install dependencies using `pip install -r requirements.txt`.
3. Run the application using Docker:
   ```bash
   docker build -t mindtickle_search .
   docker run -d --name mindtickle_search -p 80:80 mindtickle_search
