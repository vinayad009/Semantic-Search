import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List
import shutil
from sentence_transformers import SentenceTransformer, util
import numpy as np

from .database import save_document_embedding, search_documents

app = FastAPI()

# Load pre-trained SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

class SearchResults(BaseModel):
    filenames: List[str]

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Ensure the uploads directory exists
        os.makedirs("uploads", exist_ok=True)
        
        # Save the uploaded file
        file_location = f"uploads/{file.filename}"
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Read the file content with proper encoding handling
        with open(file_location, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Compute embeddings
        embedding = model.encode(content, convert_to_tensor=True)
        
        # Save document embeddings
        save_document_embedding(file.filename, embedding.numpy())
        
        return {"filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# app.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
import shutil
from sentence_transformers import SentenceTransformer
from app.database import save_document_embedding, search_documents

app = FastAPI()

# Load pre-trained SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# HTML content for the web interface
html_content = """
<!DOCTYPE html>
<html>
<head>
  <title>Document Semantic Search</title>
  <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
  <script>
    $(document).ready(function(){
      $('#search-form').submit(function(event){
        event.preventDefault();
        var formData = new FormData();
        formData.append('query', $('#search-input').val());
        $.ajax({
          type: 'GET',
          url: '/docs',
          data: formData,
          processData: false,
          contentType: false,
          success: function(response) {
            $('#search-results').empty();
            response.filenames.forEach(function(filename) {
              $('#search-results').append('<li>' + filename + '</li>');
            });
          },
          error: function(xhr, status, error) {
            alert('Error: ' + status + '\nMessage: ' + error);
          }
        });
      });
    });
  </script>
</head>
<body>
  <h1>Document Semantic Search</h1>
  <form id="search-form">
    <input type="text" id="search-input" placeholder="Enter your search query...">
    <button type="submit">Search</button>
  </form>
  <ul id="search-results"></ul>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return HTMLResponse(content=html_content, status_code=200)

@app.get("/docs", response_model=list)
async def search_docs(query: str):
    query_embedding = model.encode(query)
    results = search_documents(query_embedding)
    return results

