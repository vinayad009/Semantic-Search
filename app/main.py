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

@app.get("/docs", response_model=SearchResults)
async def search_docs(q: str):
    try:
        query_embedding = model.encode(q, convert_to_tensor=True)
        results = search_documents(query_embedding.numpy())
        return SearchResults(filenames=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
