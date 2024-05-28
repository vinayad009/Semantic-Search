from pydantic import BaseModel
from typing import List

class UploadResponse(BaseModel):
    info: str

class SearchResponse(BaseModel):
    documents: List[str]
