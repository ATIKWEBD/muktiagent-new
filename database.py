import os
import uuid
import json
import asyncio
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

import chromadb
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from models import ProductOffer, PlatformTarget

load_dotenv()

# ==========================================
# 1. INITIALIZATION 
# ==========================================
# Automatically creates a local SQLite folder in your project
client = chromadb.PersistentClient(path="./ecommerce_chroma_db")

embeddings_model = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

COLLECTION_NAME = "ecommerce_offers"

def _init_chroma():
    """Runs instantly when the module loads, bypassing async blocks."""
    client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

# Initialize immediately
_init_chroma()

# ==========================================
# 2. WRITE OPERATION (Ingestion)
# ==========================================
async def upsert_offer(offer: ProductOffer, platform: PlatformTarget, job_id: str):
    bank_offers_str = ", ".join([b.bank_name for b in offer.bank_offers])
    semantic_text = f"Product: {offer.product_name}. Platform: {platform.value}. Banks with offers: {bank_offers_str}."
    
    vector = await embeddings_model.aembed_query(semantic_text)
    
    metadata = {
        "platform": platform.value,
        "job_id": job_id,
        "selling_price": float(offer.selling_price),
        "product_name": offer.product_name,
        "raw_json": offer.model_dump_json() 
    }
    
    point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{platform.value}_{offer.product_id}"))
    
    def _upsert():
        collection = client.get_collection(name=COLLECTION_NAME)
        collection.upsert(
            ids=[point_id],
            embeddings=[vector],
            metadatas=[metadata],
            documents=[semantic_text]
        )
        
    await asyncio.to_thread(_upsert)
    print(f"✅ Upserted {offer.product_name} into ChromaDB.")

# ==========================================
# 3. READ OPERATION (Hybrid RAG Search)
# ==========================================
async def search_offers(query: str, max_price: Optional[float] = None, platform: Optional[str] = None) -> List[Dict[str, Any]]:
    query_vector = await embeddings_model.aembed_query(query)
    
    where_clause = {}
    conditions = []
    
    if max_price:
        conditions.append({"selling_price": {"$lte": max_price}})
    if platform:
        conditions.append({"platform": {"$eq": platform}})
        
    if len(conditions) == 1:
        where_clause = conditions[0]
    elif len(conditions) > 1:
        where_clause = {"$and": conditions}
        
    def _query():
        collection = client.get_collection(name=COLLECTION_NAME)
        return collection.query(
            query_embeddings=[query_vector],
            where=where_clause if where_clause else None,
            n_results=5 
        )
        
    search_results = await asyncio.to_thread(_query)
    
    results = []
    if search_results and search_results['metadatas'] and len(search_results['metadatas'][0]) > 0:
        for meta in search_results['metadatas'][0]:
            results.append(json.loads(meta["raw_json"]))
            
    return results