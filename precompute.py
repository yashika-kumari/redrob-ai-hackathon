import os
import json
import numpy as np
import sys
from sentence_transformers import SentenceTransformer

# Add workspace root to python path to import embedder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from app.services.embedder import chunk_text, mean_pool_embeddings

def build_candidate_text(cand: dict) -> str:
    profile = cand.get("profile", {})
    headline = profile.get("headline", "")
    summary = profile.get("summary", "")
    current_title = profile.get("current_title", "")
    current_industry = profile.get("current_industry", "")
    
    text_parts = [
        f"headline: {headline}",
        f"current title: {current_title}",
        f"industry: {current_industry}",
        f"summary: {summary}"
    ]
    
    career_history = cand.get("career_history", [])
    if career_history:
        text_parts.append("work history:")
        for job in career_history:
            title = job.get("title", "")
            company = job.get("company", "")
            industry = job.get("industry", "")
            desc = job.get("description", "")
            text_parts.append(f"- {title} at {company} ({industry}). description: {desc}")
            
    skills = cand.get("skills", [])
    if skills:
        skill_names = [s.get("name", "") for s in skills if s.get("name")]
        text_parts.append("skills: " + ", ".join(skill_names))
        
    education = cand.get("education", [])
    if education:
        edu_parts = []
        for edu in education:
            deg = edu.get("degree", "")
            field = edu.get("field_of_study", "")
            inst = edu.get("institution", "")
            edu_parts.append(f"{deg} in {field} from {inst}")
        text_parts.append("education: " + "; ".join(edu_parts))
        
    return "\n".join(text_parts).lower()

def main():
    data_dir = r"data\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge"
    candidates_path = os.path.join(data_dir, "candidates.jsonl")
    
    print("Loading sentence-transformer model from local directory...")
    model = SentenceTransformer("model/all-MiniLM-L6-v2")
    
    print("Reading candidates from jsonl...")
    candidates = []
    with open(candidates_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                candidates.append(json.loads(line))
                
    total_cands = len(candidates)
    print(f"Total candidates loaded: {total_cands}")
    
    # Pre-allocate array for embeddings
    embeddings = np.zeros((total_cands, 384), dtype=np.float32)
    metadata = []
    
    print("Processing candidates and chunking text...")
    all_chunks = []
    chunk_to_candidate_map = []
    
    for i, cand in enumerate(candidates):
        text = build_candidate_text(cand)
        chunks = chunk_text(text)
        for chunk in chunks:
            all_chunks.append(chunk)
            chunk_to_candidate_map.append(i)
            
        # Store metadata
        metadata.append({
            "candidate_id": cand["candidate_id"],
            "profile": {
                "anonymized_name": cand["profile"].get("anonymized_name"),
                "headline": cand["profile"].get("headline"),
                "years_of_experience": cand["profile"].get("years_of_experience"),
                "location": cand["profile"].get("location"),
                "country": cand["profile"].get("country"),
                "current_title": cand["profile"].get("current_title"),
                "current_company": cand["profile"].get("current_company"),
                "current_company_size": cand["profile"].get("current_company_size"),
                "current_industry": cand["profile"].get("current_industry"),
            },
            "career_history": [{
                "company": j.get("company"),
                "title": j.get("title"),
                "duration_months": j.get("duration_months"),
                "is_current": j.get("is_current"),
                "industry": j.get("industry"),
                "company_size": j.get("company_size"),
                "description": j.get("description"),
            } for j in cand.get("career_history", [])],
            "skills": [{
                "name": s.get("name"),
                "proficiency": s.get("proficiency"),
                "duration_months": s.get("duration_months"),
            } for s in cand.get("skills", [])],
            "redrob_signals": cand["redrob_signals"],
        })
        
        if (i + 1) % 10000 == 0:
            print(f"  Processed {i + 1}/{total_cands} candidates")
            
    print(f"Total text chunks to embed: {len(all_chunks)}")
    print("Encoding chunks with SentenceTransformer...")
    
    # We will encode chunks in batches to show progress
    batch_size = 256
    chunk_embeddings = model.encode(
        all_chunks,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype(np.float32)
    
    print("Aggregating chunk embeddings via mean pooling...")
    # Group chunk embeddings by candidate
    candidate_chunks = {}
    for chunk_idx, cand_idx in enumerate(chunk_to_candidate_map):
        if cand_idx not in candidate_chunks:
            candidate_chunks[cand_idx] = []
        candidate_chunks[cand_idx].append(chunk_embeddings[chunk_idx])
        
    for cand_idx, vecs in candidate_chunks.items():
        doc_vec = mean_pool_embeddings(np.array(vecs))
        embeddings[cand_idx] = doc_vec
        
    print("Saving pre-computed artifacts...")
    # Make sure target directory exists
    os.makedirs("data", exist_ok=True)
    np.save("data/candidate_embeddings.npy", embeddings)
    with open("data/candidate_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False)
        
    print("Pre-computation completed successfully!")

if __name__ == "__main__":
    main()
