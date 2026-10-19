"""
Run:  uvicorn server:app --reload
Open: http://127.0.0.1:8000
"""

import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from groq import Groq

from common import COLLECTION, get_client, get_embedder

app = FastAPI()
client = get_client()
model = get_embedder()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MIN_SCORE = 0.25      # below this, a chunk is treated as unrelated (tune for your data)
DEDUP_SECONDS = 30    # drop hits from the same video that are this close together


def fmt_time(seconds):
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def search_chunks(query, top_k=5):
    vector = model.encode(query, normalize_embeddings=True).tolist()
    hits = client.query_points(
        collection_name=COLLECTION,
        query=vector,
        limit=top_k * 4,               # over-fetch, then de-duplicate
        with_payload=True,
        score_threshold=MIN_SCORE,
    ).points

    kept = []
    for h in hits:
        p = h.payload
        near_duplicate = any(
            k["video_id"] == p["video_id"] and abs(k["start"] - p["start"]) < DEDUP_SECONDS
            for k in kept
        )
        if near_duplicate:
            continue
        kept.append({
            "video_id": p["video_id"],
            "title": p["title"],
            "start": p["start"],
            "end": p["end"],
            "time": fmt_time(p["start"]),
            "text": p["text"],
            "score": round(h.score, 3),
        })
        if len(kept) == top_k:
            break
    return kept


def ask_llm(question, results):
    context = "\n\n".join(
        f"[{i}] Video: {r['title']} | Time: {r['time']}\n{r['text']}"
        for i, r in enumerate(results, 1)
    )
    prompt = f"""You help people find where a topic is discussed in a video playlist.
Use ONLY the transcript excerpts below. Briefly explain what is said about the topic,
and cite sources like [1], [2]. If the excerpts don't cover it, say:
"I don't know based on the provided information."

Excerpts:
{context}

Question: {question}"""

    resp = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


@app.get("/api/search")
def api_search(q: str, top_k: int = 5):
    if not q.strip():
        raise HTTPException(400, "Empty query")
    results = search_chunks(q, top_k)
    if not results:
        return {"answer": "I don't know based on the provided information.", "results": []}
    return {"answer": ask_llm(q, results), "results": results}


@app.get("/")
def index():
    return FileResponse("static/index.html")