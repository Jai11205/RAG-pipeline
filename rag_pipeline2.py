from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import requests

WORD_TO_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
}

ORDINAL_MAP = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5
}

def extract_doc_id(query):
    query = query.lower()

    # 1. numeric match
    match = re.search(r"document\s*(\d+)", query)
    if match:
        return int(match.group(1))

    # 2. word match (three, five)
    for word, num in WORD_TO_NUM.items():
        if f"document {word}" in query:
            return num

    # 3. ordinal match (first document)
    for word, num in ORDINAL_MAP.items():
        if f"{word} document" in query:
            return num

    return None

def load_documents(folder="documents"):
    docs = []
    for file in os.listdir(folder):
        with open(os.path.join(folder, file), "r", encoding="utf-8") as f:
            docs.append(f.read())
    return docs

'''
def chunk_text(text, chunk_size=300, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start:start + chunk_size]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks
'''


# ─────────────────────────────────────────────────────────────────────────────
# Shared data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    text: str
    index: int                        # chunk number (0-based)
    char_start: int = 0               # start offset in original text
    char_end: int = 0                 # end offset in original text
    overlap_text: Optional[str] = None  # carried-over text from previous chunk
    depth: int = 0                    # recursion depth (recursive algo only)
    token_estimate: int = field(init=False)
    doc_id: Optional[int] = None   # which document this chunk belongs to


    def __post_init__(self):
        # rough estimate: 1 token ≈ 4 chars (GPT-style BPE)
        self.token_estimate = max(1, len(self.text) // 4)

    def __repr__(self):
        preview = self.text[:60].replace("\n", "↵")
        return (
            f"Chunk(doc={self.doc_id}, idx={self.index}, "
            f"chars={len(self.text)}, tokens≈{self.token_estimate}, "
            f"text='{preview}...')"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Boundary / separator definitions
# ─────────────────────────────────────────────────────────────────────────────

# Ordered from strongest (most semantic) to weakest
SEPARATOR_HIERARCHY = [
    ("paragraph", re.compile(r"\n[ \t]*\n")),
    ("sentence",  re.compile(r"(?<=[.!?])\s+")),
    ("clause",    re.compile(r"(?<=[,;:])\s+")),
]

# ─────────────────────────────────────────────────────────────────────────────
# 1. GREEDY BOUNDARY ALGORITHM
# ─────────────────────────────────────────────────────────────────────────────

class GreedyChunker:
    """
    Scan text left-to-right.  When the accumulated window hits `max_size`,
    search *backward* for the highest-priority boundary and cut there.
    Overlap is implemented by rewinding the next window's start position.

    Time complexity : O(n)  — single pass, bounded backward search
    Space complexity: O(k)  — k = number of chunks
    """

    def __init__(
        self,
        max_size: int = 500,
        overlap: int = 50,
        separators: Optional[list[tuple[str, re.Pattern]]] = None,
    ):
        if max_size <= 0:
            raise ValueError("max_size must be > 0")
        if overlap < 0 or overlap >= max_size:
            raise ValueError("overlap must be in [0, max_size)")

        self.max_size = max_size
        self.overlap = overlap
        # allow caller to pass a subset / reorder of SEPARATOR_HIERARCHY
        self.separators = separators if separators is not None else SEPARATOR_HIERARCHY

    # ── public ────────────────────────────────────────────────────────────────

    def chunk(self, text: str) -> list[Chunk]:
        """Return a list of Chunk objects for *text*."""
        if not text.strip():
            return []

        chunks: list[Chunk] = []
        pos = 0

        while pos < len(text):
            window_end = min(pos + self.max_size, len(text))

            # If we're not at the end of the text, find the best cut point
            if window_end < len(text):
                cut = self._find_best_cut(text, pos, window_end)
            else:
                cut = window_end

            raw = text[pos:cut]
            trimmed = raw.strip()

            if trimmed:
                # Carry overlap from the *previous* chunk's tail
                overlap_text = (
                    chunks[-1].text[-self.overlap:]
                    if chunks and self.overlap > 0
                    else None
                )
                chunks.append(
                    Chunk(
                        text=trimmed,
                        index=len(chunks),
                        char_start=pos,
                        char_end=cut,
                        overlap_text=overlap_text,
    # NEW:
                        doc_id=doc_id   # you add this
                    )
                )

            next_pos = cut - self.overlap

# Prevent backward or stuck movement
            if next_pos <= pos:
                next_pos = pos + self.max_size

            pos = next_pos

        return chunks

    # ── private ───────────────────────────────────────────────────────────────

    def _find_best_cut(self, text: str, start: int, end: int) -> int:
        window = text[start:end]
        min_cut = start + int(self.max_size * 0.5)  # 🔥 at least 50% size

        for _, pattern in self.separators:
            matches = list(pattern.finditer(window))
        
        # 🔥 filter only valid cuts
            valid_matches = [
            m for m in matches if (start + m.end()) >= min_cut
            ]

            if valid_matches:
                last = valid_matches[-1]
                return start + last.end()

        return end

model = SentenceTransformer("all-MiniLM-L6-v2")
documents = load_documents()
chunker = GreedyChunker(max_size=500, overlap=50)
all_chunks = []

for doc_id, doc in enumerate(documents):
    chunks = chunker.chunk(doc)
    
    for c in chunks:
        c.doc_id = doc_id   # 🔥 important
        all_chunks.append(c)
print('Chunks created:',len(all_chunks))

embeddings = model.encode([chunk.text for chunk in all_chunks])

data = [
    {"chunk": c, "embedding": e}
    for c, e in zip(all_chunks, embeddings)
]
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def retrieve(query, k=3):
    query_emb = model.encode([query])[0]

    doc_id = extract_doc_id(query)

    # 🔥 Filter by document if mentioned
    filtered_data = data
    if doc_id is not None:
        filtered_data = [
            item for item in data
            if item["chunk"].doc_id+1 == doc_id
        ]

    scores = []
    for item in filtered_data:
        sim = cosine_similarity(query_emb, item["embedding"])
        scores.append((sim, item["chunk"]))

    scores.sort(key=lambda x: x[0], reverse=True)

    top_chunks = [chunk for _, chunk in scores[:k]]

    return [
        f"### Document {chunk.doc_id+1}\n{chunk.text}"
        for chunk in top_chunks
    ]

def ask_llm(prompt):
    url = "http://localhost:11434/api/generate"

    response = requests.post(url, json={
        "model": "mistral",
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0}
    })

    return response.json()["response"]
def generate(query, chunks):

        context = "\n\n".join([
    f"{c}"
    for c in chunks
    ])
    
        prompt = f"""
You are an information extraction system.

Your task:
Extract the exact answer from the provided context.

STRICT RULES:
- Use ONLY the text present in the context
- Do NOT infer, assume, or rephrase beyond the context
- Do NOT combine information unless it is explicitly stated together
- Do NOT use prior knowledge

OUTPUT RULES:
- If an exact answer span exists → return it verbatim
- If the answer is not explicitly present → return exactly: I don't know

FORMAT:
- Output ONLY the answer
- No explanations
- No additional text

Context:
{context}

Question:
{query}

Answer:
"""

        return ask_llm(prompt)