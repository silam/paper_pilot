"""
TRANSFORM step (part 2): Chunking.

Why chunk at all? Embedding models have token limits, and retrieval precision
drops when one vector must represent a whole paper. Chunks that are too small
lose context; too large dilute relevance. Overlap prevents an answer being
split across a boundary.

Two strategies (selected by config, compared in evals):
  * recursive  - split on paragraph -> sentence -> word boundaries until each
                 piece fits chunk_size (same idea as LangChain's
                 RecursiveCharacterTextSplitter, written out so you can explain it).
  * sentence   - pack whole sentences into windows (never cuts mid-sentence).

Each chunk carries metadata (doc_id, title, position) so the answer can CITE
its source - essential for trust and hallucination checks.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from app.ingestion.loaders import RawDocument
# Regular expression for splitting text into sentences.
# Matches sentence boundaries based on punctuation followed by whitespace and a capital letter or number.
# Example: "This is a sentence. And another one!" -> ["This is a sentence.", "And another one!"]
# Note: This is a simple heuristic and may not handle all edge cases correctly.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[])")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    position: int
    metadata: dict = field(default_factory=dict)

#----------------------------------------------------------------------------------
def _recursive_split(text: str, size: int, separators: tuple[str, ...] = ("\n\n", "\n", ". ", " ")) -> list[str]:
    print("*********** _recursive_split")
    # Base case: if the text is already small enough, return it as a single chunk.
    if len(text) <= size:
        return [text]
    # If no separators are left, fall back to fixed-size chunks.
    if not separators:
        return [text[i:i + size] for i in range(0, len(text), size)]

    # Take the first separator and the rest for recursive splitting.
    sep, rest = separators[0], separators[1:]
    pieces: list[str] = []
    buf = "" 

    # Iterate over each part obtained by splitting the text with the current separator.
    for part in text.split(sep):
        # Construct a candidate chunk by combining the buffer and the current part.
        candidate = f"{buf}{sep}{part}" if buf else part

        # If the candidate chunk is small enough, keep adding to the buffer.
        if len(candidate) <= size:
            buf = candidate                       # keep packing into current chunk
            continue

        # Emit the current buffer as a full chunk if it exists.
        if buf:
            pieces.append(buf)                    # current chunk is full -> emit

        if len(part) > size:
            # A single part is itself too big -> recurse with a finer separator.
            pieces.extend(_recursive_split(part, size, rest))
            buf = ""
        else:
            # Start a new buffer with the current part.
            buf = part
    # Append any remaining buffer as the last chunk.
    if buf:
        pieces.append(buf)

    # Return the list of non-empty, stripped chunks.
    return [p.strip() for p in pieces if p.strip()]

#----------------------------------------------------------------------------------
def _with_overlap(pieces: list[str], overlap: int) -> list[str]:
    """Prefix each chunk with the tail of the previous one."""
    print("*********** _with_overlap")
    if overlap <= 0:
        return pieces
    out = [pieces[0]] if pieces else []
    for prev, cur in zip(pieces, pieces[1:]):
        tail = prev[-overlap:]
        tail = tail[tail.find(" ") + 1:] if " " in tail else tail  # start on a word
        out.append(f"{tail} {cur}")
    return out

#----------------------------------------------------------------------------------
def _sentence_split(text: str, size: int) -> list[str]:
    print("*********** _sentence_split")
    sentences = _SENTENCE_SPLIT.split(text)
    chunks, buf = [], ""
    for s in sentences:
        if buf and len(buf) + len(s) + 1 > size:
            chunks.append(buf)
            buf = s
        else:
            buf = f"{buf} {s}".strip()
    if buf:
        chunks.append(buf)
    return chunks

#------------------------------------------------------------------------
def chunk_document(doc: RawDocument, 
                   size: int = 800, # chunk size
                   overlap: int = 120, #chunk overlap
                   strategy: str = "recursive" #Chunk strategy
                   ) -> list[Chunk]:
    
    print("*********** chunk_document")
    if strategy == "sentence":
        pieces = _sentence_split(doc.text, size)
    else:
        pieces = _recursive_split(doc.text, size)

    # Prefix each chunk with the tail of the previous one
    pieces = _with_overlap(pieces, overlap)

    chunks = []
    for i, piece in enumerate(pieces):
        # Contextual chunk header: prepend the title so an isolated chunk still
        # "knows" which paper it belongs to (cheap trick, big retrieval win).
        text = f"Title: {doc.title}\n{piece}"
        cid = hashlib.md5(f"{doc.doc_id}:{i}:{piece[:50]}".encode()).hexdigest()[:16]
        chunks.append(Chunk(
            chunk_id=cid, 
            doc_id=doc.doc_id, 
            text=text, 
            position=i,
            metadata={"doc_id": doc.doc_id, 
                      "title": doc.title, 
                      "url": doc.url,
                      "source": doc.source, 
                      "published": doc.published, 
                      "position": i},
        ))
    return chunks
