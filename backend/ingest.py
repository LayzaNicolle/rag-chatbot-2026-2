"""
Pipeline de ingestão da base de conhecimento.

Executa as etapas:
1. Coleta de conteúdo (leitura dos arquivos .txt em knowledge_base/)
2. Limpeza e organização básica dos textos
3. Quebra do conteúdo em chunks
4. Geração de embeddings (modelo local, via sentence-transformers)
5. Armazenamento dos embeddings em um índice vetorial (FAISS)

Uso:
    python backend/ingest.py
"""
import json
import re
import sys
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.append(str(Path(__file__).resolve().parent.parent))

from backend import config  # noqa: E402


def clean_text(text: str) -> str:
    """Limpeza básica: normaliza espaços e quebras de linha excessivas."""
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_documents() -> list[dict]:
    """Lê todos os arquivos .txt da base de conhecimento."""
    docs = []
    for path in sorted(config.KNOWLEDGE_BASE_DIR.glob("*.txt")):
        raw = path.read_text(encoding="utf-8")
        docs.append({"source": path.name, "text": clean_text(raw)})
    if not docs:
        raise RuntimeError(
            f"Nenhum arquivo .txt encontrado em {config.KNOWLEDGE_BASE_DIR}"
        )
    return docs


def split_into_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Quebra o texto em chunks de tamanho aproximado `chunk_size` caracteres,
    com sobreposição `overlap`, respeitando limites de parágrafo/frase quando
    possível (split simples baseado em caracteres, sem dependências pesadas).
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= chunk_size:
            current = f"{current}\n\n{para}".strip()
        else:
            if current:
                chunks.append(current)
            # se o próprio parágrafo for maior que chunk_size, quebra à força
            if len(para) > chunk_size:
                start = 0
                while start < len(para):
                    end = start + chunk_size
                    chunks.append(para[start:end])
                    start = end - overlap
                current = ""
            else:
                current = para

    if current:
        chunks.append(current)

    # aplica overlap entre chunks consecutivos (em nível de caracteres)
    overlapped = []
    for i, ch in enumerate(chunks):
        if i == 0:
            overlapped.append(ch)
        else:
            prev_tail = chunks[i - 1][-overlap:] if overlap > 0 else ""
            overlapped.append((prev_tail + "\n" + ch).strip())
    return overlapped


def build_index():
    print(f"[1/5] Lendo documentos de {config.KNOWLEDGE_BASE_DIR} ...")
    documents = load_documents()
    print(f"      {len(documents)} documentos encontrados.")

    print("[2/5] Quebrando documentos em chunks ...")
    all_chunks = []  # lista de dicts: {text, source, chunk_id}
    for doc in documents:
        pieces = split_into_chunks(
            doc["text"], config.CHUNK_SIZE, config.CHUNK_OVERLAP
        )
        for idx, piece in enumerate(pieces):
            all_chunks.append(
                {
                    "id": len(all_chunks),
                    "source": doc["source"],
                    "chunk_index": idx,
                    "text": piece,
                }
            )
    print(f"      {len(all_chunks)} chunks gerados "
          f"(chunk_size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP}).")

    print(f"[3/5] Carregando modelo de embeddings '{config.EMBEDDING_MODEL_NAME}' ...")
    model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)

    print("[4/5] Gerando embeddings ...")
    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(
        texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True
    ).astype("float32")

    print("[5/5] Construindo e salvando índice FAISS ...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # produto interno = similaridade de cosseno (vetores normalizados)
    index.add(embeddings)

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(config.INDEX_PATH))
    with open(config.CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"\nÍndice salvo em: {config.INDEX_PATH}")
    print(f"Metadados salvos em: {config.CHUNKS_PATH}")
    print(f"Total de vetores no índice: {index.ntotal}")


if __name__ == "__main__":
    build_index()
