#############################################################################################
######## Solução ainda na fila de aprendizado, projeto construido via prompt para IA ########
#############################################################################################
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Tuple

import numpy as np
import faiss

from app.documents import DOCUMENTS
from app.embeddings import TransformerEmbedder

TOP_K = 3

def build_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index

def search(
    query: str,
    embedder: TransformerEmbedder,
    index: faiss.IndexFlatIP,
    docs: List[str],
    top_k: int = TOP_K,
) -> List[Tuple[float, str]]:
    q_vec = embedder.encode([query]).numpy().astype("float32")
    scores, indices = index.search(q_vec, top_k)

    return [
        (round(float(scores[0][i]), 4), docs[indices[0][i]])
        for i in range(top_k)
    ]

def display(query: str, results: List[Tuple[float, str]]) -> None:
    print(f"\nConsulta : {query!r}")
    print("─" * 60)
    for rank, (score, doc) in enumerate(results, 1):
        print(f"  {rank}. [score={score:.4f}] {doc}")
    print()

def main() -> None:
    print("Carregando modelo de embeddings (transformer leve)...", flush=True)
    embedder = TransformerEmbedder()

    print(f"Gerando embeddings para {len(DOCUMENTS)} documentos...", flush=True)
    doc_emb = embedder.encode(DOCUMENTS).numpy().astype("float32")
    index = build_index(doc_emb)
    print(f"Índice FAISS pronto — {index.ntotal} vetores, dimensão {doc_emb.shape[1]}\n")

    demo_queries = [
        "Como percorrer elementos de uma lista?",
        "Quero usar Python para inteligência artificial",
        "Como tratar erros no código?",
    ]

    print("=== Demonstração ===")
    for q in demo_queries:
        display(q, search(q, embedder, index, DOCUMENTS))

    print("=== Modo interativo (digite 'sair' para encerrar) ===\n")
    while True:
        try:
            query = input("Consulta: ").strip()
        except EOFError:
            break
        if not query:
            continue
        if query.lower() in {"sair", "exit", "quit"}:
            break
        display(query, search(query, embedder, index, DOCUMENTS))

if __name__ == "__main__":
    main()