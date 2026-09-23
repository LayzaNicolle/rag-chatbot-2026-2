"""
Fluxo de RAG orquestrado com LangGraph.

Nós do grafo:
    receive_question   -> normaliza a entrada do usuário
    retrieve_context    -> busca os chunks mais relevantes no índice FAISS
    build_prompt         -> monta o prompt (system + contexto + histórico + pergunta)
    call_llm              -> chama a LLM externa para gerar a resposta final
    finalize              -> formata a saída (resposta + fontes usadas)

O grafo é montado uma única vez (build_rag_graph) e reutilizado a cada
requisição via graph.invoke(estado_inicial).
"""
import json
from typing import TypedDict

import faiss
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, START, END

from backend import config
from backend.llm_client import generate_answer, LLMError


# ---------------------------------------------------------------------------
# Estado compartilhado entre os nós do grafo
# ---------------------------------------------------------------------------
class RAGState(TypedDict, total=False):
    question: str
    chat_history: list[dict]      # [{"role": "user"/"assistant", "content": str}, ...]
    retrieved_chunks: list[dict]  # [{"text": str, "source": str, "score": float}, ...]
    prompt_messages: list[dict]   # mensagens no formato OpenAI para a LLM
    answer: str
    sources: list[str]
    error: str


# ---------------------------------------------------------------------------
# Recursos carregados uma única vez (índice, metadados, modelo de embeddings)
# ---------------------------------------------------------------------------
class RAGResources:
    def __init__(self):
        if not config.INDEX_PATH.exists() or not config.CHUNKS_PATH.exists():
            raise FileNotFoundError(
                "Índice vetorial não encontrado. Execute primeiro:\n"
                "    python backend/ingest.py"
            )
        self.index = faiss.read_index(str(config.INDEX_PATH))
        with open(config.CHUNKS_PATH, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        self.embedder = SentenceTransformer(config.EMBEDDING_MODEL_NAME)

    def search(self, query: str, top_k: int) -> list[dict]:
        query_vec = self.embedder.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")
        scores, indices = self.index.search(query_vec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            results.append(
                {
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "score": float(score),
                }
            )
        return results


# ---------------------------------------------------------------------------
# Nós do grafo
# ---------------------------------------------------------------------------
def make_nodes(resources: RAGResources):
    def receive_question(state: RAGState) -> RAGState:
        question = (state.get("question") or "").strip()
        if not question:
            return {**state, "error": "Pergunta vazia."}
        return {**state, "question": question}

    def retrieve_context(state: RAGState) -> RAGState:
        if state.get("error"):
            return state
        chunks = resources.search(state["question"], config.TOP_K)
        return {**state, "retrieved_chunks": chunks}

    def build_prompt(state: RAGState) -> RAGState:
        if state.get("error"):
            return state

        context_text = "\n\n---\n\n".join(
            f"[Fonte: {c['source']}]\n{c['text']}" for c in state["retrieved_chunks"]
        )

        messages = [{"role": "system", "content": config.SYSTEM_PROMPT}]

        # inclui histórico recente da conversa para dar continuidade ao diálogo
        history = state.get("chat_history") or []
        for turn in history[-(config.MAX_HISTORY_TURNS * 2):]:
            messages.append({"role": turn["role"], "content": turn["content"]})

        user_content = (
            f"CONTEXTO (trechos da base de conhecimento):\n{context_text}\n\n"
            f"PERGUNTA DO USUÁRIO:\n{state['question']}"
        )
        messages.append({"role": "user", "content": user_content})

        return {**state, "prompt_messages": messages}

    def call_llm(state: RAGState) -> RAGState:
        if state.get("error"):
            return state
        try:
            answer = generate_answer(state["prompt_messages"])
        except LLMError as exc:
            return {**state, "error": str(exc)}
        return {**state, "answer": answer}

    def finalize(state: RAGState) -> RAGState:
        if state.get("error"):
            return {
                **state,
                "answer": f"Desculpe, ocorreu um erro: {state['error']}",
                "sources": [],
            }
        sources = sorted({c["source"] for c in state.get("retrieved_chunks", [])})
        return {**state, "sources": sources}

    return receive_question, retrieve_context, build_prompt, call_llm, finalize


# ---------------------------------------------------------------------------
# Montagem do grafo
# ---------------------------------------------------------------------------
def build_rag_graph():
    resources = RAGResources()
    receive_question, retrieve_context, build_prompt, call_llm, finalize = make_nodes(
        resources
    )

    graph = StateGraph(RAGState)
    graph.add_node("receive_question", receive_question)
    graph.add_node("retrieve_context", retrieve_context)
    graph.add_node("build_prompt", build_prompt)
    graph.add_node("call_llm", call_llm)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "receive_question")
    graph.add_edge("receive_question", "retrieve_context")
    graph.add_edge("retrieve_context", "build_prompt")
    graph.add_edge("build_prompt", "call_llm")
    graph.add_edge("call_llm", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()
