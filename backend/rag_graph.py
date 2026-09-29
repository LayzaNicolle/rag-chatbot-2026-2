"""
Fluxo de RAG orquestrado com LangGraph.

FLUXO REFINADO (config.USE_REFINED_FLOW = True):

  receive_question -> classify --(INJECAO / FORA_DOMINIO)--------------> finalize
                         |
                         v
                  retrieve_context --(melhor score < MIN_SCORE)--------> finalize
                         |
                         v
                  evaluate_evidence --(INSUFICIENTE)------------------> finalize
                         |
                         v
                  build_prompt -> call_llm -> finalize

Os desvios para finalize NÃO chamam a LLM de geração (economia de tokens).

FLUXO ORIGINAL (USE_REFINED_FLOW = False): usado só para o "antes":
  receive_question -> retrieve_context -> build_prompt -> call_llm -> finalize

Chaining: a saída do classificador (JSON) decide a próxima aresta; a saída do
avaliador (SUFICIENTE/INSUFICIENTE) decide se o gerador é chamado.
"""
import json
import re
from typing import TypedDict

import faiss
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, START, END

from backend import config, prompts
from backend.llm_client import generate_answer, LLMError

MSG_NOT_FOUND = "Não encontrei essa informação na base consultada."
MSG_OUT_OF_DOMAIN = (
    "Só posso ajudar com informações sobre destinos turísticos brasileiros "
    "presentes na minha base. Pode reformular sua pergunta?"
)
MSG_REFUSED = (
    "Não posso atender a esse tipo de solicitação. "
    "Posso ajudar com dúvidas sobre turismo no Brasil."
)

VALID_CATEGORIES = {"TURISMO", "FORA_DOMINIO", "INJECAO"}


# ---------------------------------------------------------------------------
# Estado compartilhado
# ---------------------------------------------------------------------------
class RAGState(TypedDict, total=False):
    question: str
    chat_history: list[dict]
    category: str          # TURISMO | FORA_DOMINIO | INJECAO
    is_safe: bool
    confidence: str
    retrieved_chunks: list[dict]
    prompt_messages: list[dict]
    answer: str
    outcome: str           # answered | refused | out_of_domain | no_evidence | error
    sources: list[str]
    error: str


# ---------------------------------------------------------------------------
# Recursos (índice, metadados, embeddings)
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
                {"text": chunk["text"], "source": chunk["source"], "score": float(score)}
            )
        return results


# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------
_DELIM_RE = re.compile(r"</?(contexto|pergunta|historico)>", re.IGNORECASE)


def sanitize(text: str) -> str:
    """Neutraliza tags delimitadoras dentro dos dados, para que nem o usuário
    nem um documento consigam 'fechar' a região <contexto>/<pergunta> e
    injetar instruções fora dela."""
    return _DELIM_RE.sub(lambda m: m.group(0).replace("<", "[").replace(">", "]"), text)


def format_context(chunks: list[dict]) -> str:
    return "\n\n---\n\n".join(
        f"[Fonte: {c['source']}]\n{sanitize(c['text'])}" for c in chunks
    )


def format_history(history: list[dict], turns: int) -> str:
    recent = (history or [])[-(turns * 2):]
    if not recent:
        return "(sem histórico)"
    lines = []
    for t in recent:
        who = "Usuário" if t["role"] == "user" else "Assistente"
        lines.append(f"{who}: {sanitize(t['content'])[:300]}")
    return "\n".join(lines)


def parse_classification(raw: str) -> dict:
    """Trata a saída da LLM como DADO da aplicação (JSON validado).
    Se falhar, cai num padrão seguro: segue o fluxo (o gerador continua protegido)."""
    fallback = {"category": "TURISMO", "is_safe": True, "confidence": "BAIXA", "parse_ok": False}
    match = re.search(r"\{.*?\}", raw, re.DOTALL)
    if not match:
        return fallback
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return fallback
    category = str(data.get("category", "")).upper()
    if category not in VALID_CATEGORIES:
        return fallback
    is_safe = bool(data.get("is_safe", True)) and category != "INJECAO"
    return {
        "category": category,
        "is_safe": is_safe,
        "confidence": str(data.get("confidence", "BAIXA")).upper(),
        "parse_ok": True,
    }


def classify_question(question: str, history: list[dict], fewshot: bool) -> dict:
    """Chama o classificador (zero-shot ou few-shot). Exposta para testes."""
    system = (
        prompts.SYSTEM_CLASSIFIER_PROMPT_FEWSHOT
        if fewshot
        else prompts.SYSTEM_CLASSIFIER_PROMPT
    )
    user = prompts.CLASSIFIER_USER_TEMPLATE.format(
        history=format_history(history, config.CLASSIFIER_HISTORY_TURNS),
        question=sanitize(question),
    )
    raw = generate_answer(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.0,
        max_tokens=80,
    )
    return parse_classification(raw)


# ---------------------------------------------------------------------------
# Nós
# ---------------------------------------------------------------------------
def make_nodes(resources: RAGResources):
    def receive_question(state: RAGState) -> RAGState:
        question = (state.get("question") or "").strip()
        if not question:
            return {"error": "Pergunta vazia."}
        return {"question": question}

    # ----- fluxo refinado -------------------------------------------------
    def classify(state: RAGState) -> RAGState:
        if state.get("error"):
            return {}
        try:
            result = classify_question(
                state["question"], state.get("chat_history") or [], config.CLASSIFIER_FEWSHOT
            )
        except LLMError as exc:
            return {"error": str(exc)}

        update = {
            "category": result["category"],
            "is_safe": result["is_safe"],
            "confidence": result["confidence"],
        }
        if result["category"] == "INJECAO" or not result["is_safe"]:
            update["outcome"] = "refused"
            update["answer"] = MSG_REFUSED
        elif result["category"] == "FORA_DOMINIO":
            update["outcome"] = "out_of_domain"
            update["answer"] = MSG_OUT_OF_DOMAIN
        return update

    def retrieve_context(state: RAGState) -> RAGState:
        if state.get("error") or state.get("outcome"):
            return {}
        chunks = resources.search(state["question"], config.TOP_K)
        update: RAGState = {"retrieved_chunks": chunks}
        # Gate por score (sem LLM): nenhum chunk relevante -> sem evidência
        if config.USE_REFINED_FLOW and (not chunks or chunks[0]["score"] < config.MIN_SCORE):
            update["outcome"] = "no_evidence"
            update["answer"] = MSG_NOT_FOUND
        return update

    def evaluate_evidence(state: RAGState) -> RAGState:
        if state.get("error") or state.get("outcome") or not config.USE_EVALUATOR:
            return {}
        user = prompts.EVALUATOR_USER_TEMPLATE.format(
            context=format_context(state["retrieved_chunks"]),
            question=sanitize(state["question"]),
        )
        try:
            verdict = generate_answer(
                [
                    {"role": "system", "content": prompts.SYSTEM_EVALUATOR_PROMPT},
                    {"role": "user", "content": user},
                ],
                temperature=0.0,
                max_tokens=10,
            ).upper()
        except LLMError as exc:
            return {"error": str(exc)}
        # "INSUFICIENTE" contém "SUFICIENTE": testar o negativo primeiro
        if "INSUFICIENTE" in verdict:
            return {"outcome": "no_evidence", "answer": MSG_NOT_FOUND}
        return {}

    def build_prompt(state: RAGState) -> RAGState:
        if state.get("error") or state.get("outcome"):
            return {}
        messages = [{"role": "system", "content": prompts.SYSTEM_GENERATOR_PROMPT}]
        for turn in (state.get("chat_history") or [])[-(config.MAX_HISTORY_TURNS * 2):]:
            messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append(
            {
                "role": "user",
                "content": prompts.GENERATOR_USER_TEMPLATE.format(
                    context=format_context(state["retrieved_chunks"]),
                    question=sanitize(state["question"]),
                ),
            }
        )
        return {"prompt_messages": messages}

    # ----- fluxo original (antes) ----------------------------------------
    def build_prompt_legacy(state: RAGState) -> RAGState:
        if state.get("error"):
            return {}
        context_text = "\n\n---\n\n".join(
            f"[Fonte: {c['source']}]\n{c['text']}" for c in state["retrieved_chunks"]
        )
        messages = [{"role": "system", "content": config.SYSTEM_PROMPT}]
        for turn in (state.get("chat_history") or [])[-(config.MAX_HISTORY_TURNS * 2):]:
            messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append(
            {
                "role": "user",
                "content": (
                    f"CONTEXTO (trechos da base de conhecimento):\n{context_text}\n\n"
                    f"PERGUNTA DO USUÁRIO:\n{state['question']}"
                ),
            }
        )
        return {"prompt_messages": messages}

    # ----- comuns ---------------------------------------------------------
    def call_llm(state: RAGState) -> RAGState:
        if state.get("error") or state.get("outcome"):
            return {}
        try:
            answer = generate_answer(state["prompt_messages"])
        except LLMError as exc:
            return {"error": str(exc)}
        update: RAGState = {"answer": answer}
        if MSG_NOT_FOUND.lower().rstrip(".") in answer.lower():
            update["outcome"] = "no_evidence"
        return update

    def finalize(state: RAGState) -> RAGState:
        if state.get("error"):
            return {
                "answer": f"Desculpe, ocorreu um erro: {state['error']}",
                "sources": [],
                "outcome": "error",
            }
        outcome = state.get("outcome") or "answered"
        if outcome == "answered":
            sources = sorted({c["source"] for c in state.get("retrieved_chunks", [])})
        else:
            sources = []  # recusas / sem evidência não citam fontes
        return {"sources": sources, "outcome": outcome}

    return {
        "receive_question": receive_question,
        "classify": classify,
        "retrieve_context": retrieve_context,
        "evaluate_evidence": evaluate_evidence,
        "build_prompt": build_prompt,
        "build_prompt_legacy": build_prompt_legacy,
        "call_llm": call_llm,
        "finalize": finalize,
    }


# ---------------------------------------------------------------------------
# Roteamento (arestas condicionais)
# ---------------------------------------------------------------------------
def route_or_end(next_node: str):
    def _route(state: RAGState) -> str:
        return "finalize" if (state.get("error") or state.get("outcome")) else next_node
    return _route


# ---------------------------------------------------------------------------
# Montagem do grafo
# ---------------------------------------------------------------------------
def build_rag_graph():
    resources = RAGResources()
    n = make_nodes(resources)
    graph = StateGraph(RAGState)

    if not config.USE_REFINED_FLOW:
        # ---- fluxo ORIGINAL ----
        graph.add_node("receive_question", n["receive_question"])
        graph.add_node("retrieve_context", n["retrieve_context"])
        graph.add_node("build_prompt", n["build_prompt_legacy"])
        graph.add_node("call_llm", n["call_llm"])
        graph.add_node("finalize", n["finalize"])

        graph.add_edge(START, "receive_question")
        graph.add_edge("receive_question", "retrieve_context")
        graph.add_edge("retrieve_context", "build_prompt")
        graph.add_edge("build_prompt", "call_llm")
        graph.add_edge("call_llm", "finalize")
        graph.add_edge("finalize", END)
        return graph.compile()

    # ---- fluxo REFINADO ----
    for name in (
        "receive_question", "classify", "retrieve_context",
        "evaluate_evidence", "build_prompt", "call_llm", "finalize",
    ):
        graph.add_node(name, n[name])

    graph.add_edge(START, "receive_question")
    graph.add_conditional_edges(
        "receive_question", route_or_end("classify"),
        {"classify": "classify", "finalize": "finalize"},
    )
    graph.add_conditional_edges(
        "classify", route_or_end("retrieve_context"),
        {"retrieve_context": "retrieve_context", "finalize": "finalize"},
    )
    graph.add_conditional_edges(
        "retrieve_context", route_or_end("evaluate_evidence"),
        {"evaluate_evidence": "evaluate_evidence", "finalize": "finalize"},
    )
    graph.add_conditional_edges(
        "evaluate_evidence", route_or_end("build_prompt"),
        {"build_prompt": "build_prompt", "finalize": "finalize"},
    )
    graph.add_edge("build_prompt", "call_llm")
    graph.add_edge("call_llm", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()