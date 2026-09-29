"""
Cliente de LLM externa, usado nas etapas de classificação, avaliação de
evidência e geração da resposta do fluxo de RAG (nunca para embeddings).

Suporta:
- Groq (padrão, possui tier gratuito e API compatível com o formato OpenAI)
- OpenAI

Para trocar de provedor, basta alterar LLM_PROVIDER no .env.
Contrato: recebe uma lista de mensagens no formato OpenAI e retorna uma string.
"""
import requests

from backend import config


class LLMError(RuntimeError):
    pass


def _call_openai_compatible(
    url: str,
    api_key: str,
    model: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
) -> str:
    if not api_key:
        raise LLMError(
            "Chave de API não configurada. Defina a variável de ambiente "
            "correspondente no arquivo .env (ver README.md)."
        )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    # Modelos de raciocínio (gpt-oss): o "pensamento" conta no limite de tokens.
    # Reduz o esforço e dá folga para não devolver resposta vazia.
    if "gpt-oss" in model:
        payload["reasoning_effort"] = "low"
        payload["max_tokens"] = max_tokens + 500

    response = requests.post(url, headers=headers, json=payload, timeout=60)
    if response.status_code != 200:
        raise LLMError(
            f"Erro ao chamar a LLM externa ({response.status_code}): {response.text[:500]}"
        )

    data = response.json()
    try:
        return (data["choices"][0]["message"]["content"] or "").strip()
    except (KeyError, IndexError) as exc:
        raise LLMError(f"Resposta inesperada da LLM externa: {data}") from exc