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

    response = requests.post(url, headers=headers, json=payload, timeout=60)
    if response.status_code != 200:
        raise LLMError(
            f"Erro ao chamar a LLM externa ({response.status_code}): {response.text[:500]}"
        )

    data = response.json()
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError) as exc:
        raise LLMError(f"Resposta inesperada da LLM externa: {data}") from exc


def generate_answer(
    messages: list[dict], temperature: float = 0.3, max_tokens: int = 700
) -> str:
    """
    Envia as mensagens (formato OpenAI) para o provedor configurado e retorna
    o texto da resposta. Classificador e avaliador usam temperature=0.
    """
    provider = config.LLM_PROVIDER.lower()

    if provider == "groq":
        return _call_openai_compatible(
            config.GROQ_API_URL, config.GROQ_API_KEY, config.GROQ_MODEL,
            messages, temperature, max_tokens,
        )
    if provider == "openai":
        return _call_openai_compatible(
            config.OPENAI_API_URL, config.OPENAI_API_KEY, config.OPENAI_MODEL,
            messages, temperature, max_tokens,
        )

    raise LLMError(
        f"Provedor de LLM '{provider}' não suportado. Use 'groq' ou 'openai', "
        "ou implemente um novo provedor em backend/llm_client.py."
    )