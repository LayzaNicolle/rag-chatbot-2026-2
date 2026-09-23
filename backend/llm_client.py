"""
Cliente de LLM externa, usado exclusivamente na etapa de GERAÇÃO da resposta
final do fluxo de RAG (nunca para embeddings ou recuperação).

Suporta:
- Groq (padrão, possui tier gratuito e API compatível com o formato OpenAI)
- OpenAI

Para trocar de provedor, basta alterar LLM_PROVIDER no .env.
Para adicionar outro provedor (Hugging Face, Gemini, NVIDIA NIM, DeepSeek...),
implemente uma nova função `_call_<provedor>` seguindo o mesmo contrato:
recebe uma lista de mensagens no formato OpenAI e retorna uma string.
"""
import requests

from backend import config


class LLMError(RuntimeError):
    pass


def _call_openai_compatible(url: str, api_key: str, model: str, messages: list[dict]) -> str:
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
        "temperature": 0.3,
        "max_tokens": 700,
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


def generate_answer(messages: list[dict]) -> str:
    """
    Envia as mensagens (formato OpenAI: [{"role": ..., "content": ...}, ...])
    para o provedor de LLM configurado e retorna o texto da resposta.
    """
    provider = config.LLM_PROVIDER.lower()

    if provider == "groq":
        return _call_openai_compatible(
            config.GROQ_API_URL, config.GROQ_API_KEY, config.GROQ_MODEL, messages
        )
    if provider == "openai":
        return _call_openai_compatible(
            config.OPENAI_API_URL, config.OPENAI_API_KEY, config.OPENAI_MODEL, messages
        )

    raise LLMError(
        f"Provedor de LLM '{provider}' não suportado. Use 'groq' ou 'openai', "
        "ou implemente um novo provedor em backend/llm_client.py."
    )
