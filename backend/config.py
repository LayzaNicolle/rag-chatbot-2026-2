import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"
DATA_DIR = BASE_DIR / "data"
INDEX_PATH = DATA_DIR / "faiss.index"
CHUNKS_PATH = DATA_DIR / "chunks.json"
FRONTEND_DIR = BASE_DIR / "frontend"

EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME", "paraphrase-multilingual-MiniLM-L12-v2"
)

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

TOP_K = int(os.getenv("TOP_K", "4"))

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "4"))

# ---------------------------------------------------------------------------
# Engenharia de Prompt (atividade atual)
# ---------------------------------------------------------------------------
# True  = fluxo refinado (classificador -> gate de score -> avaliador -> gerador)
# False = fluxo ORIGINAL (usado para gerar o "antes" da comparação)
USE_REFINED_FLOW = os.getenv("USE_REFINED_FLOW", "1") == "1"

# False = classificador zero-shot | True = classificador few-shot
CLASSIFIER_FEWSHOT = os.getenv("CLASSIFIER_FEWSHOT", "0") == "1"

# Score mínimo (similaridade de cosseno) do MELHOR chunk. Abaixo disso, o grafo
# responde "não encontrei" sem chamar a LLM. Ajuste observando os scores reais.
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.47"))

# Liga/desliga o nó avaliador de evidência (1 chamada extra de LLM)
USE_EVALUATOR = os.getenv("USE_EVALUATOR", "1") == "1"

# Quantos turnos recentes o classificador enxerga (para perguntas de continuação)
CLASSIFIER_HISTORY_TURNS = int(os.getenv("CLASSIFIER_HISTORY_TURNS", "2"))

# Prompt ORIGINAL (mantido apenas para o fluxo "antes")
SYSTEM_PROMPT = (
    "Você é um assistente virtual especialista em turismo no Brasil. "
    "Responda SEMPRE em português, de forma clara e objetiva, baseando-se "
    "exclusivamente no CONTEXTO fornecido abaixo, extraído da base de "
    "conhecimento. Se a resposta não puder ser encontrada no contexto, "
    "diga claramente que não possui essa informação na base de conhecimento "
    "em vez de inventar uma resposta. Não mencione que está usando um "
    "'contexto' na resposta final; apenas responda naturalmente."
)