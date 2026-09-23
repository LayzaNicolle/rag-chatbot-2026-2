# Chatbot RAG — Turismo no Brasil

Chatbot web desenvolvido em **Python** para responder perguntas em linguagem natural sobre **turismo no Brasil**, utilizando **Retrieval-Augmented Generation (RAG)**, busca vetorial e **LangGraph** para orquestração do fluxo.

## Equipe

* Estevão Chagas
* Layza Nicolle
* Matheus Pablo
* Vinicius Simas

## Sobre o projeto

O sistema utiliza uma **base de conhecimento própria** sobre turismo brasileiro para recuperar informações relevantes antes da geração da resposta.

O fluxo principal é:

```text
Pergunta
   ↓
Embeddings
   ↓
Busca vetorial
   ↓
Contexto relevante
   ↓
LangGraph
   ↓
LLM
   ↓
Resposta
```

A utilização de RAG permite que as respostas sejam fundamentadas nos documentos disponíveis na base de conhecimento, reduzindo a necessidade de a LLM utilizar informações externas ao conteúdo fornecido.

## Domínio e base de conhecimento

O domínio escolhido foi **Turismo no Brasil**.

A base contém **30 arquivos `.txt`**, abrangendo destinos turísticos e temas relacionados ao turismo, como:

* Rio de Janeiro
* São Paulo
* Salvador
* Fortaleza
* Foz do Iguaçu
* Amazônia
* Pantanal
* Fernando de Noronha
* Gramado e Canela
* Florianópolis
* Brasília
* Jericoacoara
* Lençóis Maranhenses
* Carnaval
* Festas Juninas
* Transporte
* Segurança
* Moeda e câmbio
* Ecoturismo
* Patrimônios da UNESCO
* Recife e Olinda

Os documentos ficam armazenados em:

```text
knowledge_base/
```

## Arquitetura

```text
┌──────────────────┐
│    Front-end     │
│    HTML/CSS/JS   │
└────────┬─────────┘
         │ HTTP
         ▼
┌──────────────────┐
│   FastAPI API    │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│    LangGraph     │
│                  │
│ Recuperação      │
│ Contexto         │
│ Geração          │
└───────┬──────────┘
        │
   ┌────┴─────┐
   ▼          ▼
 FAISS       LLM
   │
   ▼
Base de
Conhecimento
```

## Pipeline RAG

1. Os documentos da `knowledge_base/` são carregados.
2. O conteúdo é dividido em **chunks**.
3. Cada chunk é transformado em um **embedding**.
4. Os embeddings são armazenados em um índice vetorial **FAISS**.
5. O usuário envia uma pergunta pelo chatbot.
6. O sistema transforma a pergunta em embedding.
7. O FAISS recupera os trechos mais relevantes.
8. O LangGraph organiza o contexto recuperado.
9. A LLM recebe a pergunta juntamente com o contexto.
10. A resposta é retornada ao usuário.

Esse processo permite que a geração seja baseada nas informações recuperadas da própria base de conhecimento.

## Estrutura do projeto

```text
rag-chatbot/
│
├── frontend/
│   └── index.html
│
├── backend/
│   ├── main.py
│   ├── config.py
│   ├── __init__.py
│   ├── rag_graph.py
│   ├── llm_client.py
│   └── ingest.py
│
├── knowledge_base/
│   ├── 01_rio_de_janeiro.txt
│   ├── 02_sao_paulo.txt
│   ├── ...
│   └── 30_recife_olinda.txt
│
├── data/
│   └── .gitkeep
│
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

### Principais componentes

| Componente              | Função                              |
| ----------------------- | ----------------------------------- |
| `frontend/index.html`   | Interface do chatbot                |
| `backend/main.py`       | API e endpoints                     |
| `backend/config.py`     | Configurações da aplicação          |
| `backend/rag_graph.py`  | Fluxo RAG com LangGraph             |
| `backend/llm_client.py` | Comunicação com a LLM               |
| `backend/ingest.py`     | Processamento e indexação da base   |
| `knowledge_base/`       | Documentos utilizados pelo RAG      |
| `data/`                 | Arquivos gerados pelo processamento |

## Tecnologias

* **Python**
* **FastAPI**
* **LangGraph**
* **FAISS**
* **Sentence Transformers**
* **Embeddings**
* **Groq / LLM externa**
* **HTML, CSS e JavaScript**
* **Pydantic**
* **Uvicorn**

## Pré-requisitos

* Python **3.10+**
* `pip`
* Chave de API da LLM
* Git

## Configuração

Clone o projeto:

```bash
git clone https://github.com/SEU-USUARIO/SEU-REPOSITORIO.git
cd SEU-REPOSITORIO
```

Crie o ambiente virtual:

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

Crie um arquivo `.env` na raiz do projeto:

```env
LLM_PROVIDER=groq
GROQ_API_KEY=sua_chave_aqui
GROQ_MODEL=llama-3.1-8b-instant
```

## Executando o projeto

Primeiro, processe a base de conhecimento:

```bash
python backend/ingest.py
```

Depois, inicie a API:

```bash
uvicorn backend.main:app --reload
```

Acesse:

```text
http://127.0.0.1:8000
```

### Endpoints

| Método | Endpoint  | Descrição                |
| ------ | --------- | ------------------------ |
| `GET`  | `/`       | Interface do chatbot     |
| `POST` | `/chat`   | Envia uma pergunta       |
| `GET`  | `/health` | Verifica o status da API |
| `GET`  | `/docs`   | Documentação da API      |

## Exemplo de requisição

```json
{
  "message": "Quais são os principais pontos turísticos de Recife?",
  "session_id": "123"
}
```

## Exemplo de resposta

```json
{
  "session_id": "123",
  "answer": "Resposta gerada com base na base de conhecimento.",
  "sources": []
}
```

## Histórico de conversa

O chatbot mantém o histórico das mensagens por meio de um `session_id`.

Isso permite que perguntas de continuidade sejam consideradas dentro da mesma conversa.

A quantidade de turnos armazenados pode ser configurada através de:

```env
MAX_HISTORY_TURNS=4
```

## Configurações principais

As principais configurações podem ser ajustadas através do arquivo `.env`:

```env
EMBEDDING_MODEL_NAME=paraphrase-multilingual-MiniLM-L12-v2
CHUNK_SIZE=800
CHUNK_OVERLAP=120
TOP_K=4
MAX_HISTORY_TURNS=4
```

## Considerações

O projeto foi desenvolvido com foco acadêmico, demonstrando a integração entre:

```text
Base de Conhecimento
        +
Embeddings
        +
Busca Vetorial
        +
LangGraph
        +
LLM
        =
Chatbot RAG
```

A aplicação utiliza a base de conhecimento como fonte para recuperação das informações antes da geração das respostas, demonstrando na prática o funcionamento de uma arquitetura **Retrieval-Augmented Generation**.
