# Chatbot RAG — Turismo no Brasil

Chatbot web desenvolvido em **Python** para responder perguntas em linguagem natural sobre **turismo no Brasil**, utilizando **Retrieval-Augmented Generation (RAG)**, busca vetorial e **LangGraph** para orquestração do fluxo.

---

## Equipe

* Estevão Chagas
* Layza Nicolle
* Matheus Pablo
* Vinicius Simas

---

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

---

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

---

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

---

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

---

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

---

## Principais componentes

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

---

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

---

# Instalação

## Pré-requisitos

Antes de executar o projeto, certifique-se de possuir:

* Python **3.10+**
* `pip`
* Git
* Chave de API da LLM

Verifique as versões instaladas:

```bash
python --version
pip --version
git --version
```

---

## Clonar o projeto

```bash
git clone URL_DO_REPOSITORIO
cd rag-chatbot
```

Caso o projeto tenha sido baixado como `.zip`, basta extrair os arquivos e abrir o terminal dentro da pasta do projeto.

---

## Criar o ambiente virtual

### Windows

```bash
python -m venv venv
```

Ative o ambiente:

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
```

Ative o ambiente:

```bash
source venv/bin/activate
```

---

## Instalar as dependências

Com o ambiente virtual ativado:

```bash
pip install -r requirements.txt
```

---

# Configuração da API

Na raiz do projeto, crie um arquivo chamado:

```text
.env
```

Adicione:

```env
LLM_PROVIDER=groq
GROQ_API_KEY=sua_chave_aqui
GROQ_MODEL=openai/gpt-oss-20b
```

Substitua `sua_chave_aqui` pela sua chave de API.

### Configurações opcionais

Também podem ser configurados:

```env
EMBEDDING_MODEL_NAME=paraphrase-multilingual-MiniLM-L12-v2
CHUNK_SIZE=800
CHUNK_OVERLAP=120
TOP_K=4
MAX_HISTORY_TURNS=4
```

---

# Executando o projeto

## 1. Processar a base de conhecimento

Antes de iniciar a API, execute:

```bash
python backend/ingest.py
```

Esse processo irá carregar os documentos da pasta `knowledge_base/`, gerar os embeddings e criar o índice vetorial utilizado pelo RAG.

---

## 2. Iniciar a API

Depois do processamento:

```bash
uvicorn backend.main:app --reload
```

A API ficará disponível em:

```text
http://127.0.0.1:8000
```

---

## 3. Acessar o chatbot

Abra no navegador:

```text
http://127.0.0.1:8000
```

---

# Execução rápida

Depois que o projeto estiver configurado, os comandos principais são:

### Windows

```bash
venv\Scripts\activate
python backend/ingest.py
uvicorn backend.main:app --reload
```

Depois acesse:

```text
http://127.0.0.1:8000
```

---

# Endpoints

| Método | Endpoint  | Descrição                |
| ------ | --------- | ------------------------ |
| `GET`  | `/`       | Interface do chatbot     |
| `POST` | `/chat`   | Envia uma pergunta       |
| `GET`  | `/health` | Verifica o status da API |
| `GET`  | `/docs`   | Documentação da API      |

A documentação interativa do FastAPI pode ser acessada em:

```text
http://127.0.0.1:8000/docs
```

---

# Exemplo de requisição

```json
{
  "message": "Quais são os principais pontos turísticos de Recife?",
  "session_id": "123"
}
```

---

# Exemplo de resposta

```json
{
  "session_id": "123",
  "answer": "Resposta gerada com base na base de conhecimento.",
  "sources": []
}
```

---

# Exemplos de prompts

O chatbot foi desenvolvido para responder perguntas relacionadas ao conteúdo da base de conhecimento sobre turismo no Brasil.

## Perguntas sobre destinos

```text
Quais são os principais pontos turísticos de Recife?
```

```text
O que posso conhecer em Foz do Iguaçu?
```

```text
Quais são os principais pontos turísticos de Salvador?
```

```text
O que um turista pode fazer em Fernando de Noronha?
```

---

## Comparações

```text
Compare Recife e Salvador como destinos turísticos.
```

```text
Compare Gramado e Florianópolis.
```

```text
Qual a diferença entre o turismo na Amazônia e no Pantanal?
```

---

## Cultura e eventos

```text
O que é o Carnaval brasileiro?
```

```text
Quais são as características das Festas Juninas no Brasil?
```

```text
Quais destinos brasileiros possuem destaque cultural?
```

---

## Transporte

```text
Quais são as principais formas de transporte utilizadas pelos turistas no Brasil?
```

```text
Quais informações sobre transporte estão disponíveis na base de conhecimento?
```

---

## Segurança

```text
Quais cuidados de segurança são recomendados para turistas no Brasil?
```

```text
Quais informações sobre segurança estão disponíveis na base de conhecimento?
```

---

## Recife e Olinda

```text
Quais são os principais pontos turísticos de Recife e Olinda?
```

```text
O que um turista pode fazer em Recife?
```

```text
Quais atrações históricas existem em Olinda?
```

---

## Testando o RAG

Para demonstrar o funcionamento da arquitetura RAG:

```text
Com base na base de conhecimento, quais destinos brasileiros são citados como opções de ecoturismo?
```

```text
Segundo os documentos disponíveis, quais informações existem sobre Fernando de Noronha?
```

```text
Quais informações sobre turismo no Brasil estão presentes na base de conhecimento?
```

---

# Histórico de conversa

O chatbot mantém o histórico das mensagens utilizando um `session_id`.

Isso permite que perguntas de continuidade sejam consideradas dentro da mesma conversa.

Exemplo:

```text
Quais são os principais pontos turísticos de Recife?
```

Depois:

```text
E quais deles são históricos?
```

E:

```text
Qual seria um roteiro para conhecer esses lugares?
```

A quantidade de turnos armazenados pode ser configurada através de:

```env
MAX_HISTORY_TURNS=4
```

---

# Configurações principais

As principais configurações podem ser ajustadas através do arquivo `.env`:

```env
EMBEDDING_MODEL_NAME=paraphrase-multilingual-MiniLM-L12-v2
CHUNK_SIZE=800
CHUNK_OVERLAP=120
TOP_K=4
MAX_HISTORY_TURNS=4
```

### `EMBEDDING_MODEL_NAME`

Define o modelo utilizado para gerar os embeddings.

### `CHUNK_SIZE`

Define o tamanho dos trechos nos quais os documentos serão divididos.

### `CHUNK_OVERLAP`

Define a sobreposição entre os chunks.

### `TOP_K`

Define a quantidade de trechos recuperados pelo mecanismo de busca vetorial.

### `MAX_HISTORY_TURNS`

Define a quantidade de turnos da conversa mantidos no histórico.

---

# Considerações

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

A aplicação utiliza a base de conhecimento como fonte para recuperação das informações antes da geração das respostas, demonstrando na prática o funcionamento de uma arquitetura **Retrieval-Augmented Generation (RAG)**.
