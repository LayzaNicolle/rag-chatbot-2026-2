Aqui está o README.md atualizado, com os nomes da equipe destacados logo no topo do documento, 100% em texto e markdown puro, exatamente de acordo com os requisitos solicitados:

Chatbot Simples com RAG & LangGraph

Equipe:

Estevão Chagas
Layza Nicolle 
Matheus Pablo
Vinicius Simas

Descrição do Projeto:
Aplicação de chatbot web desenvolvida em Python para responder a perguntas em linguagem natural a partir de uma base de conhecimento própria, utilizando a arquitetura RAG (Retrieval-Augmented Generation) e orquestração do fluxo com LangGraph.

Requisitos Atendidos:
Interface Web de Chatbot: Campo para digitação da pergunta, envio de mensagem, exibição da resposta do sistema e comunicação com o back-end.

Back-end em Python: Integração da busca por similaridade vetorial com a API da LLM externa.

Orquestração com LangGraph: Fluxo principal do RAG organizado em nós (recebimento da pergunta, recuperação de contexto, montagem do prompt, chamada da LLM e retorno da resposta).

Base de Conhecimento Própria: Volume significativo de dados em domínio específico (PDFs/TXTs) indexados via embeddings.

Uso de LLM Externa: Utilizada estritamente na etapa de geração da resposta final.

Base de Conhecimento
Domínio Escolhido: [Inserir o domínio aqui. Ex: Documentação Técnica / Regulamento Acadêmico / Legislação]

Fontes dos Dados: Arquivos armazenados no diretório data/.

Pipeline da Aplicação
Coleta e organização do conteúdo do domínio escolhido no diretório data/.

Quebra do conteúdo em chunks e geração de embeddings.

Armazenamento dos embeddings em um índice vetorial local.

Recebimento da pergunta do usuário através da interface web.

Recuperação dos chunks mais relevantes no índice vetorial pelo LangGraph.

Montagem do contexto unindo os chunks buscados e o histórico de mensagens.

Chamada da LLM externa para geração da resposta fundamentada no contexto.

Exibição da resposta final no chat da interface web.

Como Executar o Projeto
Pré-requisitos
Python 3.10 ou superior

Chave de API de uma LLM externa (OpenAI, Gemini, Groq, etc.)

Passo a Passo
Clonar o repositório:

Bash
git clone https://github.com/SEU-USUARIO/SEU-REPOSITORIO.git
cd SEU-REPOSITORIO
Criar e ativar o ambiente virtual:

Bash
python -m venv venv
# No Linux/Mac:
source venv/bin/activate
# No Windows:
venv\Scripts\activate
Instalar as dependências:

Bash
pip install -r requirements.txt
Configurar as variáveis de ambiente:
Crie um arquivo .env na raiz do projeto com sua chave de API:

Snippet de código
OPENAI_API_KEY=sua_chave_aqui
Gerar a base vetorial:

Bash
python src/ingest.py
Executar a interface do chatbot:

Bash
streamlit run src/app.py
Estrutura do Repositório
Plaintext
├── data/              # Arquivos da base de conhecimento (PDFs/TXTs)
├── vectorstore/       # Armazenamento do índice vetorial
├── src/
│   ├── ingest.py      # Script de limpeza, chunking e embeddings
│   ├── graph.py       # Fluxo de RAG orquestrado com LangGraph
│   └── app.py         # Back-end Python e interface web do chatbot
├── requirements.txt   # Dependências do projeto
└── README.md          # Instruções de execução e documentação