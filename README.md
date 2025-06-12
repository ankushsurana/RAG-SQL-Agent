# Text2SQL-Agent

Text2SQL-Agent is a modular, production-ready, and extensible agent framework for natural language to SQL, RAG (Retrieval-Augmented Generation), and web search workflows. It leverages state-of-the-art open-source LLMs (such as Llama-3, deepseek-R1) and integrates with vector databases, SQL databases, and web search APIs to provide a unified conversational AI experience for enterprise and research use cases.

## Features
- **Conversational Agent**: Chatbot interface powered by LLMs for natural language queries.
- **SQL Agent**: Converts user questions into SQL queries and executes them on your database.
- **RAG Tool**: Retrieves relevant documents from a vector database and augments LLM responses.
- **Web Search Tool**: Integrates with Tavily API for up-to-date web search results.
- **Graph-based Orchestration**: Uses LangGraph for flexible, stateful multi-tool workflows.
- **Memory & History**: Saves chat history for each session.
- **Extensible**: Easily add new tools, databases, or LLMs.

## Project Structure
- `src/` — Main application code (agent graph, chatbot backend, tool integrations)
- `data/` — SQLite databases, vector stores, and unstructured documents
- `configs/` — YAML configuration files for tools and project settings
- `memory/` — Chat history logs

## Requirements
- Python 3.9+
- [Hugging Face Token](https://huggingface.co/settings/tokens) (for LLMs)
- [Tavily API Key](https://app.tavily.com/) (for web search)
- [LangChain API Key](https://smith.langchain.com/) (for tracing)

Install dependencies:
```bash
pip install -r requirements.txt
```

## Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/Text2SQL-Agent.git
   cd Text2SQL-Agent
   ```
2. **Configure environment variables:**
   - Copy `.env.example` to `.env` and fill in your API keys:
     ```bash
     cp .env.example .env
     # Edit .env and set HF_TOKEN, TAVILY_API_KEY, LANGCHAIN_API_KEY
     ```
3. **Prepare vector databases:**
   ```bash
   python src/prepare_vector_db.py
   ```
4. **Run the application:**
   ```bash
   python src/app.py
   ```
   - The Gradio app will be available at `http://127.0.0.1:7860` (add `share=True` in `app.py` to get a public link).

## Usage
- Use the chatbot UI to ask questions about your data, company policies, or the web.
- The agent will automatically select the best tool (SQL, RAG, or search) to answer your query.
- All chat history is saved in the `memory/` folder for each session.

## Customization
- Add new tools or modify existing ones in `src/agent_graph/`.
- Update configuration in `configs/tools_config.yml` and `configs/project_config.yml`.
- Add new databases or documents in the `data/` directory.
