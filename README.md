# Text2SQL-Agent

Text2SQL-Agent is a chatbot that answers questions from SQL databases, PDF documents, and the web. You ask a question in plain English, and an agent built with LangGraph chooses the right tool, runs it, and replies. It uses Llama 3.3 70B through Hugging Face by default and has a Gradio chat interface.

## Features
- **Natural language to SQL**: Turns questions into SQL and runs them on two SQLite databases. The databases are opened read-only, so generated SQL can't change the data.
- **Document search (RAG)**: Finds relevant passages in PDF documents using Chroma vector databases.
- **Web search**: Gets up-to-date results through the Tavily API.
- **Conversation memory**: Each browser session has its own conversation. The Clear button starts a new one.
- **Chat logs**: Saves every question and answer to daily CSV files.

## Tools

| Tool | Data source | Example question |
|---|---|---|
| `query_travel_sqldb` | Swiss Airline flights and bookings (`data/travel.sqlite`) | How many aircraft are in the database? |
| `query_chinook_sqldb` | Digital music store (`data/Chinook.db`) | Which artists have the most albums? |
| `lookup_swiss_airline_policy` | Swiss Airlines FAQ (PDF) | Can I cancel my ticket? |
| `lookup_novel` | *The Adventures of Huckleberry Finn* (PDF) | Who is Jim in the novel? |
| `tavily_search_results_json` | The web | What's the latest airline news? |

## Project Structure
- `src/app.py` — Gradio app and entry point
- `src/agent_graph/` — Agent graph, tools, and LLM setup
- `src/chatbot/` — Chat backend, project config, and chat logging
- `src/prepare_vector_db.py` — Builds the vector databases from the PDFs
- `configs/` — YAML settings for the tools and the project
- `data/` — SQLite databases, vector databases, and source PDFs
- `memory/` — Chat logs, created when the app runs

## Requirements
- Python 3.11 (the pinned dependencies were tested on this version)
- [Hugging Face token](https://huggingface.co/settings/tokens) for the LLM. The default model runs on Inference Providers, so the account needs inference credits.
- [Tavily API key](https://app.tavily.com/) for web search
- [LangChain API key](https://smith.langchain.com/) (optional) for LangSmith tracing

## Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/ankushsurana/RAG-SQL-Agent.git
   cd RAG-SQL-Agent
   ```
2. **Create and activate a virtual environment:**

   Windows (PowerShell):
   ```powershell
   py -3.11 -m venv venv
   .\venv\Scripts\Activate.ps1
   ```
   macOS or Linux:
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate
   ```
3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Add your API keys:**

   Copy `.env.example` to `.env`, then fill in the values.

   Windows (PowerShell):
   ```powershell
   Copy-Item .env.example .env
   ```
   macOS or Linux:
   ```bash
   cp .env.example .env
   ```
5. **Add the travel database:**

   `data/travel.sqlite` isn't in the repository because of its size. Download the [Swiss Airline sample database](https://storage.googleapis.com/benchmarks-artifacts/travel-db/travel2.sqlite) from the LangGraph tutorial data and save it as `data/travel.sqlite`. Without it, the travel tool returns an error and the other tools keep working.
6. **Build the vector databases:**
   ```bash
   python src/prepare_vector_db.py
   ```
   The vector databases aren't in the repository either. This script downloads the embedding model the first time, and it skips any database that is already built.

## Running the App
```bash
python src/app.py
```
Then open `http://127.0.0.1:7860`.

Type a question, then press **Shift+Enter** or click **Send**. Plain Enter adds a new line.

Conversations are kept in memory, so they are lost when the app stops. The CSV logs in `memory/` stay on disk.

## Customization
- **LLM**: Change the `llm` values in `configs/tools_config.yml`. `primary_agent` chooses the tools, and the SQL sections write the queries.
- **History length**: `max_history_messages` in `configs/tools_config.yml` limits how many recent messages are sent to the LLM. A lower value uses fewer tokens.
- **Embedding model or chunk size**: Change them in `configs/tools_config.yml`, delete that tool's vector database folder in `data/`, and run `python src/prepare_vector_db.py` again. Queries must use the same embedding model as the database.
- **New tools**: Add the tool in `src/agent_graph/`, then add it to the `tools` list in `src/agent_graph/build_full_graph.py`.
- **Public link**: `demo.launch(share=True)` in `src/app.py` creates a public URL. The app has no login, so anyone with the link can use your API credits.

## Troubleshooting

| Problem | Cause and fix |
|---|---|
| The chat shows `402 Payment Required` | The Hugging Face account has no inference credits. Add credits in your [billing settings](https://huggingface.co/settings/billing). |
| The app stops at startup with a Tavily API key error | `TAVILY_API_KEY` is missing from `.env`. |
| The travel tool reports a database error | `data/travel.sqlite` is missing. See step 5 of the setup. |
| `ImportError` when the app starts | The installed packages don't match `requirements.txt`. Install them again in a new virtual environment. |
