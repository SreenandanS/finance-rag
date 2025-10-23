
# Real-Time Financial News RAG Agent Baseline Built On Pathway

**Participant:** Sreenandan Shashidharan

**Problem Statement:** Pathway - Financial Agents that work with Real-time Insights 

## 1. Problem Statement

The challenge is to build a "production grade, streaming native, agentic system"  for the finance world. The core requirement is to use the Pathway framework as the streaming data engine to create applications that react to data as it arrives, moving beyond traditional batch processing.

This solution must leverage streaming to enable features like "always up to date metrics" and "live indexes that refresh as documents and events change".

## 2. Solution

This project implements the foundational backend for a real-time financial insights agent. It is a **streaming-native RAG (Retrieval-Augmented Generation) pipeline** built entirely on the **Pathway framework**.

Our application (`app.py`) continuously monitors a stream of financial news articles (simulated via `feed.jsonl`). As new articles arrive, the pipeline automatically:
1.  **Ingests** the streaming data (`headline`, `body`).
2.  **Processes** the text by splitting it into manageable chunks (`TokenCountSplitter`).
3.  **Embeds** each chunk into a vector representation using a local `SentenceTransformer` model.
4.  **Indexes** these vectors in a real-time `DocumentStore`.

This `DocumentStore` is then exposed via a `DocumentStoreServer`, providing a live REST API. Any LLM or financial agent can query this API to retrieve the most up-to-the-second relevant news to make informed decisions, directly addressing the need for "live indexing for retrieval".

## 3. Architecture & Pathway Usage

The architecture is simple, modular, and built entirely on Pathway's core components, as required by the "Problem Statement".

 -> [TokenCountSplitter] -> [SentenceTransformerEmbedder] -> [DocumentStore] -> [DocumentStoreServer (REST API)]]

**How Pathway is Used:**

* **Streaming Data Ingest:** Uses `pw.io.jsonlines.read` in `"streaming"` mode to read `feed.jsonl`. The `autocommit_duration_ms=500` ensures new data is picked up almost instantly (within 500ms).
* **Real-time Transformations:** The application uses Pathway's functional API (`.select()`, `.flatten()`) to transform the raw data into indexed chunks.
* **Live Indexing (LLM xPack):** Uses `pathway.xpacks.llm.embedders.SentenceTransformerEmbedder` and `pathway.xpacks.llm.document_store.DocumentStore`. The `DocumentStore` is connected directly to the streaming input, so its index is **incrementally updated** as new articles arrive, requiring no manual re-indexing.
* **API Server (LLM xPack):** Uses `pathway.xpacks.llm.servers.DocumentStoreServer` to instantly expose our live index via a standard REST API. This aligns with the RAG templates provided in the developer resources.

## 4. Installation & Running the Demo

This application is containerized using the provided `Dockerfile` for easy, one-command execution.

### Prerequisites

* [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running.
* A PowerShell terminal.

### Step 1: Build the Docker Image

Open a PowerShell terminal in the project's root directory (where the `Dockerfile` is) and run:

```powershell
docker build -t pathway-financial-agent .
```
### Step 2: Run the Docker Container
Run the following command to start the server. We use -v to mount the local feed.jsonl file into the container, allowing us to modify it from our host machine to simulate a live data stream.

```PowerShell

docker run -p 8000:8000 -v ${PWD}/feed.jsonl:/app/feed.jsonl --rm --name pathway-app pathway-financial-agent
```
Leave this terminal running. You will see logs from the Pathway server, indicating it's running on 0.0.0.0:8000.

### Step 3: Run the Demonstration (Live Demo)
Open a new, separate PowerShell terminal. You will use this to query the server and simulate new data.

See the PATHWAY OUTPUT SCRIPT.pdf for the step-by-step commands to run in this new terminal.
