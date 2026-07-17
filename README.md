# UNYFY: AI Multi-Agent E-commerce Discovery Platform

An enterprise-grade, asynchronous AI platform built to autonomously scrape, extract, and semantically index dynamic e-commerce product offers. 

Powered by a **LangGraph Agent Swarm**, **FastAPI**, **Google Gemini 1.5**, and **ChromaDB**, this system bypasses brittle traditional scraping (XPath/CSS selectors) by using LLM-driven semantic text extraction. It is fully resilient to sudden website layout changes.

---

## 🚀 Key Features

*   **Multi-Agent State Machine:** Utilizes LangGraph to orchestrate a resilient execution loop with dedicated Browser, Extractor, and Validator agents.
*   **Semantic Extraction:** Converts dynamic DOM structures to Markdown, utilizing Google Gemini 1.5 with strict Pydantic structural tool binding to guarantee perfectly formatted JSON outputs.
*   **Vector DB Integration:** Automatically embeds extracted data into ChromaDB using `text-embedding-004` for high-dimensional semantic search.
*   **Hybrid RAG Search:** Combines Natural Language vector matching (e.g., "gaming laptops under 70k") with strict scalar metadata filtering (price bounds, platform names).
*   **Non-Blocking API Gateway:** Built on FastAPI with asynchronous background task management, ready for Kafka integration.

---

## 🧠 The Agent Swarm Architecture

1.  🌐 **Browser Agent:** Launches headless Chromium via Playwright, waits for network idleness (to capture dynamic JS offers), and condenses HTML into token-efficient Markdown.
2.  🧠 **Extractor Agent:** Consumes the Markdown and uses zero-shot reasoning via Gemini to populate strict Pydantic data schemas.
3.  🛡️ **Validator Agent:** Assesses data integrity, ensures price logic, generates a SHA-256 deduplication hash, and commits the final state to the Vector DB.

---

## ⚙️ Installation & Setup

### 1. Prerequisites
Ensure you have **Python 3.10+** installed on your machine.

### 2. Clone and Install Dependencies
```bash
# Clone the repository
git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
cd YOUR_REPOSITORY_NAME

# Create and activate a virtual environment (Windows)
python -m venv venv
.\venv\Scripts\activate

# Install required packages
pip install fastapi uvicorn pydantic playwright html2text langchain-google-genai langchain-core python-dotenv chromadb

# Install the Playwright Chromium browser binary
playwright install chromium