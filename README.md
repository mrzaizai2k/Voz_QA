# VOZ Thread QA

An AI-powered Question Answering system for VOZ forum threads.

Instead of simply summarizing a thread, the system:

- Crawls VOZ threads asynchronously
- Extracts forum posts
- Caches thread data locally
- Builds structured context
- Detects user intent automatically
- Routes questions to specialized prompts
- Uses an LLM to answer questions based on the discussion

Examples:

- "Tóm tắt thread này"
- "Mọi người nghĩ gì về VinFast?"
- "Liệt kê các comment nói về mua nhà"
- "Những lời khuyên nào được chia sẻ nhiều nhất?"
- "Vì sao nhiều người nghỉ việc?"
- "Ý kiến nào được đồng thuận cao nhất?"

---

# Table of Contents

- [Architecture](#architecture)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Environment Variables](#environment-variables)
- [Crawler](#crawler)
- [Running the System](#running-the-system)
- [Running Docker](#running-docker)

---

# Architecture

```text
VOZ Thread URL
        │
        ▼
 AsyncThreadCrawler
        │
        ▼
 Thread Cache (.json)
        │
        ▼
 Posts DataFrame
        │
        ▼
 Context Builder
        │
        ▼
 Intent Router
        │
        ▼
 Specialized Prompt
        │
        ▼
 LLM
        │
        ▼
 Final Answer
```

---

# Features

## Thread Crawling

- Async crawling
- Concurrent page fetching
- Automatic pagination detection
- Retry logic
- Rate limiting
- Local caching

## Knowledge Extraction

- Noise filtering
- Meme filtering
- Spam filtering
- Experience extraction
- Consensus detection
- Opinion clustering

## Question Answering

Supports much more than summarization.

Examples:

```text
Summarize this thread

What are people saying about VinFast?

List comments discussing real estate.

Why are people quitting their jobs?

What advice appears most frequently?

Which opinion has the strongest consensus?
```

---

# Project Structure

```text
project/

├── cache/
│   └── *.json

├── output/
│   └── *.csv

├── src/
│
├── crawler.py
├── prompts.py
├── router.py
├── qa.py
│
├── requirements.txt
└── README.md
```

---

# Installation

## 1. Create Conda Environment

```bash
conda create -n vozqa python=3.10 -y
```

Activate:

```bash
conda activate vozqa
```

---

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Environment Variables

Create a `.env` file.

## OpenAI

```env
OPENAI_API_KEY=your_key_here
```

## Anthropic

```env
ANTHROPIC_API_KEY=your_key_here
```

---

# Crawler

The crawler is implemented in:

```text
src/crawler.py
```

Main class:

```python
AsyncThreadCrawler
```


# Intent Router

Instead of forcing every question into a summary format,
the system first detects user intent.

Flow:

```text
Question
    │
    ▼
Intent Detection
    │
    ▼
Specialized Prompt
    │
    ▼
Answer
```

---

## Router Output

Example:

```json
{
  "intent": "opinions",
  "reason": "User asks what people think."
}
```

---

# Running the System

```bash
    make run
```

# Running Docker

```bash
docker build -t voz-thread-qa .

docker run -d \
  --name voz-thread-qa \
  -p 8501:8501 \
  --env-file .env \
  -v $(pwd)/src:/app/src \
  -v $(pwd)/cache:/app/cache \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/data:/app/data \
  voz-thread-qa
```