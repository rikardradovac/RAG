# RAG

This repository serves as the home for the RAG project for the course DAT450. It contains code for intializing and evaluating a question and answer chatbot, using langchain and Pinecone, with scraped data from <a href="https://quotes.toscrape.com">quotes.toscrape.com</a>.

## Installation

Install this repo:

```bash
git clone https://github.com/username/RAG.git
cd RAG
pip install -r requirements.txt
```

## Evaluation

[See run_eval.ipynb for a full example](run_eval.ipynb)

```bash
python3 -m rag.eval
```