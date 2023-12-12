import pandas as pd
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from sentence_transformers import SentenceTransformer
from torch import cuda, bfloat16
from pinecone_embedder import PineconeEmbedder
import transformers
from langchain.llms import HuggingFacePipeline
from collections import Counter
import string
import re
import numpy as np
import torch.nn.functional as F
import evaluate
from .config import HF_AUTH
import logging


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def handle_punc(text):
        exclude = set(string.punctuation + "".join(["‘", "’", "´", "`"]))
        return "".join(ch if ch not in exclude else " " for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace("_", " ")

    return white_space_fix(handle_punc(lower(replace_underscore(s)))).strip()


def exact_match_score(predictions, ground_truths):
    exact_match = evaluate.load("exact_match")
    return exact_match.compute(references=ground_truths, predictions=predictions, ignore_case=True, ignore_punctuation=True)["exact_match"]


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def calculate_f1_scores(predictions, ground_truths):
    scores = []
    for index in range(len(predictions)):
        result = f1_score(predictions[index], ground_truths[index])
        scores.append(result)

    return np.mean(scores)


def similarity_search(predictions, ground_truths, model_path="jamesgpt1/sf_model_e5"):
    model = SentenceTransformer(model_path)
    predicted_embeddings = model.encode(predictions, convert_to_tensor=True)
    ground_truths = model.encode(ground_truths, convert_to_tensor=True)
    scores = F.cosine_similarity(predicted_embeddings, ground_truths)
    return float(scores.mean())


def load_llm(model_path):
    device = f"cuda:{cuda.current_device()}" if cuda.is_available() else "cpu"

    # set quantization configuration to load large model with less GPU memory
    # this requires the `bitsandbytes` library
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=bfloat16,
    )

    # begin initializing HF items, need auth token for these
    
    model_config = transformers.AutoConfig.from_pretrained(
        model_path, use_auth_token=HF_AUTH
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        config=model_config,
        quantization_config=bnb_config,
        device_map="auto",
        use_auth_token=HF_AUTH,
    )
    model.eval()

    print(f"Model loaded on {device}")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path, use_auth_token=HF_AUTH
    )

    generate_text = transformers.pipeline(
        model=model,
        tokenizer=tokenizer,
        return_full_text=True,  # langchain expects the full text
        task="text-generation",
        temperature=0.0,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
        max_new_tokens=512,  # mex number of tokens to generate in the output
        repetition_penalty=1.1,  # without this output begins repeating
    )

    model = HuggingFacePipeline(pipeline=generate_text)
    return model


def run_eval(pinecone: Pinecone, sentence_emb_name: str, gpt_name: str, data_path: str):
    pinecone.load_model(sentence_emb_name)
    llm = load_llm(gpt_name)
    pinecone.create_vectorstore(namespace=sentence_emb_name)
    retriever = pinecone.get_retriever()

    rag_pipeline = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
    )
    results = []
    data = pd.read_csv(data_path)[:10]
    for _, item in data.iterrows():
        query, answer = item[0], item[1]
        print(query)
        result = rag_pipeline(query)
        result["answer"] = answer
        results.append(result)

    logging.info("Sucessfully ran RAG pipeline, now evaluating results")
    predictions = [res["result"] for res in results]
    ground_truths = [res["answer"] for res in results]
    
    exact_matches = exact_match_score(predictions, ground_truths)
    f1_scores = calculate_f1_scores(predictions, ground_truths)
    similarity_scores = similarity_search(predictions, ground_truths)
    
    return {"exact_matches": [exact_matches], "f1_scores": [f1_scores], "similarity_scores": [similarity_scores]}


if __name__ == "__main__":
    sentence_embedding_models = ["paraphrase-distilroberta-base-v1", "intfloat/e5-base"]
    llms = ["meta-llama/Llama-2-13b-chat-hf", "mistralai/Mistral-7B-Instruct-v0.2"]
    data_paths = ["data/rag_data.csv", "data/rag_data2.csv"]
    pinecone_embedder = PineconeEmbedder("rag")
    
    for sentence_emb_name in sentence_embedding_models:
        for gpt_name in llms:
            for data_path in data_paths:
                logging.info(f"Running evaluation for {sentence_emb_name} and {gpt_name} on {data_path}")
                result = run_eval(pinecone=pinecone_embedder, sentence_emb_name=sentence_emb_name, gpt_name=gpt_name, data_path=data_path)
                logging.info("Saving results")
                pd.DataFrame(result).to_csv(f"results/{sentence_emb_name}_{gpt_name}_{data_path}.csv")
    