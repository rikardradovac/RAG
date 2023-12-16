import pandas as pd
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer
from torch import cuda, bfloat16
from .pinecone_embedder import PineconeEmbedder
import transformers
from langchain.llms import HuggingFacePipeline
from collections import Counter
import string
import numpy as np
import torch.nn.functional as F
import evaluate
from .config import HF_AUTH
import logging
import os
from .prompts import prompt_template_llama, prompt_template_mistral
from typing import List

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def clean_answer(answer: str):
    """Cleans the answer for evaluation"""

    def white_space_fix(text):
        return " ".join(text.split())

    def handle_punc(text):
        exclude = set(string.punctuation + "".join(["‘", "’", "´", "`"]))
        return "".join(ch if ch not in exclude else " " for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace("_", " ")

    return white_space_fix(handle_punc(lower(replace_underscore(answer)))).strip()


def exact_match_score(predictions: List[str], ground_truths: List[str]):
    """Computes exact match score

    Args:
        predictions (List[str]): List of outputs
        ground_truths (List[str]): List of answers

    Returns:
        float: mean exact match score
    """
    exact_match = evaluate.load("exact_match")
    return exact_match.compute(
        references=ground_truths,
        predictions=predictions,
        ignore_case=True,
        ignore_punctuation=True,
        regexes_to_ignore="\n:",
    )["exact_match"]


def f1_score(prediction: str, ground_truth: str):
    """Computes f1 score

    Args:
        prediction (str): output
        ground_truth (str): answer

    Returns:
        float: mean f1 score
    """
    prediction_tokens = clean_answer(prediction).split()
    ground_truth_tokens = clean_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def calculate_f1_scores(predictions: List[str], ground_truths: List[str]):
    """Computes f1 score

    Args:
        predictions (List[str]): List of outputs
        ground_truths (List[str]): List of answers

    Returns:
        float: mean f1 score
    """
    scores = []
    for index in range(len(predictions)):
        result = f1_score(predictions[index], ground_truths[index])
        scores.append(result)

    return np.mean(scores)


def similarity_search(
    predictions: List[str],
    ground_truths: List[str],
    model_path: str = "jamesgpt1/sf_model_e5",
):
    """Computes similarity score

    Args:
        predictions (List[str]): List of outputs
        ground_truths (List[str]): List of answers
        model_path (str, optional): Embedding model to use. Defaults to "jamesgpt1/sf_model_e5".

    Returns:
        float: mean cosine similarity score
    """
    model = SentenceTransformer(model_path)
    predicted_embeddings = model.encode(predictions, convert_to_tensor=True)
    ground_truths = model.encode(ground_truths, convert_to_tensor=True)
    scores = F.cosine_similarity(predicted_embeddings, ground_truths)
    return float(scores.mean())


def load_llm(model_path: str):
    """Loads a large language model in 4bit quantization

    Args:
        model_path (str): huggingface model path

    Returns:
        HuggingFacePipeline: llm pipeline
    """
    device = f"cuda:{cuda.current_device()}" if cuda.is_available() else "cpu"

    # set quantization configuration to load large model with less GPU memory
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=bfloat16,
    )

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

    logger.info(f"Model loaded on {device}")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path, use_auth_token=HF_AUTH
    )

    generate_text = transformers.pipeline(
        model=model,
        tokenizer=tokenizer,
        return_full_text=True, 
        task="text-generation",
        temperature=1.0,  
        max_new_tokens=512, 
        repetition_penalty=1.1, # reduces repetiton
    )

    model = HuggingFacePipeline(pipeline=generate_text)
    return model


def run_eval(
    pinecone: Pinecone,
    sentence_emb_name: str,
    gpt_name: str,
    data_path: str,
    prompt_template: str,
):
    """Runs evaluation

    Args:
        pinecone (Pinecone): Pinecone index
        sentence_emb_name (str): sentence embedding model path
        gpt_name (str): gpt model path
        data_path (str): path to csv data
        prompt_template (str): prompt to use

    Returns:
        dict: mean f1, exact match and similarity scores
    """
    pinecone.load_model(sentence_emb_name)
    llm = load_llm(gpt_name)
    pinecone.create_vectorstore(namespace=sentence_emb_name)
    retriever = pinecone.get_retriever()

    rag_pipeline = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"],
            ),
        },
    )
    results = []
    data = pd.read_csv(data_path)
    for _, item in data.iterrows():
        query, answer = item[0], item[1]
        # e5 is trained with prefix
        if "e5" in sentence_emb_name:
            query = f"query: {query}"
        result = rag_pipeline(query)
        result["answer"] = answer
        results.append(result)

    logger.info("Sucessfully ran RAG pipeline, now evaluating results")
    predictions = [res["result"] for res in results]
    ground_truths = [res["answer"] for res in results]

    exact_matches = exact_match_score(predictions, ground_truths)
    f1_scores = calculate_f1_scores(predictions, ground_truths)
    similarity_scores = similarity_search(predictions, ground_truths)

    return {
        "exact_matches": [exact_matches],
        "f1_scores": [f1_scores],
        "similarity_scores": [similarity_scores],
    }


if __name__ == "__main__":
    # change these to run different experiments
    sentence_embedding_models = ["paraphrase-distilroberta-base-v1", "intfloat/e5-base"]
    llms = [
        ("meta-llama/Llama-2-13b-chat-hf", prompt_template_llama),
        ("mistralai/Mistral-7B-Instruct-v0.2", prompt_template_mistral),
    ]
    data_paths = ["data/rag_data.csv", "data/rag_data2.csv"]
    pinecone_embedder = PineconeEmbedder("rag")

    for sentence_emb_name in sentence_embedding_models:
        for gpt_name, prompt_template in llms:
            for data_path in data_paths:
                logger.info(
                    f"Running evaluation for {sentence_emb_name} and {gpt_name} on {data_path}"
                )
                result = run_eval(
                    pinecone=pinecone_embedder,
                    sentence_emb_name=sentence_emb_name,
                    gpt_name=gpt_name,
                    data_path=data_path,
                    prompt_template=prompt_template,
                )
                logger.info("Saving results")
                save_emb_name = "".join(sentence_emb_name.split("/"))
                save_gpt_name = "".join(gpt_name.split("/"))
                save_data_path = data_path.split("/")[-1]
                os.makedirs("results", exist_ok=True)
                pd.DataFrame(result).to_csv(
                    f"results/{save_emb_name}_{save_gpt_name}_{save_data_path}.csv"
                )
