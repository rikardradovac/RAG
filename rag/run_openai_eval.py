from .eval import calculate_f1_scores, exact_match_score, similarity_search
from openai import OpenAI
from .config import OPENAI_KEY
from .prompts import prompt_template_openai
import pandas as pd
import time

client = OpenAI(api_key=OPENAI_KEY)

data_files = ["data/rag_data.csv", "data/rag_data2.csv"]


for path in data_files:
    data = pd.read_csv(path)

    
    results = []
    for iteration, item in data.iterrows():
        question, answer = item[0], item[1]
        response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt_template_openai.format(question=question)},
        ]
        )
        result = response.choices[0].message.content
        results.append({"result": result, "answer": answer, "question": question})
        
        if iteration % 10 == 0:
            # sleep for 5 seconds to avoid rate limit
            time.sleep(5)

    predictions = [res["result"] for res in results]
    ground_truths = [res["answer"] for res in results]

    exact_matches = exact_match_score(predictions, ground_truths)
    f1_scores = calculate_f1_scores(predictions, ground_truths)
    similarity_scores = similarity_search(predictions, ground_truths)
    
    filename = path.split("/")[-1]
    pd.DataFrame({"exact_matches": [exact_matches], "f1_scores": [f1_scores], "similarity_scores": [similarity_scores]}).to_csv(f"{filename}_openai_predictions.csv")
        
