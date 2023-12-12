

prompt_template_llama = """System prompt: You answer questions in a short, concise and informative way. I ask you a question and you answer it. You only answer the question without repeating the question. The answer is only based on the following context: {context}.
question: {question}
"""


prompt_template_mistral = """<s>[INST]System prompt: You answer questions in a short, concise and informative way. I ask you a question and you answer it. You only answer the question without repeating the question. The answer is only based on the following context: {context}.
question: {question} [/INST]
"""