

prompt_template_llama = """[INST] <<SYS>>
Answer the question based on the context below. Keep the answer short and concise. Respond "Unsure about answer" if not sure about the answer. Do not provide any prefix, simply write out the answer. {context}.
<</SYS>>
question: {question}[/INST]
"""

prompt_template_mistral = """<s>[INST]Answer the question based on the context below. Keep the answer short and concise. Respond "Unsure about answer" if not sure about the answer. Do not provide any prefix, simply write out the answer. {context}.

question: {question}[/INST]

"""

prompt_template_openai = """System prompt: You answer questions in a short, concise and informative way. I ask you a question and you answer it. You only answer the question without repeating the question.
question: {question}
"""