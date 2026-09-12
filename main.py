from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vectors import retriever

model = OllamaLLM(model="llama3.2")

template = """
You are an exeprt in answering questions about a pizza restaurant

Here are some relevant information: {information}

Here is the question to answer: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    print("\n\n-------------------------------")
    question = input("Ask your question (n to quit): ")
    print("\n\n")
    if question == "n":
        break
    
    context = retriever.invoke(question)
    result = chain.invoke({"information": context, "question": question})
    print(result)