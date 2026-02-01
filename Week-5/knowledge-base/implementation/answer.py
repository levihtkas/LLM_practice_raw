from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage,HumanMessage,convert_to_messages
from dotenv import load_dotenv


load_dotenv(override=True)

model = "gpt-4.1-nano"
db_name = "vector_db"

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
reterival_k=5

SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
If relevant, use the given context to answer any question.
If you don't know the answer, say so.
Context:
{context}
"""

vectorstore = Chroma(persist_directory=db_name,embedding_function=embeddings)

reteriver = vectorstore.as_retriever()

llm= ChatOpenAI(model_name=model)

def fetch_document(question):
  return reteriver.invoke(question,k=reterival_k)

def combined_question(question,history):
  prior = "\n".join(m['content'] for m in history if m['role'] =="user")
  return prior + '\n' +question

def answer_question(question,history):
  combined = combined_question(question,history)
  docs = fetch_document(combined)
  context= "\n\n".join(doc.page_content for doc in docs)
  system_prompt = SYSTEM_PROMPT.format(context=context)
  messages = [SystemMessage(content=system_prompt)]
  messages.extend(convert_to_messages(history))
  messages.append(HumanMessage(content=question))
  response = llm.invoke(messages)
  return response.content,docs

