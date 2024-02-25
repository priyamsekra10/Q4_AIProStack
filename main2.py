from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Create a chat model
chat = ChatOpenAI()

# Create a prompt
prompt = ChatPromptTemplate(
    input_variables=["content"],
    messages=[
        HumanMessagePromptTemplate.from_template("{content}"),
    ],
)

chain = LLMChain(
    llm=chat,
    prompt=prompt,
)

# while True:
content = "Write an 4 line Instagram caption for an image of an envelope"

result = chain({"content": content})

print(result["text"])

