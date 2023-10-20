import logging
import mimetypes
import os
import pickle
import sys

import openai  # or any library used to access the LLM model
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader, CSVLoader, UnstructuredWordDocumentLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts.prompt import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from telegram import InlineQueryResultArticle, InputTextMessageContent
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler
from telegram.ext import filters
from langchain.vectorstores import FAISS as BaseFAISS

from chatbot import Chatbot
from embedder import Embedder

# Initialize your OpenAI API key or any other authentication method
# OPENAI_API_KEY = os.getenv('OPEN_AI_KEY')
OPENAI_API_KEY = ""
openai.api_key = OPENAI_API_KEY
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
load_dotenv()
logger = logging.getLogger(__name__)
conversations = {}
faiss_index = None

bot = None
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# qa_template = """
#         You are a helpful AI assistant named Dobby. The user gives you a file its content is represented by the following pieces of context, use them to answer the question at the end.
#         If you don't know the answer, just say you don't know. Do NOT try to make up an answer.
#         If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context.
#         Use as much detail as possible when responding.
#
#         context: {context}
#         =========
#         question: {question}
#         ======
#         """
# QA_PROMPT = PromptTemplate(template=qa_template, input_variables=["context", "question"])
qa_template = """
        You are a helpful AI assistant named Dobby. The user gives you a file its content is represented by the following pieces of context, use them to answer the question at the end.
        If you don't know the answer, just say you don't know. Do NOT try to make up an answer.
        If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context.
        Use as much detail as possible when responding.

        
        """

QA_PROMPT = PromptTemplate(template=qa_template, input_variables=[])


class FAISS(BaseFAISS):
    def save(self, file_path):
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)


def get_loader(file_path):
    mime_type, _ = mimetypes.guess_type(file_path)

    if mime_type == 'application/pdf':
        return PyPDFLoader(file_path)
    elif mime_type == 'text/csv':
        return CSVLoader(file_path)
    elif mime_type in ['application/msword',
                       'application/vnd.openxmlformats-officedocument.wordprocessingml.document']:
        return UnstructuredWordDocumentLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {mime_type}")


def train_or_load_model(train, faiss_obj_path, file_path, index_name):
    global embeddings
    if train:
        loader = get_loader(file_path)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000,
                                                       chunk_overlap=400)
        pages = loader.load_and_split(text_splitter=text_splitter)
        # for i in pages:
        #     print("\n ______________")
        #     print(i.page_content)

        # Save pages to a text file
        # with open('output.txt', 'w', encoding='utf-8') as f:
        #     sys.stdout = f
        #     print(pages)
        #
        #     sys.stdout = sys.__stdout__

        if os.path.exists(faiss_obj_path):
            faiss_index = FAISS.load(faiss_obj_path)
            new_embeddings = faiss_index.from_documents(pages, embeddings)
            new_embeddings.save(faiss_obj_path)
        else:
            faiss_index = FAISS.from_documents(pages, embeddings)
            faiss_index.save(faiss_obj_path)

        return FAISS.load(faiss_obj_path)
    else:
        return FAISS.load(faiss_obj_path)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="I'm a bot, please talk to me!")


async def prompt_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global bot
    query = update.message.text
    if not query:
        return
    # results = []
    # response = openai.Completion.create(
    #     engine="text-davinci-003",  # Use the appropriate LLM engine
    #     prompt=query,
    #     max_tokens=50  # Adjust this as needed
    # )
    #
    # generated_text = response.choices[0].text

    # generated_text = bot.conversational_chat(query, update.effective_chat.id)

    generated_text = conversation_tracking(query, update.effective_chat.id)

    # results.append(
    #     InlineQueryResultArticle(
    #         id=query.upper(),
    #         title='Caps',
    #         input_message_content=InputTextMessageContent("Inline respo " + query)
    #     )
    # )
    await context.bot.send_message(chat_id=update.effective_chat.id, text=generated_text)


def generate_response_chat(message_list):
    global faiss_index
    if faiss_index:
        # Add extra text to the content of the last message
        last_message = message_list[-1]

        # Get the most similar documents to the last message
        try:
            docs = faiss_index.similarity_search(query=last_message["content"], k=2)

            updated_content = last_message["content"] + "\n\n"
            for doc in docs:
                updated_content += doc.page_content + "\n\n"
        except Exception as e:
            print(f"Error while fetching : {e}")
            updated_content = last_message["content"]

        print(updated_content)

        # Create a new HumanMessage object with the updated content
        # updated_message = HumanMessage(content=updated_content)
        updated_message = {"role": "user", "content": updated_content}

        # Replace the last message in message_list with the updated message
        message_list[-1] = updated_message

    openai.api_key = OPENAI_API_KEY
    # Send request to GPT-3 (replace with actual GPT-3 API call)
    gpt3_response = openai.ChatCompletion.create(
        model="gpt-4",
        temperature=0,
        messages=[
                     {"role": "system",
                      "content": qa_template},
                 ] + message_list
    )

    assistant_response = gpt3_response["choices"][0]["message"]["content"].strip()

    return assistant_response


def conversation_tracking(text_message, user_id):
    """
    Make remember all the conversation
    :param old_model: Open AI model
    :param user_id: telegram user id
    :param text_message: text message
    :return: str
    """
    # Get the last 10 conversations and responses for this user
    user_conversations = conversations.get(user_id, {'conversations': [], 'responses': []})
    user_messages = user_conversations['conversations'][-9:] + [text_message]
    user_responses = user_conversations['responses'][-9:]

    # Store the updated conversations and responses for this user
    conversations[user_id] = {'conversations': user_messages, 'responses': user_responses}

    # Construct the full conversation history in the user:assistant, " format
    conversation_history = []

    for i in range(min(len(user_messages), len(user_responses))):
        conversation_history.append({
            "role": "user", "content": user_messages[i]
        })
        conversation_history.append({
            "role": "assistant", "content": user_responses[i]
        })

    # Add last prompt
    conversation_history.append({
        "role": "user", "content": text_message
    })
    # Generate response
    response = generate_response_chat(conversation_history)
    # task = generate_response_chat.apply_async(args=[conversation_history])
    # response = task.get()

    # Add the response to the user's responses
    user_responses.append(response)

    # Store the updated conversations and responses for this user
    conversations[user_id] = {'conversations': user_messages, 'responses': user_responses}

    return response


def setup_chatbot(uploaded_file, model, temperature):
    """
    Sets up the chatbot with files, model, and temperature
    """
    global faiss_index
    faiss_obj_path = "models/paura.pickle"
    file_path = "files/IMC_ABPAS_Help_Manual.pdf"
    index_name = "paura"
    faiss_index = train_or_load_model(True, faiss_obj_path, file_path, index_name)
    # embeds = Embedder()
    # global bot
    # uploaded_file.seek(0)
    # file = uploaded_file.read()
    # Get the document embeddings for the uploaded file
    # vectors = embeds.getDocEmbeds(file, uploaded_file.name)

    # Create a Chatbot instance with the specified model and temperature
    # chatbot = Chatbot(model, temperature, vectors)
    # bot = chatbot
    # return chatbot


if __name__ == '__main__':
    application = ApplicationBuilder().token('').build()
    bot = setup_chatbot("", "gpt-3.5-turbo", 0)

    start_handler = CommandHandler('start', start)
    prompt_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), prompt_handler)
    application.add_handler(start_handler)
    application.add_handler(prompt_handler)

    application.run_polling()

# if __name__ == "__main__":
#     main()
