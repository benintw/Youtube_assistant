from langchain.document_loaders import YoutubeLoader
from youtube_transcript_api import NoTranscriptFound
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI

# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings

from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain

# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv

# load_dotenv()
# video_url = "https://www.youtube.com/watch?v=SBaFwemRrJ4"
# embeddings = OpenAIEmbeddings()


def create_vector_db_from_youtube_url(video_url: str, api_key: str) -> FAISS:
    # print(f"Loading transcript from URL: {video_url}")
    # loader = YoutubeLoader.from_youtube_url(video_url)
    # transcript = loader.load()

    # print(f"Transcript: {transcript}")  # Debugging statement

    # if not transcript:
    #     raise ValueError(
    #         "Transcript is empty. Please check the video URL or try another video."
    #     )

    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    # docs = text_splitter.split_documents(transcript)
    # print(f"Documents: {docs[:5]}")  # Debugging statement

    # embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    # test_embedding = embeddings.embed_documents(["test document"])

    # db = FAISS.from_documents(docs, embedding=embeddings)

    # return db
    try:
        print(f"Loading transcript from URL: {video_url}")  # Debugging statement
        loader = YoutubeLoader.from_youtube_url(video_url)
        transcript = None

        try:
            transcript = loader.load()
        except NoTranscriptFound:

            raise ValueError("Transcript is empty.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100
        )
        docs = text_splitter.split_documents(transcript)
        # print(f"Documents created: {docs[:5]}")  # Debugging statement

        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        test_embedding = embeddings.embed_documents(["test document"])
        # print(f"Test Embedding: {test_embedding}")  # Debugging statement

        db = FAISS.from_documents(docs, embedding=embeddings)

        return db

    except Exception as e:
        print(f"Error in create_vector_db_from_youtube_url: {e}")
        raise


def get_response_from_query(db, query, api_key, k=4):

    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    llm = OpenAI(openai_api_key=api_key)

    prompt = PromptTemplate(
        input_variables=["question", docs],
        template="""
        You are a helpful Youtube assistant that can answer questions about vidoes based on the video's transcript.
        answer the following question: {question}
        by searching the following video transcript: {docs}
        Only use the factual information from the transcript to answer the question.
        If you dont know, just say 'I dont know'.
        Your answers should be less than 30 tokens.
        """,
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")

    return response, docs
