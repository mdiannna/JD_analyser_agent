from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


async def read_pdf(file_path):
    loader = PyPDFLoader(file_path)
    pages = []
    async for page in loader.alazy_load():
        pages.append(page)
    return pages


def tokenize_pdf_text(pages):
    """ Tokenize a pdf text into sentences or paragraphs """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(pages)
    return all_splits

# _ = vector_store.add_documents(documents=all_splits)