# from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma


from fastapi import FastAPI
from typing import Annotated
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request, HTTPException
from fastapi import File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware

from llm_init import init_llm_and_embeddings
from langgraph_init import init_agent
import uvicorn
import os
import json
from pathlib import Path
import shutil

from text_processing import read_pdf, tokenize_pdf_text

llm, embeddings = init_llm_and_embeddings()
agent = init_agent(llm)

vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# texts = ["AI agents are autonomous decision-making systems.", "Vector databases help store and retrieve embeddings efficiently."]
# vector_store.add_texts(texts)



app = FastAPI()
templates = Jinja2Templates("templates")


UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)  # Create uploads folder if it doesn't exist


app.mount("/static", StaticFiles(directory="static"), name="static")

# Allow your Vue dev server origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:8080", "http://localhost:8000"],  # add your frontend's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def read_root(request:Request):
    return templates.TemplateResponse("index.html", {"request":request})


@app.post("/ask")
def ask_agent(message: Annotated[str, Form()]):

    question = message
    print("question:", question)
    response = "test test"
    relevant_docs = vector_store.similarity_search(question, k=1)
    context = " ".join([doc.page_content for doc in relevant_docs])

    # TODO: add context to the AI Agent!! 

    print("question:", question)
    # response = chain.run(question=question + " " + context)
    response = agent.invoke(
        {"messages": [{"role": "user", "content": f"question: {question}\njob info:{context}"}]},
    )
    print("response:", response)
    return {"status":"success", "answer": response["messages"][-1].content}


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    # file_path = UPLOAD_DIR / file.filename
    jd_filename = "JobDescriptionPDF"
    # file is saved under same name and rewritten each time
    file_path = UPLOAD_DIR / jd_filename
    print("file will be saved to:", file_path)


    # Save uploaded file to disk
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"filename": file.filename, "path": str(file_path)}


@app.post("/processpdf/")
async def process_file():
    # first delete everything in the vector store:
    all_docs_ids = vector_store.get()['ids']
    if len(all_docs_ids)>0:
        _ = vector_store.delete(all_docs_ids); # deletes all entries

    jd_filename =  UPLOAD_DIR / "JobDescriptionPDF"
    pages = await read_pdf(jd_filename)
    splits_pages = tokenize_pdf_text(pages)
    print("type splits pages:", type(splits_pages[0]))
    # TODO: enable after testing the frontend and make POST request
    ids_docs = vector_store.add_documents(documents=splits_pages)

    print("ids of the docs added:", ids_docs)
    return {"status":"success", "nr_of_pages":len(pages)} #TODO: check if succesful



@app.get("/analyse-db-docs")
def analyse_db_docs(request: Request):
    all_documents = vector_store.get()["documents"]
    # return all_documents
    return templates.TemplateResponse("view_db.html", {"request":request, "documents": json.dumps(all_documents)})



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)