# from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma

from fastapi import FastAPI, Request, HTTPException, File, UploadFile, Form, Depends, status
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

from llm_init import init_llm_and_embeddings
from langgraph_init import init_agent
import uvicorn
import os
import json
from pathlib import Path
import shutil

from typing import Annotated

from pydantic import BaseModel
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta

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


# Security settings
SECRET_KEY = "your-secret-key-here-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)


#dummy users data
fake_users_db = {
    "alice": {
        "id": 1,
        "username": "alice",
        "full_name": "Alice Smith",
        "email": "alice@example.com",
        "hashed_password": get_password_hash("secretpassword"),
        "disabled": False,
    },
    "bob": {
        "id": 2,
        "username": "bob",
        "full_name": "Bob Johnson",
        "email": "bob@example.com",
        "hashed_password": get_password_hash("secretpassword"),
        "disabled": False,
    },
    "charlie": {
        "id": 3,
        "username": "charlie",
        "full_name": "Charlie Lee",
        "email": "charlie@example.com",
        "hashed_password": get_password_hash("secretpassword"), 
        "disabled": False,
    },
}


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


class User(BaseModel):
    username: str
    email: str | None = None
    full_name: str | None = None
    disabled: bool | None = None


class UserInDB(User):
    hashed_password: str


def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)


def create_access_token(data: dict, expires_delta: timedelta | None = None):
    """Create a JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def authenticate_user(fake_db, username: str, password: str):
    """Authenticate a user with username and password."""
    user = get_user(fake_db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user



async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]):
    """Get the current user from JWT token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = get_user(fake_users_db, username=username)
    if user is None:
        raise credentials_exception
    return user


async def get_current_active_user(
    current_user: Annotated[User, Depends(get_current_user)],
):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user



@app.post("/token")
async def login(form_data: Annotated[OAuth2PasswordRequestForm, Depends()]):
    """Login endpoint to get access token."""
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/users/me")
async def read_users_me(
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    return current_user


# TODO: make auth work then add the urrent_user to the template
@app.get("/", response_class=HTMLResponse)
async def read_root(request:Request,
                    # current_user: Annotated[User, Depends(get_current_active_user)]
                    ):

    return templates.TemplateResponse("index.html", {"request":request, "user": current_user})


@app.get("/login-form", response_class=HTMLResponse)
async def read_root(request:Request):
    return templates.TemplateResponse("login.html", {"request":request})



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