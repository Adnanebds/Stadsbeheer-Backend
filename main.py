from typing import Union

from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/omar")
def read_root():
    return {"Hoi": "Omar"}


@app.get("/tarik")
def read_root():
    return {"Hoi": "Tarik"}


@app.get("/adnane")
def read_root():
    return {"Hoi": "adanane"}