from fastapi import FastAPI
from router import event_post


app = FastAPI()
app.include_router(event_post.router)
