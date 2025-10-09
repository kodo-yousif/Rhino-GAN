import copy
import uvicorn

from loguru import logger
from fastapi import FastAPI, Request

from setting import setting

logger.add("app.log", rotation="5 MB", retention="10 days", level="INFO")
logger.info("🚀 Nose-AI FastAPI starting...")

app = FastAPI(
    title="Nose-AI API",
    description="A minimal studio for nose manipulation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",  
    openapi_url="/openapi.json"
)
configDetails = copy.deepcopy(setting)   

@app.get("/", tags=["Welcome"])
def welcome(request: Request):
    print(request.client.host)
    logger.info(f"Home called by:: {request.client.host}")
    
    return "Hi Welcome to Nose Manipulation!!"


@app.get("/health", tags=["System"])
def health_check():
    logger.info("Health check endpoint called.")
    return {"status": "ok", "message": "System healthy ✅"}

# -----------------------------------------------------
# Run server
# -----------------------------------------------------
if __name__ == "__main__":
    logger.info("✅ Listening on 0.0.0.0:3001")
    uvicorn.run("main:app", host="0.0.0.0", port=3001, reload=True)