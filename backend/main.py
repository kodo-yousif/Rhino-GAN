import os
import copy
import torch
import uvicorn
import traceback
import threading
from queue import Queue
from pathlib import Path
from loguru import logger
from fastapi import FastAPI, Request, UploadFile, File, HTTPException, BackgroundTasks
from file_process import get_all, error_field, set_inversion_image
from fastapi.responses import JSONResponse
from face_processor import process_face
from models.Embedding import Embedding
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

maxWorkers = 1
currentWorkers = 0
imageQueue = Queue()


ii2s = Embedding(copy.deepcopy(setting))
def process_inversion(image_name:str):  
    try:
        global ii2s
        global imageQueue
        global currentWorkers

        torch.cuda.empty_cache()

        ii2s.invert_images_in_W(image_name)

        # ii2s.invert_images_in_FS(item.im)
        
        torch.cuda.empty_cache()

        if(imageQueue.empty()):
          currentWorkers = currentWorkers - 1
        else:
          process_inversion(imageQueue.get())

    except Exception as e: 
        print("Exception occurred:")
        traceback.print_exc()  # 🔍 shows full traceback
        logger.error(f"Inversion processing error: {e}")
        error_field(image_name)
        if(imageQueue.empty()):
            currentWorkers = currentWorkers - 1
        else:
            process_inversion(imageQueue.get())


def start_inversion(image_name:str):
    thread = threading.Thread(target=process_inversion, args = (image_name,))
    thread.start()


@app.post("/upload-image")
async def upload_image(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    allowed_types = ["image/png", "image/jpeg"]
    
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Only PNG and JPG images are allowed")

    file_path = os.path.join(setting.unprocessed_dir, file.filename)

    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    process_face(file_path)

    global maxWorkers
    global imageQueue
    global currentWorkers

    inversion = {
    "current_step": 0,        
    "total_steps": setting.W_steps + setting.FS_steps,
    }

    image_name = Path(file_path).stem

    set_inversion_image(image_name, inversion)
    
    if (currentWorkers == maxWorkers):
        imageQueue.put(image_name)
        return "Queued"
    else: 
        currentWorkers = currentWorkers + 1
        background_tasks.add_task(start_inversion, image_name)
        return "Started"

@app.get("/get-images")
def get_images():
    extensions = {".png"}

    BASE_IMAGE_DIR = Path(setting.output_dir)

    image_files = [
        str(path.relative_to(BASE_IMAGE_DIR))
        for path in BASE_IMAGE_DIR.rglob("*")
        if path.suffix.lower() in extensions
    ]

    return image_files

@app.get("/files-status")
def files_status(request: Request):
    return get_all()

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