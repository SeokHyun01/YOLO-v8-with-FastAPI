import asyncio
from typing import Optional, List
from fastapi import APIRouter, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from ultralytics import YOLO
from PIL import Image
import aiofiles
from concurrent.futures import ThreadPoolExecutor
import io


router = APIRouter(
    prefix='/event',
    tags=['event']
)

model = YOLO("detect-fire.pt")
executor = ThreadPoolExecutor()


class ObjectDetectionRequest(BaseModel):
    Path: str


class PredictionResult(BaseModel):
    Label: int
    Left: int
    Top: int
    Right: int
    Bottom: int
    
    
class PredictionResults(BaseModel):
    Results: List[Optional[PredictionResult]]


async def load_image_async(path: str):
    async with aiofiles.open(path, 'rb') as f:
        image_data = await f.read()
    return Image.open(io.BytesIO(image_data))

def predict(image):
    return model.predict(image)

@router.post('/create', status_code=status.HTTP_200_OK)
async def create_event(request_data: ObjectDetectionRequest, response: Response):
    try:
        image = await load_image_async(request_data.Path)
    except FileNotFoundError:
        response.status_code = status.HTTP_404_NOT_FOUND
        return {'ErrorMessage': 'File not found.'}

    try:
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(executor, predict, image)
    except Exception as exception:
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {'ErrorMessage': str(exception)}

    image.close()

    prediction_results = []
    for result in results:
        for bbox, cls in zip(result.boxes.xyxy, result.boxes.cls):
            left, top, right, bottom = bbox.tolist()
            prediction_results.append(PredictionResult(Label=cls.item(), Left=left, Top=top, Right=right, Bottom=bottom))

    return PredictionResults(Results=prediction_results)
