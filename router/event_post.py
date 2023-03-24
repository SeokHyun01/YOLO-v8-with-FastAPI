from typing import Optional, List
from fastapi import APIRouter, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from ultralytics import YOLO
from PIL import Image


router = APIRouter(
    prefix='/event',
    tags=['event']
)

model = YOLO("detect-fire.pt")


class ObjectDetectionRequest(BaseModel):
    Path: str


class PredictionResult(BaseModel):
    Label: int
    Left: int
    Top: int
    Right: int
    Bottom: int
    
    
class PredictionResults(BaseModel):
    results: List[Optional[PredictionResult]]


@router.post('/create', status_code=status.HTTP_200_OK)
def create_event(request_data: ObjectDetectionRequest, response: Response):
    try:
        image = Image.open(request_data.Path)
    except FileNotFoundError:
        response.status_code = status.HTTP_404_NOT_FOUND
        return {'ErrorMessage': 'File not found.'}

    try:
        results = model.predict(image)
    except Exception as exception:
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {'ErrorMessage': str(exception)}

    image.close()

    prediction_results = []
    for result in results:
        for bbox, cls in zip(result.boxes.xyxy, result.boxes.cls):
            left, top, right, bottom = bbox.tolist()
            prediction_results.append(PredictionResult(Label=cls.item(), Left=left, Top=top, Right=right, Bottom=bottom))

    return JSONResponse(content=PredictionResults(results=prediction_results))
