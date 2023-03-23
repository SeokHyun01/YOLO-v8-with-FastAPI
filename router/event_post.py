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

model = YOLO("yolov8l.pt")


class EventHeader(BaseModel):
    Created: str
    Path: str
    IsRequiredObjectDetection: bool
    CameraId: int

class EventBody(BaseModel):
    Label: str
    Left: int
    Top: int
    Right: int
    Bottom: int


class Event(BaseModel):
    EventHeader: EventHeader
    EventBodies: Optional[List[EventBody]]


@router.post('/create', status_code=status.HTTP_200_OK)
def create_event(header: EventHeader, response: Response):
    camera_id = header.CameraId
    created = header.Created
    path = header.Path

    try:
        image = Image.open(path)
    except FileNotFoundError:
        response.status_code = status.HTTP_404_NOT_FOUND
        return {'ErrorMessage': 'File not found.'}

    try:
        results = model.predict(image)
    except Exception as exception:
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {'ErrorMessage': str(exception)}

    image.close()

    event_bodies = []
    for result in results:
        for bbox, cls in zip(result.boxes.xyxy, result.boxes.cls):
            left, top, right, bottom = bbox.tolist()
            event_bodies.append({
                'Label': int(cls.item()),
                'Left': left,
                'Top': top,
                'Right': right,
                'Bottom': bottom
            })

    send_event = Event(
        EventHeader = {
            'CameraId': camera_id,
            'Created': created,
            'Path': path,
            'IsRequiredObjectDetection': False
        },
        EventBodies = event_bodies
    ).dict()

    return JSONResponse(content=send_event)
