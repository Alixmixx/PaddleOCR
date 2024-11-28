from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.responses import JSONResponse
from typing import List, Dict, Any
from pydantic import BaseModel
from paddleocr import PaddleOCR
import os
import numpy as np
import cv2
import base64
import logging
import imghdr
from pdf2image import convert_from_bytes
from functools import lru_cache

app = FastAPI()

API_KEY = os.environ.get('PADDLE_OCR_API_KEY', 'mirinae')

# Cache for OCR models based on parameters
ocr_models = {}

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OCRRequest(BaseModel):
    images: List[str]  # Base64-encoded images
    params: Dict[str, Any]  # Parameters for PaddleOCR.__init__()
    ocr_params: Dict[str, Any]  # Parameters for ocr_model.ocr()

def get_ocr_model(params: Dict[str, Any]) -> PaddleOCR:
    key = frozenset(params.items())
    return _get_ocr_model(key)

@lru_cache(maxsize=10)
def _get_ocr_model(key):
    params = dict(key)
    return PaddleOCR(**params)

def convert_pdf_to_images(pdf_bytes):
    try:
        images = convert_from_bytes(pdf_bytes)
        return [np.array(img) for img in images]
    except Exception as e:
        logger.exception("Error converting PDF to images")
        raise HTTPException(status_code=400, detail=f"Error converting PDF to images: {str(e)}")

@app.post("/ocr")
async def perform_ocr(
    request: OCRRequest,
    raw_request: Request,
):
    # Extract the API key from the headers and check it
    print(raw_request.headers)
    api_key = raw_request.headers.get("api_key")
    if api_key != API_KEY:
        logger.warning("Invalid API Key.")
        raise HTTPException(status_code=401, detail="Invalid API Key")

    # Extract parameters, probably should be validated
    params = request.params
    ocr_params = request.ocr_params
    try:
        ocr_model = get_ocr_model(params)
    except Exception as e:
        logger.exception("OCR model initialization failed.")
        raise HTTPException(status_code=500, detail="OCR model initialization failed.")

    formatted_results = []

    for img_str in request.images:
        # Detect if it's an image or PDF based on content
        try:
            img_data = base64.b64decode(img_str)
        except Exception as e:
            logger.exception("Invalid base64 data.")
            raise HTTPException(status_code=400, detail=f"Invalid base64 data: {str(e)}")

        file_type = imghdr.what(None, h=img_data)

        if file_type:
            # Handle image
            logger.info(f"Detected image format: {file_type}")
            img_array = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None:
                raise HTTPException(status_code=400, detail="Image decoding failed. Invalid image format.")
            images = [img]
        else:
            # Handle PDF
            logger.info("Detected PDF format, converting to images")
            try:
                images = convert_pdf_to_images(img_data)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to process PDF: {str(e)}")

        # Perform OCR on each image
        for img in images:
            try:
                result = ocr_model.ocr(img, **ocr_params)
            except Exception as e:
                logger.exception("An unexpected error occurred.")
                raise HTTPException(status_code=500, detail="An internal server error occurred.")

            formatted_result = []
            for res in result:
                for line in res:
                    coordinates = line[0]  # Bounding box coordinates
                    text = line[1][0]
                    confidence = line[1][1]
                    formatted_result.append({
                        'text': text,
                        'confidence': confidence,
                        'coordinates': coordinates
                    })
            formatted_results.append(formatted_result)

    return JSONResponse(content={'results': formatted_results})


"""
Run the server with the following command:



gunicorn -w 4 -k uvicorn.workers.UvicornWorker --timeout 60 paddle_ocr_server:app

or

uvicorn paddle_ocr_server:app --host 0.0.0.0 --port 8000

"""