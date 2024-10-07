from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.responses import JSONResponse
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from paddleocr import PaddleOCR
import os
import numpy as np
import cv2
import base64
import logging

app = FastAPI()

API_KEY = os.environ.get('PADDLE_OCR_API_KEY', 'your_api_key_here')

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
    # Create a unique key based on parameters to cache models
    key = frozenset(params.items())
    if key not in ocr_models:
        ocr_models[key] = PaddleOCR(**params)
    return ocr_models[key]

@app.post("/ocr")
async def perform_ocr(
    request: OCRRequest,
    raw_request: Request,
):
    logger.info("===== RECEIVED OCR REQUEST =====")

    # Log request headers
    headers = dict(raw_request.headers)
    logger.info("Headers:")
    for key, value in headers.items():
        logger.info(f"{key}: {value}")

    # Log the raw request body
    try:
        body = await raw_request.json()
        logger.info("Body:")
        logger.info(body)
    except Exception as e:
        logger.exception("Failed to parse request body.")
        raise HTTPException(status_code=400, detail=f"Failed to parse request body: {str(e)}")

    # Extract the API key from the headers and check it
    api_key = raw_request.headers.get("api_key")
    if api_key != API_KEY:
        logger.warning("Invalid API Key.")
        raise HTTPException(status_code=401, detail="Invalid API Key")

    # Extract parameters and print them
    params = request.params
    ocr_params = request.ocr_params
    logger.info("Params:")
    logger.info(params)
    logger.info("OCR Params:")
    logger.info(ocr_params)

    try:
        ocr_model = get_ocr_model(params)
    except Exception as e:
        logger.exception("OCR model initialization failed.")
        raise HTTPException(status_code=500, detail="OCR model initialization failed.")

    formatted_results = []

    for img_str in request.images:
        # Decode base64 image
        try:
            img_data = base64.b64decode(img_str)
            img_array = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Image decoding failed. The result is None.")
        except Exception as e:
            logger.exception("Invalid image data.")
            raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")

        # Run OCR on the image
        try:
            result = ocr_model.ocr(img, **ocr_params)
        except Exception as e:
            logger.exception("OCR processing failed.")
            raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")

        # Format the result
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
