# PaddleOCR Server

## Installation
    
### Create virtual env
    ```
        python3 -m venv venv
        source venv/bin/activate
        pip install -r requirements.txt
    ```

### Run the server
    ```
        gunicorn -w 4 -k uvicorn.workers.UvicornWorker --timeout 60 paddle_ocr_server:app
    ```

    ```
        uvicorn paddle_ocr_server:app --host 0.0.0.0 --port 8000
    ```

### TODO

    - setup the server on test machine
    - enable gpu usage
    - Find correct version of paddleOCR for server
    - Make documentation
    - Run tests