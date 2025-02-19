import time
import uvicorn
from loguru import logger
from src.validator import InputValidator, OutputValidator
from fastapi import FastAPI, status, HTTPException
from fastapi.responses import JSONResponse

from src.model import ModelClassifcation

app = FastAPI()

def predict(data:dict):
    '''
    Calling the main function of the prediction
    '''
    logger.info(f'calling the topic classification model on complaint_id:{data.get("complaint_id","Not present")}......:')
    start_time=time.time()
    data_object=ModelClassifcation(data)
    result=data_object.predict()
    end_time=time.time()
    logger.info(f"Total time taken for prediction: {end_time-start_time} sec")
    return result

@app.get("/")
def read_root() -> JSONResponse:
    return JSONResponse(
        content={"message": "This is a topic classification model for complaints"},
        status_code=status.HTTP_200_OK,
        media_type="application/json",
    )

@app.get("/health")
def health_check() -> JSONResponse:
    return JSONResponse(
        content={"message": "OK"},
        status_code=status.HTTP_200_OK,
        media_type="application/json",
    )


@app.post("/predict", response_model=OutputValidator)
def predict_endpoint(request_data: InputValidator) -> JSONResponse:
    # Convert Pydantic model to dictionary
    try:
        data=request_data.model_dump()
        prediction_result = predict(data)
        return JSONResponse(
            content=prediction_result,
            status_code=status.HTTP_200_OK,
            media_type="application/json",
        )
    except Exception as e:
        logger.error(f"Error in predict_endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Run the FastAPI application
    uvicorn.run(
        app,
        host="0.0.0.0",  # nosec
        port=8000,
    )
