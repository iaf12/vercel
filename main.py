import cv2
import numpy as np
from ultralytics import YOLO
from fastapi import FastAPI, File, UploadFile
import tempfile
import os
import json
from fastapi.responses import StreamingResponse


app = FastAPI()

# Load YOLO model
model = YOLO('best.pt')

# Async generator function that yields frame data
async def frame_generator(temp_file):
    try:
        cap = cv2.VideoCapture(temp_file.name)
        crowd_density_threshold = 20  # Minimum people count for "Crowded"
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            cnf = 0.5
            results = model(frame, conf=cnf)
            people_found = sum(1 for r in results for _ in r.boxes)

            crowd_status = "1" if people_found >= crowd_density_threshold else "0"
            yield f"data: crowdStatus:{crowd_status}\n"
    finally:
        cap.release()
        os.remove(temp_file.name)  # Clean up temporary file



@app.post("/upload-video/")
async def upload_video(file: UploadFile = File(...)):
    # Save the uploaded video file to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    video_bytes = await file.read()
    temp_file.write(video_bytes)
    temp_file.close()  

    # Open the video file using OpenCV
    

    # if not cap.isOpened():
    #     os.remove(temp_file.name)  # Clean up the temporary file in case of failure
    #     return {"error": "Failed to open video"}



    # # Clean up the temporary file after the video is processed and streaming starts
    # cap.release()
    # os.remove(temp_file.name)

    return StreamingResponse(frame_generator(temp_file), media_type="text/event-stream")



if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
