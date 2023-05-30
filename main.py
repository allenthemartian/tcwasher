from fastapi import FastAPI, File
from segmentation_onnx import get_yolov5, get_image_from_bytes
from starlette.responses import Response
import io
import uvicorn
import base64
import cv2
from PIL import Image
import json
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import uvicorn
from multiprocessing import cpu_count, freeze_support
import time
import uvicorn

# def start_server(host="0.0.0.0",
#                  port=10000,
#                  num_workers=4,
#                  loop="asyncio",
#                  reload=False):
#     uvicorn.run("main:app",
#                 host=host,
#                 port=port,
#                 workers=num_workers,
#                 loop=loop,
#                 reload=reload)
LOCAL_WORKSPACE_PATH = '.'

paths = {
    'TEST_PATH': LOCAL_WORKSPACE_PATH + '/test_data',
    'PRED_PATH': LOCAL_WORKSPACE_PATH + '/preds'
 }


model = get_yolov5()

app = FastAPI(
    title="Custom YOLOV5 Machine Learning API",
    description="""Obtain object value out of image
                    and return image and json result""",
    version="0.0.1",
)

origins = [
    "http://localhost",
    "http://localhost:8000",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/v1.0/check-status")
async def root():
    return {"Alive": True}

@app.get('/notify/v1/health')
def get_health():
    """
    Usage on K8S
    readinessProbe:
        httpGet:
            path: /notify/v1/health
            port: 80
    livenessProbe:
        httpGet:
            path: /notify/v1/health
            port: 80
    :return:
        dict(msg='OK')
    """
    return dict(msg='OK')


@app.post("/object-to-json")
async def detect_object_return_json_result(file: bytes = File(...)):
    input_image = get_image_from_bytes(file)
    results = model(input_image)
    detect_res = results.pandas().xyxy[0].to_json(orient="records")  # JSON img1 predictions
    detect_res = json.loads(detect_res)
    return {"result": detect_res}


@app.post("/telturbo/predict/image")
async def detect_object_return_base64_img(file: bytes = File(...)):
    input_image = get_image_from_bytes(file)
    results = model(input_image)
    results.render(labels=False)  # labels=False, hide label name + conf
    for img in results.ims:
        # bytes_io = io.BytesIO()
        # img_base64 = Image.fromarray(img)
        # im_bytes = img_base64.tobytes()
        # im_b64 = base64.b64encode(im_bytes)
        
    #     print(im_b64)
    #     img_base64.save(bytes_io, format="jpeg")
    # return Response(content=bytes_io.getvalue(), media_type="image/jpeg")

        image_np_with_detections = img.copy()
        array = cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50] # Reduce quality (preserve aspect ratio) - default starts at 95
        _, im_arr = cv2.imencode('.jpg', array, encode_param)  # im_arr: image in Numpy one-dim array format.
        im_bytes = im_arr.tobytes()
        im_b64 = base64.b64encode(im_bytes)
       
    return im_b64


@app.post("/telturbo/washer/detect/json")
async def detect_object_json(file: bytes = File(...)):    
    # start_time = time.time()

    # Datetime For Filename
    now = datetime.now()
    now = str(now)
    now = now.replace('-', '').replace(' ', '_').replace(':', '').replace('.', '_')
    detection_classes = []
    input_image = get_image_from_bytes(file)
    results = model(input_image)
    detect_res = results.pandas().xyxy[0].to_json(orient="records")  # JSON predictions
    detect_res = json.loads(detect_res)
    for result in detect_res:
        detection_classes.append(result["name"])
    results.render(labels=False)
    for img in results.ims:
        image_np_with_detections = img.copy()
        local_pred_im = Image.fromarray(image_np_with_detections)
        local_pred_im.save(paths['PRED_PATH'] + '/' + f'{now}.jpg')
        array = cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50] # Reduce quality (preserve aspect ratio) - default starts at 95
        _, im_arr = cv2.imencode('.jpg', array, encode_param)  # im_arr: image in Numpy one-dim array format.
        im_bytes = im_arr.tobytes()
        im_b64 = base64.b64encode(im_bytes)
    inspection_result = True

    if len(detection_classes) < 2:
        inspection_result = False
    else:     
        for class_names in detection_classes:
            if class_names == 'DoubleStackedWasher':
                inspection_result = False

    # print("--- %s seconds ---" % (time.time() - start_time))
    # print(inspection_result)
    # return {"json": detect_res, "b64" : im_b64}
    return {"inspection_result" : inspection_result, "classes": detection_classes, "b64" : im_b64}


# if __name__ == '__main__':
#     # uvicorn.run(app, port=8082, host='0.0.0.0')
#     freeze_support()  # Needed for pyinstaller for multiprocessing on WindowsOS
#     num_workers = int(cpu_count() * 0.75)
#     start_server(num_workers=num_workers)
    