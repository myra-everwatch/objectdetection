from ultralytics import YOLO

model = YOLO("yolov8n.yaml")

training = model.train(data="data.yaml", epochs=100, imgsz=640, device=0)

metrics = model.val()
map50_95 = metrics.box.map
map50 = metrics.box.map50
map75 = metrics.box.map75
maps = metrics.box.maps


metrics_df = pd.DataFrame({
    'mAP50-95': [map50_95],
    'mAP50': [map50],
    'mAP75': [map75],
    'mAPs': [maps]
})




#results = model("")

# Export the model to ONNX format
path = model.export(format="onnx")
