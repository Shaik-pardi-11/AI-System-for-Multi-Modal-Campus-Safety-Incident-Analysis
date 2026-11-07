from ultralytics import YOLO

# Export YOLO26n to OpenVINO FP16 for Intel Arc GPU inference.
model = YOLO("yolo26n.pt")
exported = model.export(
    format="openvino",
    imgsz=512,
    quantize=16,
)
print("Export complete:", exported)
