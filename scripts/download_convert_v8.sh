for s in n s m l x; do
    if [ ! -f "yolov8${s}-pose.pt" ]; then
        curl -L -o "yolov8${s}-pose.pt" \
            "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8${s}-pose.pt"
    fi
    if [ ! -f "yolov8${s}-pose.safetensors" ]; then
        mlx-yolos convert \
            --pt "yolov8${s}-pose.pt" \
            --out "yolov8${s}-pose.safetensors"
    fi
done