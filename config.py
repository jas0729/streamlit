# config.py

# 定义类别颜色映射
CATEGORY_COLORS = {
    "boots": (255, 255, 0),  # Yellow
    "gloves": (255, 0, 0),  # Blue
    "helmet": (248, 20, 4),  # Apple Cherry
    "human": (155, 89, 182),  # Purple
    "vest": (31, 255, 214),  # 青
    # 可以继续添加其他类别及其颜色
}

# 模型路径
MODEL_PATHS = {
    "STYOLO-p": "weight/starnet_pruned.pt",
    "M4YOLO-p": "weight/mobilenetv4_pruned.pt",
    "yolov8n": "weight/yolov8n.pt",
    "yolov8s": "weight/yolov8s.pt"
}

# 默认置信度阈值
DEFAULT_CONFIDENCE_THRESHOLD = 0.7
