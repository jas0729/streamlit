import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
from PIL import Image
import config
import warnings

warnings.filterwarnings('ignore')
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import logging

# Streamlit 页面标题
st.title("安全上岗检测")

# 在侧边栏添加导航栏选项
st.sidebar.title("导航栏")
confidence_threshold = st.sidebar.slider("置信度阈值", 0.0, 1.0, config.DEFAULT_CONFIDENCE_THRESHOLD)
model_option = st.sidebar.selectbox("选择模型", ("starnet_pruned", "mobilenet_pruned", "yolov8n", "yolov8s"))
video_frame_width = st.sidebar.slider("视频框宽度", 200, 800, 400)

# 根据用户选择加载相应的模型
model_path = config.MODEL_PATHS[model_option]

# 加载 YOLO 模型
model = YOLO(model_path)

# 上传文件选项
upload_option = st.sidebar.selectbox("选择上传类型",
                                     ("图片", "视频流", "RTSP流", "本地摄像头", "高并发推理"))

# 处理单帧图像
def process_frame(frame, confidence_threshold, model):
    results = model(frame)
    detections = {
        "human_count": 0,
        "helmet_detected": False,
        "vest_detected": False
    }

    for result in results:
        for box in result.boxes:
            if box.conf[0] < confidence_threshold:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            confidence = box.conf[0]
            class_name = result.names[int(box.cls[0])]

            color = config.CATEGORY_COLORS.get(class_name, (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if class_name == "human":
                detections["human_count"] += 1
            elif class_name == "helmet":
                detections["helmet_detected"] = True
            elif class_name == "vest":
                detections["vest_detected"] = True

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame, detections

# 显示检测结果
def display_results(stframe, result_text, frame, detections, width):
    # 调整帧的大小
    frame = cv2.resize(frame, (width, int(frame.shape[0] * width / frame.shape[1])))
    stframe.image(frame)
    human_count = detections["human_count"]
    helmet_detected = detections["helmet_detected"]
    vest_detected = detections["vest_detected"]

    if human_count == 1:
        if helmet_detected and vest_detected:
            result_text.markdown("检测结果：**Pass**")
        else:
            result_text.markdown("检测结果：**Wrong**")
    elif human_count > 1:
        result_text.markdown("检测结果：检测到多人，请单人完成检测")
    else:
        result_text.markdown("检测结果：请站远些，上半身完整进入摄像头")

# 处理视频
def process_video(caps, stframes, result_texts, confidence_threshold, model, frame_width):
    while any(cap.isOpened() for cap in caps):
        for i, cap in enumerate(caps):
            ret, frame = cap.read()
            if not ret:
                continue
            frame, detections = process_frame(frame, confidence_threshold, model)
            display_results(stframes[i][0], stframes[i][1], frame, detections, width=frame_width)
    for cap in caps:
        cap.release()

# 高并发推理相关代码
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 全局进度字典，记录分割片段
progress_dict = {}
progress_lock = threading.Lock()

# 分段处理视频
def process_video_segment(model, video_path, start_frame, num_frames):
    thread_name = threading.current_thread().name
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames_processed = 0

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    segment_frames = []

    logging.info(
        f'Thread {thread_name} - Start processing: {video_path} from frame {start_frame} to {start_frame + num_frames}')

    while cap.isOpened() and frames_processed < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        for result in results:
            result_frame = result.plot()
            segment_frames.append(result_frame)
        frames_processed += 1

    logging.info(
        f'Thread {thread_name} - Finished processing: {video_path} from frame {start_frame} to {start_frame + num_frames}')

    cap.release()
    return start_frame, segment_frames, fps, frame_width, frame_height

# 保存视频
def save_video(output_path, all_frames, fps, frame_width, frame_height):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    for frame in all_frames:
        out.write(frame)
    out.release()

# 高并发情况下的多线程推理
def detect(video_files, num_frames_per_segment, thread_count, output_dir, original_names):
    for video in video_files:
        progress_dict[video] = 0

    with ThreadPoolExecutor(max_workers=thread_count) as executor:
        futures = []
        video_segments = {video: [] for video in video_files}

        while True:
            all_finished = True
            for video in video_files:
                with progress_lock:
                    start_frame = progress_dict[video]

                if start_frame == -1:
                    continue

                all_finished = False
                future = executor.submit(process_video_segment, model, video, start_frame, num_frames_per_segment)
                futures.append((future, video))

                with progress_lock:
                    total_frames = int(cv2.VideoCapture(video).get(cv2.CAP_PROP_FRAME_COUNT))
                    if start_frame + num_frames_per_segment >= total_frames:
                        progress_dict[video] = -1
                    else:
                        progress_dict[video] += num_frames_per_segment

            if all_finished:
                break

        for future, video in futures:
            try:
                start_frame, segment_frames, fps, frame_width, frame_height = future.result()
                video_segments[video].append((start_frame, segment_frames, fps, frame_width, frame_height))
            except Exception as exc:
                logging.error(f"Video processing generated an exception: {exc}")

    for video, segments in video_segments.items():
        all_frames = []
        for start_frame, segment_frames, fps, frame_width, frame_height in sorted(segments, key=lambda x: x[0]):
            all_frames.extend(segment_frames)
        original_name = original_names[video]
        output_path = os.path.join(output_dir, original_name)
        save_video(output_path, all_frames, fps, frame_width, frame_height)
        logging.info(f"Saved processed video: {output_path}")

if upload_option == "视频流":
    uploaded_files = st.file_uploader("上传视频文件，支持mp4、avi、mov、mkv", type=["mp4", "avi", "mov", "mkv"],
                                      accept_multiple_files=True)
    if uploaded_files:
        start_button = st.button("开始检测")
        if start_button:
            stframes = [[st.empty(), st.empty()] for _ in uploaded_files]
            temp_files = []
            caps = []
            for uploaded_file in uploaded_files:
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(uploaded_file.read())
                temp_files.append(tfile)
                caps.append(cv2.VideoCapture(tfile.name))

            process_video(caps, stframes, [stframe[1] for stframe in stframes], confidence_threshold, model,
                          frame_width=video_frame_width)

elif upload_option == "图片":
    uploaded_image = st.file_uploader("上传图片文件，支持jpg、jpeg、png", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        start_button = st.button("开始检测")
        if start_button:
            image = Image.open(uploaded_image)
            frame = np.array(image)

            stframe = st.empty()
            result_text = st.empty()

            frame, detections = process_frame(frame, confidence_threshold, model)
            display_results(stframe, result_text, frame, detections, width=video_frame_width)

elif upload_option == "高并发推理":
    uploaded_files = st.file_uploader("上传多个视频文件，支持mp4、avi、mov、mkv", type=["mp4", "avi", "mov", "mkv"],
                                      accept_multiple_files=True)
    core_thread_count = st.number_input("核心线程数", min_value=1, max_value=16, value=4)
    num_frames_per_segment = st.number_input("每段处理帧数", min_value=1, max_value=1000, value=100)
    if uploaded_files:
        start_button = st.button("开始检测")
        if start_button:
            output_dir = os.path.join(os.getcwd(), "high_concurrent_detect_results")
            os.makedirs(output_dir, exist_ok=True)
            temp_files = []
            original_names = {}

            for uploaded_file in uploaded_files:
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(uploaded_file.read())
                temp_files.append(tfile.name)
                original_names[tfile.name] = uploaded_file.name

            detect(temp_files, num_frames_per_segment, core_thread_count, output_dir, original_names)

            st.write("高并发推理处理完成。结果已保存到 high_concurrent_detect_results 文件夹。")
            for video in temp_files:
                st.write(f"处理完成的视频: {original_names[video]}")
                st.video(os.path.join(output_dir, original_names[video]))

elif upload_option == "RTSP流":
    rtsp_urls = st.text_area("输入RTSP流地址，多个时每行一个", height=150).splitlines()
    if rtsp_urls:
        start_button = st.button("开始检测")
        if start_button:
            stframes = [[st.empty(), st.empty()] for _ in rtsp_urls]
            caps = [cv2.VideoCapture(url) for url in rtsp_urls]

            def process_rtsp_stream(caps, stframes, confidence_threshold, model, frame_width):
                while any(cap.isOpened() for cap in caps):
                    for i, cap in enumerate(caps):
                        ret, frame = cap.read()
                        if not ret:
                            continue
                        frame, detections = process_frame(frame, confidence_threshold, model)
                        display_results(stframes[i][0], stframes[i][1], frame, detections, width=frame_width)
                for cap in caps:
                    cap.release()

            process_rtsp_stream(caps, stframes, confidence_threshold, model, video_frame_width)

elif upload_option == "本地摄像头":
    camera_index = st.number_input("希望调用的摄像头：", value=0, step=1)
    if st.button("开始检测"):
        cap = cv2.VideoCapture(camera_index)
        stframe = st.empty()
        result_text = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame, detections = process_frame(frame, confidence_threshold, model)
            display_results(stframe, result_text, frame, detections, width=video_frame_width)
        cap.release()
