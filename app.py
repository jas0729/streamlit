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
model_option = st.sidebar.selectbox("选择模型", ("STYOLO-p", "M4YOLO-p", "yolov8n", "yolov8s"))
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
    # 检查图像是否有4个通道，并将其转换为3个通道，同时保留alpha通道
    if frame.shape[2] == 4:
        alpha_channel = frame[:, :, 3]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    else:
        alpha_channel = None

    # 使用模型进行检测
    results = model(frame)
    detections = {
        "human_count": 0,
        "helmet_detected": False,
        "vest_detected": False
    }

    # 解析检测结果
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

            # 统计检测到的对象类型
            if class_name == "human":
                detections["human_count"] += 1
            elif class_name == "helmet":
                detections["helmet_detected"] = True
            elif class_name == "vest":
                detections["vest_detected"] = True

    # 如果有alpha通道，将检测后的图像和alpha通道合并
    if alpha_channel is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        frame[:, :, 3] = alpha_channel

    return frame, detections


# 显示检测结果
def display_results(stframe, result_text, frame, detections, width):
    # 调整帧的大小
    frame = cv2.resize(frame, (width, int(frame.shape[0] * width / frame.shape[1])))
    # 确保将帧转换为 RGB 以便正确显示颜色
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    stframe.image(frame)
    human_count = detections["human_count"]
    helmet_detected = detections["helmet_detected"]
    vest_detected = detections["vest_detected"]

    # 根据检测结果显示文本
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

            # 转换颜色空间 BGR -> RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frame, detections = process_frame(frame, confidence_threshold, model)

            # 转换颜色空间 RGB -> BGR 以便显示
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            display_results(stframes[i][0], stframes[i][1], frame, detections, width=frame_width)
    for cap in caps:
        cap.release()


# 处理 RTSP 流
def process_rtsp_stream(caps, stframes, confidence_threshold, model, frame_width):
                while any(cap.isOpened() for cap in caps):
                    for i, cap in enumerate(caps):
                        ret, frame = cap.read()
                        if not ret:
                            continue

                        # 转换颜色空间 BGR -> RGB
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                        frame, detections = process_frame(frame, confidence_threshold, model)

                        # 转换颜色空间 RGB -> BGR 以便显示
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

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
    thread_name = threading.current_thread().name  # 获取当前线程的名称
    cap = cv2.VideoCapture(video_path)  # 打开视频文件
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)  # 设置视频帧的位置
    frames_processed = 0  # 初始化处理的帧数为0

    # 获取视频的宽度、高度和帧率
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 初始化一个列表，用于存储处理后的帧
    segment_frames = []

    # 记录开始处理的日志
    logging.info(
        f'Thread {thread_name} - Start processing: {video_path} from frame {start_frame} to {start_frame + num_frames}')

    # 循环，直到视频结束或者处理的帧数达到指定的数量
    while cap.isOpened() and frames_processed < num_frames:
        ret, frame = cap.read()  # 读取一帧视频
        if not ret:  # 如果读取失败，跳出循环
            break
        results = model(frame)  # 使用模型处理这一帧
        for result in results:  # 对于处理结果中的每一项
            result_frame = result.plot()  # 绘制结果
            segment_frames.append(result_frame)  # 将处理后的帧添加到列表中
        frames_processed += 1  # 处理的帧数加1

    # 记录结束处理的日志
    logging.info(
        f'Thread {thread_name} - Finished processing: {video_path} from frame {start_frame} to {start_frame + num_frames}')

    cap.release()  # 释放视频文件
    # 返回处理的开始帧、处理后的帧、帧率、宽度和高度
    return start_frame, segment_frames, fps, frame_width, frame_height


# 分段处理RTSP流
def process_rtsp_segment(model, rtsp_url, num_frames):
    cap = cv2.VideoCapture(rtsp_url)  # 打开RTSP流
    frames_processed = 0  # 初始化处理的帧数为0

    # 初始化一个列表，用于存储处理后的帧
    segment_frames = []

    # 如果无法打开RTSP流，记录错误日志，并返回空的处理结果
    if not cap.isOpened():
        logging.error(f"Failed to open RTSP stream: {rtsp_url}")
        return segment_frames, 0, 0, 0

    # 获取视频的宽度、高度和帧率
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 循环，直到处理的帧数达到指定的数量
    while frames_processed < num_frames:
        ret, frame = cap.read()  # 读取一帧视频
        if not ret:  # 如果读取失败，跳出循环
            break
        results = model(frame)  # 使用模型处理这一帧
        for result in results:  # 对于处理结果中的每一项
            result_frame = result.plot()  # 绘制结果
            segment_frames.append(result_frame)  # 将处理后的帧添加到列表中
        frames_processed += 1  # 处理的帧数加1

    cap.release()  # 释放RTSP流
    # 返回处理后的帧、帧率、宽度和高度
    return segment_frames, fps, frame_width, frame_height


# 保存视频
def save_video(output_path, all_frames, fps, frame_width, frame_height):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    for frame in all_frames:
        out.write(frame)
    out.release()


# 高并发情况下的多线程推理
def detect(video_files_or_urls, num_frames_per_segment, thread_count, output_dir, original_names):
    # 初始化每个视频文件或URL的处理进度为0
    for item in video_files_or_urls:
        progress_dict[item] = 0
    # 创建一个线程池，用于并发处理视频文件或URL
    with ThreadPoolExecutor(max_workers=thread_count) as executor:
        # 初始化一个列表，用于存储每个视频文件或URL的处理任务
        futures = []
        # 初始化一个字典，用于存储每个视频文件或URL的处理结果
        video_segments = {item: [] for item in video_files_or_urls}

        # 循环，直到所有的视频文件或URL都处理完毕
        while True:
            # 假设所有的视频文件或URL都已经处理完毕
            all_finished = True
            # 对于每个视频文件或URL
            for item in video_files_or_urls:
                # 获取当前的处理进度
                with progress_lock:
                    start_frame = progress_dict[item]

                # 如果当前的处理进度为-1，表示该视频文件或URL已经处理完毕，跳过
                if start_frame == -1:
                    continue

                # 如果有未处理完的视频文件或URL，设置all_finished为False
                all_finished = False
                # 如果是RTSP流，使用process_rtsp_segment函数处理
                if item.startswith("rtsp://"):
                    future = executor.submit(process_rtsp_segment, model, item, num_frames_per_segment)
                else:
                    # 如果是视频文件，使用process_video_segment函数处理
                    future = executor.submit(process_video_segment, model, item, start_frame, num_frames_per_segment)
                # 将处理任务添加到futures列表中
                futures.append((future, item))
                
                # 更新处理进度
                with progress_lock:
                    if item.startswith("rtsp://"):
                        progress_dict[item] += num_frames_per_segment
                    else:
                        # 获取视频文件的总帧数
                        total_frames = int(cv2.VideoCapture(item).get(cv2.CAP_PROP_FRAME_COUNT))
                        # 如果剩余的帧数少于num_frames_per_segment，设置处理进度为-1，表示该视频文件已经处理完毕
                        if start_frame + num_frames_per_segment >= total_frames:
                            progress_dict[item] = -1
                        else:
                            # 否则，增加处理进度
                            progress_dict[item] += num_frames_per_segment

            # 如果所有的视频文件或URL都已经处理完毕，退出循环
            if all_finished:
                break
        # 等待所有的处理任务完成        
        for future, item in futures:
            try:
                # 获取处理结果，并添加到video_segments字典中
                if item.startswith("rtsp://"):
                    segment_frames, fps, frame_width, frame_height = future.result()
                    video_segments[item].append((0, segment_frames, fps, frame_width, frame_height))
                else:
                    start_frame, segment_frames, fps, frame_width, frame_height = future.result()
                    video_segments[item].append((start_frame, segment_frames, fps, frame_width, frame_height))
            except Exception as exc:
                # 如果处理过程中发生异常，记录异常信息
                logging.error(f"Video processing generated an exception: {exc}")
    # 对于每个视频文件或URL，将处理后的视频片段保存到文件中
    for item, segments in video_segments.items():
        all_frames = []
        for start_frame, segment_frames, fps, frame_width, frame_height in sorted(segments, key=lambda x: x[0]):
            all_frames.extend(segment_frames)
        original_name = original_names[item]
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

            # 将图像从 BGR 转换回 RGB 以便显示
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            display_results(stframe, result_text, frame, detections, width=video_frame_width)

elif upload_option == "高并发推理":
    concurrent_option = st.radio("选择上传类型", ("视频文件", "RTSP流"))
    if concurrent_option == "视频文件":
        uploaded_files = st.file_uploader("上传多个视频文件，支持mp4、avi、mov、mkv", type=["mp4", "avi", "mov", "mkv"],
                                          accept_multiple_files=True)
    elif concurrent_option == "RTSP流":
        rtsp_urls = st.text_area("输入多个RTSP流地址，多个时每行一个", height=150).splitlines()

    core_thread_count = st.number_input("工作线程数", min_value=1, max_value=16, value=4)
    num_frames_per_segment = st.number_input("每段处理帧数", min_value=1, max_value=1000, value=100)
    start_button = st.button("开始检测")

    if start_button:
        output_dir = os.path.join(os.getcwd(), "high_concurrent_detect_results")
        os.makedirs(output_dir, exist_ok=True)
        temp_files = []
        original_names = {}

        if concurrent_option == "视频文件" and uploaded_files:
            for uploaded_file in uploaded_files:
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(uploaded_file.read())
                temp_files.append(tfile.name)
                original_names[tfile.name] = uploaded_file.name
        elif concurrent_option == "RTSP流" and rtsp_urls:
            for i, url in enumerate(rtsp_urls):
                temp_files.append(url)
                original_names[url] = f"rtsp_stream_{i}.mp4"

        detect(temp_files, num_frames_per_segment, core_thread_count, output_dir, original_names)

        st.write("高并发推理处理完成。结果已保存到 high_concurrent_detect_results 文件夹。")
        for item in temp_files:
            st.write(f"处理完成的视频: {original_names[item]}")
            if item.startswith("rtsp://"):
                st.video(os.path.join(output_dir, original_names[item]))
            else:
                st.video(item)

elif upload_option == "RTSP流":
    rtsp_urls = st.text_area("输入RTSP流地址，多个时每行一个", height=150).splitlines()
    if rtsp_urls:
        start_button = st.button("开始检测")
        if start_button:
            stframes = [[st.empty(), st.empty()] for _ in rtsp_urls]
            caps = [cv2.VideoCapture(url) for url in rtsp_urls]
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

            # 转换颜色空间 BGR -> RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frame, detections = process_frame(frame, confidence_threshold, model)

            # 转换颜色空间 RGB -> BGR 以便显示
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            display_results(stframe, result_text, frame, detections, width=video_frame_width)
        cap.release()
