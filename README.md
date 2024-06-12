# 安装说明

要安装所需的软件包，运行：

```sh
pip install -r requirements.txt -I https://pypi.tuna.tsinghua.edu.cn/simple
```
如果报错，尝试去掉
```sh
-I https://pypi.tuna.tsinghua.edu.cn/simple
```

# v1.0
整合了图片、视频流、RTSP流、本地摄像头、高并发推理等推理方式

可以选择已经训练好的模型，目前有"STYOLO-p", "M4YOLO-p", "yolov8n", "yolov8s"可选

可以选择置信度阈值，自动过滤较低置信度的识别物，减小误差

手动调整实时推理时的预览视频框大小

高并发推理模式下选择核心线程数和视频切片帧数

# v1.1
修复了两个bug

1、不能检测颜色空间为：BGRA的图片

2、图片检测结果图颜色错乱

现在可以愉快地进行各种格式的图片检测了！ ^_^

changed branch name to 'main'

# 已知bug：

高并发推理的RTSP流方式目前有些问题，推理非常慢且会卡住，只能controlC中断
