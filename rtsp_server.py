import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GstRtspServer, GObject

# 初始化GStreamer
Gst.init(None)

# 自定义RTSP媒体工厂类，用于创建媒体流
class CustomRTSPMediaFactory(GstRtspServer.RTSPMediaFactory):
    def __init__(self, filepath):
        GstRtspServer.RTSPMediaFactory.__init__(self)
        self.filepath = filepath

    # 重写创建元素的方法
    def do_create_element(self, url):
        # 设置管道从文件读取并通过RTSP流媒体传输
        src = f"filesrc location={self.filepath} ! decodebin ! x264enc ! rtph264pay name=pay0 pt=96"
        return Gst.parse_launch(src)

# RTSP服务器类
class RTSPServer:
    def __init__(self, streams):
        self.server = GstRtspServer.RTSPServer()
        self.server.set_service("8554")  # 设置服务端口
        mount_points = self.server.get_mount_points()

        # 为每个流创建一个媒体工厂并添加到挂载点
        for mount_point, filepath in streams.items():
            factory = CustomRTSPMediaFactory(filepath)
            factory.set_shared(True)  # 设置工厂共享
            mount_points.add_factory(mount_point, factory)

        self.server.attach(None)  # 启动服务器
        self.print_stream_urls(streams)

    # 打印流的URL
    def print_stream_urls(self, streams):
        print("RTSP服务器正在运行，以下是可用的URL：")
        for mount_point in streams.keys():
            print(f"rtsp://localhost:8554{mount_point}")

if __name__ == '__main__':
    # 字典，键为挂载点，值为文件路径
    streams = {
        "/stream1": "videos/video1.mp4",
        "/stream2": "videos/video2.mp4",
        "/stream3": "videos/video3.mp4"
    }

    # 创建RTSP服务器实例并运行事件循环
    server = RTSPServer(streams)
    loop = GObject.MainLoop()
    loop.run()
