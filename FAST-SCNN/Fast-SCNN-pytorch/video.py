import torch
import cv2
import numpy as np
import time
from models.fast_scnn import get_fast_scnn  # 确保fast_scnn.py在同一目录下
from flask import Flask, request, jsonify
import base64



class FastSCNNVideoProcessor:
    def __init__(self, model_path=None, use_cuda=False, half_precision=False):
        # 初始化模型，确保使用CPU
        self.device = torch.device("cpu")  # 强制使用CPU
        self.half = half_precision
        self.model = get_fast_scnn('citys', pretrained=bool(model_path)).to(self.device)

        if model_path:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)

        self.model.eval()
        # 移除半精度设置，因为CPU通常不支持半精度
        # if self.half:
        #     self.model.half()

        # # 定义颜色映射（根据Cityscapes的19个类别）
        # self.colors = np.random.randint(0, 255, (256, 3), dtype=np.uint8)  # 简化颜色映射

        self.colors = np.array([
            # 索引 0-18 对应Cityscapes的19个类别
            [128, 64, 128],  # 0: road (道路)
            [244, 35, 232],  # 1: sidewalk (人行道)
            [70, 70, 70],  # 2: building (建筑)
            [102, 102, 156],  # 3: wall (墙)
            [190, 153, 153],  # 4: fence (栅栏)
            [153, 153, 153],  # 5: pole (杆)
            [250, 170, 30],  # 6: traffic light (交通灯)
            [220, 220, 0],  # 7: traffic sign (交通标志)
            [107, 142, 35],  # 8: vegetation (植被)
            [152, 251, 152],  # 9: terrain (地形)
            [70, 130, 180],  # 10: sky (天空)
            [220, 20, 60],  # 11: person (人)
            [255, 0, 0],  # 12: rider (骑行者)
            [0, 0, 142],  # 13: car (汽车)
            [0, 0, 70],  # 14: truck (卡车)
            [0, 60, 100],  # 15: bus (公交车)
            [0, 80, 100],  # 16: train (火车)
            [0, 0, 230],  # 17: motorcycle (摩托车)
            [119, 11, 32],  # 18: bicycle (自行车)

            # 填充剩余237个位置为黑色（索引19-255）
            *[[0, 0, 0]] * 237
        ], dtype=np.uint8)

        # 由于OpenCV使用BGR顺序，需要进行颜色通道转换
        self.colors = self.colors[:, ::-1]  # RGB -> BGR

    def preprocess(self, frame):
        # 预处理流程
        orig_h, orig_w = frame.shape[:2]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (512, 256))  # 匹配模型输入尺寸
        tensor = torch.from_numpy(frame).permute(2, 0, 1).float().div(255.0)
        tensor = tensor.unsqueeze(0).to(self.device)
        # 移除半精度设置，因为CPU不支持
        # if self.half:
        #     tensor = tensor.half()
        return tensor, (orig_h, orig_w)

    def postprocess(self, output, orig_size):
        # 后处理流程
        output = output.squeeze(0).argmax(0).cpu().numpy().astype(np.uint8)
        mask = cv2.resize(output, (orig_size[1], orig_size[0]), interpolation=cv2.INTER_NEAREST)
        color_mask = self.colors[mask]
        return color_mask

    def process_frame(self, frame):
        # 完整处理流程
        tensor, orig_size = self.preprocess(frame)
        with torch.no_grad():
            outputs = self.model(tensor)[0]
        return self.postprocess(outputs[0], orig_size)


    def blend_images(self, frame, mask, alpha=0.5):
        # 将分割结果与原始图像混合
        return cv2.addWeighted(frame, alpha, mask, 1 - alpha, 0)

    def run(self, video_source=0, output_file=None, show=True, fps=30):
        # 初始化视频流
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise IOError("无法打开视频源")

        # 获取视频参数
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 初始化视频写入器
        if output_file:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

        # 性能监控
        frame_count = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 处理帧
            mask = self.process_frame(frame)
            blended = self.blend_images(frame, mask)

            # 显示性能信息
            frame_count += 1
            fps = frame_count / (time.time() - start_time)
            cv2.putText(blended, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 显示/保存结果
            if show:
                cv2.imshow('Fast-SCNN Segmentation', blended)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            if output_file:
                out.write(blended)

        # 释放资源
        cap.release()
        if output_file:
            out.release()
        cv2.destroyAllWindows()

app = Flask(__name__)
processor = FastSCNNVideoProcessor(model_path="./weights/fast_scnn_citys.pth")


@app.route('/process', methods=['POST'])
def handle_frame():
    print("\n[HTTP请求] 接收到新的处理请求")
    start_time = time.time()

    try:
        print("▷ 正在解析Base64数据")
        enc_data = request.json['image'].split(',')[-1]
        dec_data = base64.b64decode(enc_data)
        print(f"✓ 数据解析完成 | 长度：{len(dec_data)//1024}KB")

        # 图像解码
        print("▷ 正在解码图像数据")
        nparr = np.frombuffer(dec_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        print(f"✓ 图像解码成功 | 尺寸：{frame.shape[1]}x{frame.shape[0]}")

        # 处理流程
        print("▷ 开始处理图像帧")
        processed = processor.process_frame(frame)
        blended = processor.blend_images(frame, processed)
        print("✓ 图像处理完成")

        # 编码返回
        print("▷ 正在编码JPEG结果")
        _, buffer = cv2.imencode('.jpg', blended)
        b64_result = base64.b64encode(buffer).decode()
        print(f"✓ 编码完成 | 数据长度：{len(b64_result)//1024}KB")

        total_time = (time.time()-start_time)*1000
        print(f"✓ 请求处理完成 | 总耗时：{total_time:.1f}ms")
        return jsonify({
            'result': 'data:image/jpeg;base64,'+base64.b64encode(buffer).decode()
        })

    except Exception as e:
        print(f"✕ 处理异常：{str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=False)


# if __name__ == "__main__":
#     # 使用示例
#     processor = FastSCNNVideoProcessor(
#         model_path="./weights/fast_scnn_citys.pth",  # 需要提前下载权重
#         use_cuda=False,  # 强制使用CPU
#         half_precision=False  # CPU不支持半精度
#     )
#a
#     # 选择输入源（0=摄像头，或视频文件路径）
#     processor.run(
#         video_source=0,  # 或 video_source=0 使用摄像头
#         output_file="output.mp4",
#         show=True,
#         fps=30
#     )