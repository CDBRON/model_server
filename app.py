import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
from werkzeug.utils import secure_filename

# 确保你的 test_yolo 和 test_ocr 文件与 app.py 在同一目录下
from test_yolo import yolo_predict
from test_ocr import improved_ocr

app = Flask(__name__)
CORS(app)

# --- 配置 ---
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 替换为你的真实 API Key
API_KEY = "sk-8e8eeb4d4c6a4821af256ea4185ccfc3"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_MODEL = "qwq-32b"
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# 你的 YOLO 模型权重路径
YOLO_PATH = './checkpoints/best.pt'


@app.route('/generate', methods=['POST'])
def generate_description():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filename = secure_filename(file.filename)
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(img_path)
        try:
            # 调用模型
            _, yolo_prompt_list = yolo_predict(YOLO_PATH, img_path, save=False)
            ocr_prompt_list, _ = improved_ocr('en', img_path, yolo_prompt_list)

            yolo_info_str = "\n".join(yolo_prompt_list)
            ocr_info_str = "\n".join(ocr_prompt_list)

            prompt_text = (
                "现有一张BPMN流程图，请根据下面给出的BPMN元素与文字的坐标信息理解这张图像，并用一段简洁且逻辑清晰的文字概述这张流程图所描述的业务过程，不要流水账式的描述内容。"
                "元素和文字会以“【名称/文本内容】 【左上角x坐标】 【左上角y坐标】 【右下角x坐标】【右下角y坐标】”的格式给出。"
                f"BPMN元素信息如下：\n{yolo_info_str}\n文字信息如下\n{ocr_info_str}"
            )

            # 1. 启用 stream 模式
            completion_stream = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt_text}],
                stream=True  # <--- 关键改动在这里！
            )

            # 2. 初始化一个空字符串，用来拼接流式返回的结果
            full_response = ""

            # 3. 循环遍历返回的数据流，并将内容块拼接起来
            for chunk in completion_stream:
                # 检查每个数据块中是否有内容
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content

            # 4. 将拼接好的完整描述返回给前端
            return jsonify({'description': full_response})
        except Exception as e:
            print(f"An error occurred: {e}")
            return jsonify({'error': 'Failed to process the image on the server.'}), 500
        finally:
            if os.path.exists(img_path):
                os.remove(img_path)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)