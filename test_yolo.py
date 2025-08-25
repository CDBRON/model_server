from ultralytics import YOLO
from PIL import Image
import os
from tqdm import tqdm
import numpy as np
import cv2

classes = {
    "0": "任务",
    "1": "折叠子流程",
    "2": "展开子流程",
    "3": "调用活动",
    "4": "开始事件",
    "5": "中间事件",
    "6": "结束事件",
    "7": "消息启动事件",
    "8": "中间消息捕获事件",
    "9": "中间消息抛出事件",
    "10": "结束消息事件",
    "11": "定时启动事件",
    "12": "中间定时事件",
    "13": "独占网关",
    "14": "并行网关",
    "15": "包容网关",
    "16": "基于事件的网关",
    "17": "池",
    "18": "泳道",
    "19": "数据对象",
    "20": "数据存储",
    "21": "标签",
    "22": "顺序流",
    "23": "消息流",
    "24": "数据关联"
}

class_eng = {
    "0": "Task",
    "1": "Collapsed Sub-Process",
    "2": "Expanded Sub-Process",
    "3": "Call Activity",
    "4": "Start Event",
    "5": "Intermediate Event",
    "6": "End Event",
    "7": "Message Start Event",
    "8": "Intermediate Message Catch Event",
    "9": "Intermediate Message Throw Event",
    "10": "Message End Event",
    "11": "Timer Start Event",
    "12": "Intermediate Timer Event",
    "13": "Exclusive Gateway",
    "14": "Parallel Gateway",
    "15": "Inclusive Gateway",
    "16": "Event-Based Gateway",
    "17": "Pool",
    "18": "Lane",
    "19": "Data Object",
    "20": "Data Store",
    "21": "Label",
    "22": "Sequence Flow",
    "23": "Message Flow",
    "24": "Data Association"
}


def yolo_predict(model_path, img_path, save=True, flag=0):
    model = YOLO(model_path)
    img = cv2.imread(img_path)
    img2 = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    results = model.predict(source=img, save=save)

    a = np.array(results[0].boxes.xyxy.cpu())
    b = np.array(results[0].boxes.cls.cpu())
    prompt_yolo = []

    cnt = 0
    object_list = []
    for c, xy in zip(b, a):
        prompt = "{} {:.1f} {:.1f} {:.1f} {:.1f}".format(classes[f"{int(c)}"], xy[0], xy[1], xy[2], xy[3])
        prompt_yolo.append(prompt)
        try:
            if c not in [22, 23, 24]:
                cropped_img = img2.crop((xy[0] - 20, xy[1] - 20, xy[2] + 20, xy[3] + 20))
                cropped_img.save("./temp/{}_{}_{}.png".format(cnt, flag, class_eng[f"{int(c)}"]))
                object_list.append("{}_{}_{}.png".format(cnt, flag, class_eng[f"{int(c)}"]))
                cnt += 1
        except Exception as e:
            continue
        print(prompt)

    return object_list, prompt_yolo


def make_database(model_path, img_path, save=False):
    model = YOLO(model_path)
    img = cv2.imread(img_path)
    results = model.predict(source=img, save=save)

    a = np.array(results[0].boxes.xyxy.cpu())
    b = np.array(results[0].boxes.cls.cpu())
    prompt_yolo = []

    for c, xy in zip(b, a):
        if c == 22 or c == 23 or c == 24:
            prompt = "{},{:.1f},{:.1f},{:.1f},{:.1f}".format(img_path.split('/')[4], xy[0], xy[1], xy[2], xy[3])
            prompt_yolo.append(prompt)

    # 解析检测结果并在图像上绘制边界框
    for result in prompt_yolo:
        parts = result.split(',')
        x1 = int(float(parts[1]))
        y1 = int(float(parts[2]))
        x2 = int(float(parts[3]))
        y2 = int(float(parts[4]))

        # 绘制边界框
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 保存带有边界框的图像
    output_path = './temp/boxes/{}_with_boxes.png'.format(img_path.split('/')[4])
    cv2.imwrite(output_path, img)

    return prompt_yolo


if __name__ == '__main__':
    # yolo_predict('./checkpoints/best.pt', './sample/0601.png', save=True)
    # make_database('./checkpoints/best.pt', './sample/0601.png')

    txt = []
    root = './sam/datasets/images'
    for item in tqdm(os.listdir(root)):
        prompt = make_database('./checkpoints/best.pt', root+'/'+item)
        for t in prompt:
            txt.append(t)

    with open('./sam/datasets/annotations.txt', 'w') as f:
        for item in txt:
            f.write(item + '\n')

