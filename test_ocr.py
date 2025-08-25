from paddleocr import PaddleOCR
from math import floor, ceil
import cv2
import numpy as np


def ocr_predict(language, image_path):
    # You can set the parameter `lang` as `ch`, `en`, `french`, `german`, `korean`, `japan`
    ocr = PaddleOCR(use_angle_cls=True, lang=language)  # need to run only once to download and load model into memory
    img_path = image_path
    # result = ocr.ocr(img_path, cls=True)
    result = ocr.ocr(img_path)

    prompts = []
    for idx in range(len(result)):
        res = result[idx]
        for line in res:
            prompt = f"{line[1][0]} {line[0][0][0]} {line[0][0][1]} {line[0][2][0]} {line[0][2][1]}"
            prompts.append(prompt)
            print(prompt)
    return prompts


def improved_ocr(language, image_path, yolo_prompt):
    ocr = PaddleOCR(use_angle_cls=True, lang=language)  # need to run only once to download and load model into memory

    img = cv2.imread(image_path)
    img_list = []  # 裁剪图片列表
    prompt_list = []  # prompt列表
    idx_list = []  # 裁剪图片原序号列表

    cnt = -1
    for prompt in yolo_prompt:
        if not (prompt.startswith("顺序流") or prompt.startswith("消息流") or prompt.startswith("数据关联")):
            cnt += 1
        if prompt.startswith("任务") or prompt.startswith("标签"):
            idx_list.append(cnt)
            prompt_s = prompt.split(" ")
            try:
                img_list.append(img[floor(float(prompt_s[2])):ceil(float(prompt_s[4])), floor(float(prompt_s[1])):ceil(float(prompt_s[3]))].copy())
            except Exception:
                img_list.append(img[float(prompt_s[2]):float(prompt_s[4]), float(prompt_s[1]):float(prompt_s[3])].copy())
            prompt_list.append([prompt_s[1], prompt_s[2], prompt_s[3], prompt_s[4]])

    prompts = []
    object2 = []  # 序号+识别文本内容，用于socket返回
    for idx in range(len(img_list)):
        # result = ocr.ocr(img_list[idx], cls=True)
        result = ocr.ocr(img_list[idx])
        res = result[0]
        reply = ""
        for line in res:
            reply += f"{line[1][0]} "
        prompt = f"{reply} {prompt_list[idx][0]} {prompt_list[idx][1]} {prompt_list[idx][2]} {prompt_list[idx][3]}"
        prompts.append(prompt)
        object2.append("{}_{}".format(idx_list[idx], reply))
        print(prompt)
    return prompts, object2


if __name__ == '__main__':
    yolo_prompt = ["任务 1083.9 678.7 1381.8 815.1", "任务 1525.8 660.1 1810.9 808.9"]
    # ocr_predict("en", "./en_任务 1525.8 660.1 1810.9 808.9.png")
    print(improved_ocr("en", "./sample/0003.png", yolo_prompt))