# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import cv2
import time
import numpy as np
from PIL import ImageGrab
from pymouse import PyMouse

#导入与训练YOLOv3模型
net = cv2.dnn.readNet('best.weights','best.cfg')
# 获取三个尺度输出层的名称
layersNames = net.getLayerNames()
output_layers_names = ['yolo_46','yolo_58','yolo_70']

# 导入数据集个类别
with open('best.names','r') as f:
	classes = f.read().splitlines()
CONF_THRES = 0.6 # 指定置信度阈值，阈值越大，置信度过滤越强
NMS_THRES = 0.4 # 指定NMS阈值，阈值越小、NMS越强

def process_frame(img):
	# 获取图像的宽高
	height,width,_ = img.shape
	#对图像预处理
	blob = cv2.dnn.blobFromImage(img,1/255,(416,416),(0,0,0),swapRB = True,crop = False)
	net.setInput(blob)
	# 输入YOLOv3神经网络，前向推断预测
	prediction = net.forward(output_layers_names)
	# 从三个尺度输出结果中解析所有预测框信息
	# 存放预测框坐标
	boxes = []
	# 存放置信度
	objectness = []
	# 存放类别概率
	class_probs = []
	# 存放预测框类别索引号
	class_ids = []
	# 存放预测框类别名称
	class_names = []

	for scale in prediction: # 遍历三种尺度
		for bbox in scale: # 遍历每个预测框
			obj = bbox[4] # 获取该预测框的confidence(objectness)
			class_scores = bbox[5:] # 获取该预测框在COCO数据集80个类别的概率
			class_id = np.argmax(class_scores) # 获取概率最高类别的索引号
			class_name = classes[class_id] # 获取概率最高类别的名称
			class_prob = class_scores[class_id] # 获取概率最高类别的概率

			# 获取预测框中心点坐标，预测框宽高
			center_x = int(bbox[0] * width)
			center_y = int(bbox[1] * height)
			w = int(bbox[2] * width)
			h = int(bbox[3] * height)

			# 计算预测框左上角坐标
			x = int(center_x - w/2)
			y = int(center_y - h/2)

			# 将每个预测框的结果存放至上面的列表中
			boxes.append([x, y, w, h])
			objectness.append(float(obj))
			class_ids.append(class_id)
			class_names.append(class_name)
			class_probs.append(class_prob)
	confidences = np.array(class_probs) * np.array(objectness)
	indexes = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRES, NMS_THRES)
	print(indexes)    
    #indexes.flatten()
    #len(indexes.flatten())
    

	# 遍历留下的每一个预测框，可视化
	for i in indexes:
		# 获取坐标
		x, y, w, h = boxes[i]
		break
	return int(525+(x+w)/2),int(329+(y+h)/2)

def capture(left, top, right, bottom):
    img = ImageGrab.grab(bbox=(left, top, right, bottom))
    img = np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0], 3)
    r, g, b = cv2.split(img)
    cv2.merge([b, g, r], img)
    return img



if __name__ == '__main__':
    mouse = PyMouse()
    # 无限循环，直到break被触发
    while True:
    	# 获取画面
        #525,239 942,633
        frame = capture(525, 329, 942, 633)
        start_time = time.time()
    	# !!!处理帧函数
        x,y = process_frame(frame)
        mouse.move(int(x), int(y))
        #三秒截屏一次
        time.sleep(3)
        if cv2.waitKey(1) in [ord('q'), 27]: # 按键盘上的q或esc退出
            break