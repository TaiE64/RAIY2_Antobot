#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
sys.path.append(os.path.dirname(__file__))  # 让 Python 找到 reid.py

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import cv2
import os
import torch
from torch.nn.functional import cosine_similarity
from ultralytics import YOLO
from reid import ReID  # reid.py

class YOLOVisionNode:
    def __init__(self):
        self.bridge = CvBridge()
        self.model = YOLO('yolov8n.pt')  # 可改为 yolov8s.pt

        # === 初始化 ReID 模块并加载 user.jpg ===
        self.reid = ReID()
        user_path = os.path.join(os.path.dirname(__file__), "user.jpg")
        user_img = cv2.imread(user_path)

        if user_img is None:
            rospy.logerr("❌ Failed to load user.jpg!")
            exit(1)

        self.user_feat = self.reid.extract(user_img)
        rospy.loginfo("✅ user.jpg 特征加载成功")
        
        # === 发布目标人的像素中心点 ===
        self.pub_target_pixel = rospy.Publisher(
            "/target_person/pixel", Point, queue_size=1
        )

        # === 订阅图像 ===
        rospy.Subscriber(
            "/mobile_base/zed2_front/left_raw/image_raw_color",
            Image,
            self.image_callback,
            queue_size=1,
            buff_size=2**24
        )

        rospy.loginfo("✅ YOLO + ReID 节点已启动")

#从图像中检测所有人，用 ReID 比对“最像注册用户”的人，并在图像中只给这个人画框。
    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8') #将 ROS 图像消息转为 OpenCV 的图像格式（BGR），你才能用 cv2 和 YOLO 来处理。
        except Exception as e:
            rospy.logerr("CVBridge error: %s", str(e))
            return

        #只关心人类，类别ID=0）运行 YOLOv8 模型，得到目标检测结果。结果中包含所有检测框（box）、类别（cls）、置信度（conf）等。
        results = self.model(frame, verbose=False)[0]
        #初始化两个变量，用来记录“最像用户的检测框”及其相似度
        best_score = -1
        best_box = None

        for box in results.boxes:#遍历所有检测结果（每个 box 是一个候选目标）
            cls_id = int(box.cls[0])
            if cls_id != 0:
                continue  # 跳过非“人”类目标

            x1, y1, x2, y2 = map(int, box.xyxy[0])#提取检测框的坐标（左上角到右下角）
            person_crop = frame[y1:y2, x1:x2]#把这个人的图像从整帧图中“裁剪”出来作为 person_crop。如果图像为空，跳过。
            if person_crop.size == 0:
                continue

            try:
                feat = self.reid.extract(person_crop)#ReID 模型提取这张“人脸/身体图像”的特征向量。
                score = cosine_similarity(feat, self.user_feat, dim=0).item()#将这个人的特征和注册用户（user.jpg）的特征做 余弦相似度对比。

                if score > best_score:
                    best_score = score#只保留“最像”的目标框（即得分最高者）
                    best_box = (x1, y1, x2, y2)
            except Exception as e:
                rospy.logwarn("ReID 提特征失败: %s", str(e))

        #在原图像上画绿色边框并写上“目标 + 相似度分数”，只画最像注册用户的那一个人。
        if best_box:
            x1, y1, x2, y2 = best_box
            u = int((x1 + x2) / 2)
            v = int((y1 + y2) / 2)
            
            # === 发布像素坐标 ===
            pt = Point()
            pt.x = u
            pt.y = v
            pt.z = 0  # 占位，后续可以加时间戳或ID
            self.pub_target_pixel.publish(pt)
            
            # === 显示图像与框 ===
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Target ({best_score:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
            # ✅ 画十字表示 (u, v)
            cv2.drawMarker(frame, (u, v), (0, 0, 255),  # 红色十字
		       markerType=cv2.MARKER_CROSS,
		       markerSize=10,
		       thickness=2)

        cv2.imshow("YOLO + ReID", frame)
        cv2.waitKey(1)

if __name__ == '__main__':
    rospy.init_node('yolo_reid_node', anonymous=True)
    try:
        YOLOVisionNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()

