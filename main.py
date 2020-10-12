import cv2
from lane_detection import LaneDetection

image_file = "D:/AutoDrive/0928/start.jpg"
video_file = "D:/AutoDrive/0928/raw.mp4"
output_file = "D:/AutoDrive/0928/out.mp4"

# 创建车道线检测对象
lane_det = LaneDetection(video_file, output_file)
img = cv2.imread(image_file)

# lane_det.gray_scale(image_file)  # 输出灰度图
# lane_det.binarization(image_file)  # 输出二值化图
# blur_gray = lane_det.gaussian_blur(image_file, show=False)  # 输出高斯平滑图
# edges = lane_det.edge_detect(blur_gray, show=False)  # 输出边界检测图，需要用到高斯平滑图结果
# roi_edges = lane_det.roi(edges, show=True)  # 输出兴趣区域截取后的边界，需要用到边界检测图
# lane_det.hough_transform(roi_edges, show=True)  # 输出霍夫变换图，需要用到兴趣区域边界

# cv2.imwrite("D:/AutoDrive/0928/start_edges.jpg", edges)  # 将图片输出为文件

# res_img = lane_det.process_an_image(img, show=True)  # 处理一张图片
lane_det.process_a_video()  # 处理一个视频
# lane_det.generate_video()  # 输出视频


