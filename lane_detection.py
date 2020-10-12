import cv2
import numpy as np
from moviepy.editor import VideoFileClip


class LaneDetection():
    """简单车道线检测算法"""

    def __init__(self, video_file, output_file):
        """储存文件路径和其他参数"""
        # 视频路径
        self.video_file = video_file
        self.output_file = output_file
        # 读取图片
        cap = cv2.VideoCapture(self.video_file)
        ret, self.img = cap.read()
        # 高斯平滑核大小
        self.blur_ksize = 5
        # Canny算子的上下阈值
        self.canny_lthreshold = 50
        self.canny_hthreshold = 150
        # 兴趣区域顶点坐标
        self.roi_vtx = np.array([[(0, self.img.shape[0]), (0, 400), (100,360), (1000, 360),
                                  (self.img.shape[1], 400), (self.img.shape[1], self.img.shape[0])]])

        # hough转换参数
        self.rho = 1  # 线段以像素为单位的距离精度（1像素）
        self.theta = np.pi / 180  # 线段以弧度为单位的角度精度
        self.threshold = 50  # 累加平面的阈值参数，int类型，超过设定阈值才被检测出线段
        self.min_line_length = 50  # 线段以像素为单位的最小长度
        self.max_line_gap = 10  # 同一方向上两条线段判定为一条线段的最大允许间隔
        self.slope_lower_threshold = 0.15 # 斜率下限
        self.slope_upper_threshold = 1  # 斜率上限
        self.power = 2  # 曲线拟合幂次数

    def gray_scale(self, image_file):
        """图像灰度处理"""
        img = cv2.imread(image_file)
        new = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        self.show_image({"img": self.img, "gray scale": new})

    def binarization(self, image_file):
        """图像二值化处理"""
        img = cv2.imread(image_file)
        sp = img.shape
        height = sp[0]
        width = sp[1]
        new = np.zeros(sp, np.uint8)
        for i in range(height):
            for j in range(width):
                new[i, j] = max(img[i, j][0], img[i, j][1], img[i, j][2])

        ret, a_new = cv2.threshold(new, 127, 255, cv2.THRESH_BINARY)
        ret, b_new = cv2.threshold(new, 127, 255, cv2.THRESH_BINARY_INV)

        self.show_image({"A1": self.img, "A2": new, "A3": a_new, "A4": b_new})

    def gaussian_blur(self, image_file, show=False):
        """高斯平滑处理"""
        img = cv2.imread(image_file)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        blur_gray = cv2.GaussianBlur(gray, (self.blur_ksize, self.blur_ksize), 0, 0)
        if show:
            self.show_image({"gray": gray, "blur gray": blur_gray})

        return blur_gray

    def edge_detect(self, blur_gray, show=False):
        """Canny算子边缘检测"""
        edges = cv2.Canny(blur_gray, self.canny_lthreshold, self.canny_hthreshold)
        if show:
            self.show_image({"blur gray": blur_gray, "Canny": edges})

        return edges

    def roi(self, edges, show=False):
        """截取兴趣区域"""
        roi_edges = self.roi_mask(edges, self.roi_vtx, show)
        if show:
            self.show_image({"edges": edges, "roi_edges": roi_edges})

        return roi_edges

    def roi_mask(self, img, vertices, show):
        """掩膜,用于截取兴趣区域roi"""
        mask = np.zeros_like(img)
        mask_color = 255  # 代表白色

        cv2.fillPoly(mask, vertices, mask_color)
        if show:
            self.show_image({"mask": mask})

        masked_img = cv2.bitwise_and(img, mask)
        return masked_img

    def draw_lines(self, img, lines, color=[255, 0, 0], thinckness=2):
        """画出霍夫变换后的多条直线"""
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thinckness)

    def hough_transform(self, roi_edges, show=False):
        """Hough变换"""
        line_img = self.hough_lines(roi_edges, self.rho, self.theta,
                                    self.threshold, self.min_line_length, self.max_line_gap)
        if show:
            self.show_image({"roi_edges": roi_edges, "line_img": line_img})

    def hough_lines(self, img, rho, theta, threshold, min_line_len, max_line_gap):
        lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                                maxLineGap=max_line_gap)
        line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        # self.draw_lines(line_img, lines)
        self.draw_lanes(line_img, lines)
        return line_img

    def draw_lanes(self, img, lines, color=(255, 255, 0), thickness=5):
        left_lines, right_lines = [], []  # 用于存储左边和右边的直线
        if lines is not None:
            for line in lines:  # 对直线进行分类
                for x1, y1, x2, y2 in line:
                    k = (y2 - y1) / (x2 - x1)
                    if k < 0:
                        left_lines.append(line)
                    else:
                        right_lines.append(line)

        cleaned_line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        if left_lines:
            left_lines = self.clean_lines(left_lines, 0.1)  # 弹出左侧不满足斜率要求的直线
            # self.draw_lines(img, left_lines, color=(255, 0, 255))
        if left_lines:
            left_points = [(x1, y1) for line in left_lines for x1, y1, x2, y2 in line]  # 提取左侧直线族中的所有的第一个点
            left_points = left_points + [(x2, y2) for line in left_lines for x1, y1, x2, y2 in line]  # 提取左侧直线族中的所有的第二个点
            left_vtx = self.calc_lane_vertices(left_points, 325, img.shape[0])  # 拟合点集，生成直线表达式，并计算左侧直线在图像中的两个端点的坐标
            # 画直线
            # cv2.line(img, left_vtx[0], left_vtx[1], color, thickness)  # 画出直线

            # 画多项式曲线
            cv2.polylines(img, [left_vtx], False, color, thickness)  # 参数：图片，点，封闭，颜色，厚度

        if right_lines:
            right_lines = self.clean_lines(right_lines, 0.1)  # 弹出右侧不满足斜率要求的直线
            # self.draw_lines(img, right_lines, color=(255, 0, 255))
        if right_lines:
            right_points = [(x1, y1) for line in right_lines for x1, y1, x2, y2 in line]  # 提取右侧直线族中的所有的第一个点
            right_points = right_points + [(x2, y2) for line in right_lines for x1, y1, x2, y2 in line]  # 提取右侧侧直线族中的所有的第二个点
            right_vtx = self.calc_lane_vertices(right_points, 325, img.shape[0])  # 拟合点集，生成直线表达式，并计算右侧直线在图像中的两个端点的坐标
            # 画直线
            # cv2.line(img, right_vtx[0], right_vtx[1], color, thickness)  # 画出直线

            # 画多项式曲线
            cv2.polylines(img, [right_vtx], False, color, thickness)

    def clean_lines(self, lines, threshold):
        """将不满足条件的直线弹出"""
        # 将斜率绝对值过小和过大的直线弹出
        slope = [(y2 - y1) / (x2 - x1) for line in lines for x1, y1, x2, y2 in line]
        idx = [slope.index(a) for a in slope if self.slope_lower_threshold < abs(a) < self.slope_upper_threshold]
        valid_slope = []
        valid_lines = []
        for i in idx:
            valid_slope.append(slope[i])
            valid_lines.append(lines[i])

        while len(valid_lines) > 0:
            mean = np.mean(valid_slope)  # 计算斜率的平均值，因为后面会将直线和斜率值弹出
            diff = [abs(s - mean) for s in valid_slope]  # 计算每条直线斜率与平均值的差值
            idx = np.argmax(diff)  # 计算差值的最大值的下标
            if diff[idx] > threshold:  # 将差值大于阈值的直线弹出
                valid_slope.pop(idx)  # 弹出斜率
                valid_lines.pop(idx)  # 弹出斜率
            else:
                break

        return valid_lines

    def calc_lane_vertices(self, point_list, ymin, ymax):
        """拟合点集，生成直线表达式，并计算直线在图像中的两个端点的坐标 """
        x = [p[0] for p in point_list]  # 提取x
        y = [p[1] for p in point_list]  # 提取y
        # 方法一：一次直线拟合
        # fit = np.polyfit(y, x, 1)  # 用一次多项式x=a*y+b拟合这些点，1表示多项式次数，返回值fit是参数(a,b)
        # fit_fn = np.poly1d(fit)  # 生成多项式对象a*y+b
        #
        # xmin = int(fit_fn(ymin))  # 计算这条直线在图像中最左侧的横坐标
        # xmax = int(fit_fn(ymax))  # 计算这条直线在图像中最右侧的横坐标
        # return [(xmin, ymin), (xmax, ymax)]

        # 方法二：多次曲线拟合
        fit = np.polyfit(y, x, self.power) # 选择拟合曲线次数（还是一次效果最好）
        fit_fn = np.poly1d(fit)
        draw_y = np.arange(min(y), max(y), 1).astype(int)
        draw_x = fit_fn(draw_y).astype(int)
        draw_points = np.asarray([draw_x, draw_y]).T.astype(np.int32)

        return draw_points



    def process_an_image(self, img, show=False):
        """处理图片的完整流程"""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # 灰值
        blur_gray = cv2.GaussianBlur(gray, (self.blur_ksize, self.blur_ksize), 0, 0)  # 高斯平滑
        edges = cv2.Canny(blur_gray, self.canny_lthreshold, self.canny_hthreshold)  # canny边缘
        roi_edges = self.roi_mask(edges, self.roi_vtx, False)  # 兴趣区间
        line_img = self.hough_lines(roi_edges, self.rho, self.theta, self.threshold,
                                    self.min_line_length, self.max_line_gap)  # hough变换
        res_img = cv2.addWeighted(img, 0.8, line_img, 1, 0)  # 加到原图上
        if show:
            self.show_image({'processed image': res_img})

        return res_img

    def process_a_video(self):
        # 实时显示
        cap = cv2.VideoCapture(self.video_file)  # 指定路径读取视频。如果cv2.VideoCapture(0)，没有指定路径，则从电脑自带摄像头取视频。
        ret = True
        # ret,frame = cap.read()
        while (ret):
            ret, frame = cap.read()  # 按帧读取视频，它的返回值有两个：ret, frame。其中ret是布尔值，如果读取帧是正确的则返回 True，如果文件读取到结尾，它的返回值就为False。frame就是每一帧的图像，是个三维矩阵。
            if ret == True:
                #cv2.imshow('Imge', frame)  # 播放视频，第一个参数是视频播放窗口的名称，第二个参数是视频的当前帧。
                res_img = self.process_an_image(frame, False)
                cv2.imshow('AQ', res_img)
            k = cv2.waitKey(100)  # 每一帧的播放时间，毫秒级,该参数可以根据显示速率调整
            if (k & 0xff == ord('q')):  # 如果中途想退出，q键退出，或播放完后，按任意键退出
                cap.release()
                cv2.destroyAllWindows()  # 释放对象和销毁窗口
                break

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def generate_video(self):
        """输出生成视频"""
        clip = VideoFileClip(self.video_file)  # input video
        out_clip = clip.fl_image(self.process_an_image)  # 对视频的每一帧进行处理
        out_clip.write_videofile(self.output_file, audio=True)  # 将处理后的视频写入新的视频文件

    def show_image(self, images):
        """
        显示图像，并关闭图像窗口
        images以字典形式储存，键为窗口名称，值为图片数据
        """
        for image in images.items():   # 遍历字典中每个要显示的图片
            cv2.imshow(image[0], image[1])

        cv2.waitKey(0)
        cv2.destroyAllWindows()