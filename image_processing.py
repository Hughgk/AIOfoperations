import numpy as np
import queue
from tqdm import tqdm
import cv2
import os
def getListFiles(path):
    #获取目录下文件路径
    ret = []
    for root, dirs, files in os.walk(path):  # 遍历path下文件
        for filespath in files:
            ret.append(os.path.join(root,filespath))   # 将各个文件路径用ret列表保存
    return ret

def get_x_y_cuts(data, n_lines=1):    # 获取各个小图像的位置范围
    # data代表待分割图像的灰度值矩阵，n_lines代表分割图像中符号的行数
    w, h = data.shape    # 图片尺寸
    visited = set()         # 存储已访问过的坐标
    q = queue.Queue()       # 存储一个连通的图像团的坐标
    offset = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]    # 中心周围8连通域坐标
    cuts = []       # 存储分割后的数字图片所在范围坐标
    for y in range(h):              # 逐列扫描
        for x in range(w):
            x_axis = []            # 存储灰度值满足条件的x坐标
            y_axis = []            # 存储灰度值满足条件的坐标
            if data[x][y] < 200 and (x, y) not in visited:      # 判断像素点是否满足条件
                q.put((x, y))
                visited.add((x, y))        # 存储灰度值小于200的已访问像素点坐标
            while not q.empty():        # 遍历同一团图像
                x_p, y_p = q.get()
                for x_offset, y_offset in offset:       # 遍历周围8个像素点
                    x_c, y_c = x_p + x_offset, y_p + y_offset
                    if (x_c, y_c) in visited:       # 判断周围8个像素点是否已访问过
                        continue
                    visited.add((x_c, y_c))
                    try:
                        if data[x_c][y_c] < 200:
                            # 满足条件的同一团像素点入队有并分别记录x和y坐标
                            q.put((x_c, y_c))
                            x_axis.append(x_c)
                            y_axis.append(y_c)
                    except:
                        pass
            if x_axis:
                min_x, max_x = min(x_axis), max(x_axis)         # 记录一个图像团的x起点和终点
                min_y, max_y = min(y_axis), max(y_axis)         # 记录一个图像团的y起点和终点
                if max_x - min_x > 3 and max_y - min_y > 3:
                    # 记录范围大于3X3的图像团所在的块坐标
                    cuts.append([min_x, max_x + 1, min_y, max_y + 1])
    if n_lines == 1:
        cuts = sorted(cuts, key=lambda x: x[2])        # 将cuts列表按y起点坐标升序排序
        pr_item = cuts[0]
        count = 1
        len_cuts = len(cuts)
        new_cuts = [cuts[0]]          # 判断’÷‘符号，也可以防止写数字时部分中断
        pr_k = 0
        for i in range(1, len_cuts):        # 依次与上一个图像团比较
            pr_item = new_cuts[pr_k]
            now_item = cuts[i]
            if not (now_item[2] > pr_item[3]):      # ’÷‘符号和上下中断时将中断数字坐标拼接
                new_cuts[pr_k][0] = min(pr_item[0], now_item[0])
                new_cuts[pr_k][1] = max(pr_item[1], now_item[1])
                new_cuts[pr_k][2] = min(pr_item[2], now_item[2])
                new_cuts[pr_k][3] = max(pr_item[3], now_item[3])
            else:
                new_cuts.append(now_item)
                pr_k += 1
        cuts = new_cuts
    return cuts

def get_image_cuts(image, dir=None, is_data=False, n_lines=1, data_needed=False, count=0):
    # 获取图像中的各个小分割图像
    # image代表待分割图像,dir为图像保存目的路径,n_lines代表分割图像中符号的行数
    if is_data:     # is_data表示image是灰度值矩阵还是一个文件名
        data = image
    else:
        data = cv2.imread(image, 2)     #  读取图片
    cuts = get_x_y_cuts(data, n_lines=n_lines)        # 获取分割后的图片坐标列表
    image_cuts = None
    for i, item in enumerate(cuts):
        count += 1
        max_dim = max(item[1] - item[0], item[3] - item[2])     # 返回x和y方向中最大的像素个数
        new_data = np.ones((int(1.4 * max_dim), int(1.4 * max_dim))) * 255     # 建立一个稍大于手写数字的正方形矩阵区域
        x_min, x_max = (max_dim - item[1] + item[0]) // 2, (max_dim - item[1] + item[0]) // 2 + item[1] - item[0]
        y_min, y_max = (max_dim - item[3] + item[2]) // 2, (max_dim - item[3] + item[2]) // 2 + item[3] - item[2]
        new_data[int(0.2 * max_dim) + x_min:int(0.2 * max_dim) + x_max, int(0.2 * max_dim) + y_min:int(0.2 * max_dim) + y_max] = data[item[0]:item[1], item[2]:item[3]]
        # 将数字图片数据存入new_data中
        standard_data = cv2.resize(new_data, (28, 28))          # 将图片大小设为28x28
        if not data_needed:     # data_needed表示是否需要以数据的形式返回图像的数据集
            cv2.imwrite(dir + str(count) + ".jpg", standard_data)   # 将切割好的图片数据以图片形式保存
        if data_needed:
            data_flat = (255 - np.resize(standard_data, (1, 28 * 28))) / 255        # 以数据方式返回
            if image_cuts is None:
                image_cuts = data_flat
            else:
                image_cuts = np.r_[image_cuts, data_flat]
    if data_needed:
        return image_cuts
    return count        # count为了方便统计分割符号数量的一个parameter，可忽略

def get_images_labels():
    # 获取图片,并且得到图片对应符号的标签
    operators = ['plus', 'sub', 'mul', 'div', '(', ')']
    images = None
    labels = None
    for i, op in enumerate(operators):
        image_file_list = getListFiles('./cfs/' + op + '/')
        print('Loading the ' + op + ' operator...')
        for filename in tqdm(image_file_list):      # 加载图片并显示进度条
            image = cv2.imread(filename, 2)         # 读取图片
            if image.shape != (28, 28):             # 将图片格式设为28x28
                image = cv2.resize(image, (28, 28))
            image = np.resize(image, (1, 28 * 28))  # image格式为1行,28*28列
            image = (255 - image) / 255
            label = np.zeros((1, 10 + len(operators)))  # label格式为1行，0-9+符号
            label[0][10 + i] = 1
            if images is None:
                images = image
                labels = label
            else:
                images = np.r_[images, image]
                labels = np.r_[labels, label]
    return images, labels

