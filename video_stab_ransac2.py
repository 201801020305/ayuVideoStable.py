# 此版本是:获取相邻帧之间的光流,得到dx dy 和dz,累加后得到三个运动曲线
# 利用三个曲线与0(原本应是平滑后的曲线)的差值抵消平台的运动
# 第一次检验方法:从新采集稳像后的图像的关键帧.对比前后视频的运动曲线的差别
# 结果发现, 稳像后的视频,运动曲线计算为0,这是检验方法和算法原理相同导致的,要采用其他方法
# 第二次检验:利用PSNR
import math
import cv2
import numpy as np
import os
import pandas as pd


def data_to_excel(data_name, output_name):
    data_df = pd.DataFrame(data_name)
    with pd.ExcelWriter(output_name) as writer:
        data_df.to_excel(writer, sheet_name='page_1', float_format='%.2f')


def psnr_iamges(images_path, mask, out_path):
    def psnr(img1, img2):
        mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
        if mse < 1.0e-10:
            return 100
        PIXEL_MAX = 1
        return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

    # 打印提示
    print("正在进行psnr计算...")
    # 获取文件夹下的文件名数组并排序
    file_names_arr = os.listdir(images_path)
    file_names_arr.sort(key=lambda x: int(x.replace("s_", "").split('.')[0]))
    # 获取图片数量
    image_count = len(file_names_arr)
    # 定义存储psnr的矩阵
    psnr_arr = np.zeros((image_count - 1, 1), np.float32)
    for i in range(image_count - 1):
        # 得到前一张图片
        pre_im_name = file_names_arr[i]
        pre_im_path = os.path.join(images_path, pre_im_name)
        pre_image = cv2.imread(pre_im_path)
        pre_image_gray = cv2.cvtColor(pre_image, cv2.COLOR_BGR2GRAY)
        pre_image_gray_masked = cv2.bitwise_and(pre_image_gray, mask)
        # 得到后一张图片
        curr_im_name = file_names_arr[i + 1]
        curr_im_path = os.path.join(images_path, curr_im_name)
        curr_im_name = cv2.imread(curr_im_path)
        curr_im_name = cv2.cvtColor(curr_im_name, cv2.COLOR_BGR2GRAY)
        curr_image_gray_masked = cv2.bitwise_and(curr_im_name, mask)
        # 计算psnr存入矩阵
        psnr_arr[i] = psnr(pre_image_gray_masked, curr_image_gray_masked)

        # 输出PSNR矩阵到excel
        data_to_excel(psnr_arr, out_path)
    print("输出数据到" + out_path)


def draw_all_flow(images_path, pts_data, start_image_name, delay_time):
    # 获取文件夹下的文件名数组并排序
    file_names_arr = os.listdir(images_path)
    file_names_arr.sort(key=lambda x: int(x.replace("s_", "").split('.')[0]))
    # 加载特征点文件
    pts_arr = np.load(pts_data)
    # 得到起始索引
    start_index = file_names_arr.index(start_image_name)

    for i in range(len(file_names_arr) - start_index):
        # 得到第一张图片
        first_im_name = file_names_arr[0]
        first_im_path = os.path.join(images_path, first_im_name)
        first_image = cv2.imread(first_im_path)
        # 得到目标图片
        curr_im_path = os.path.join(images_path, file_names_arr[i + start_index])
        curr_image = cv2.imread(curr_im_path)
        # 得到特征点对
        many_pts_couples = pts_arr[i + start_index - 1, :, :, :]
        # 画图
        for j in range(many_pts_couples.shape[0]):
            # 第一帧画点
            cv2.circle(first_image, (many_pts_couples[j][0][0], many_pts_couples[j][0][1]), 2, (255, 255, 255), 2)
            # 当前帧两批点都画上
            cv2.circle(curr_image, (many_pts_couples[j][0][0], many_pts_couples[j][0][1]), 2, (255, 255, 255), 2)
            cv2.circle(curr_image, (many_pts_couples[j][1][0], many_pts_couples[j][1][1]), 2, (0, 0, 255), 2)

        # 把画完圆的图片缩小一下,再输出查看
        first_image = cv2.resize(first_image, (1000, 800))
        curr_image = cv2.resize(curr_image, (1000, 800))
        # 再裁剪一下
        # 定义要裁剪的左侧和右侧的像素数量
        left_cut = 600

        # 进行裁剪
        first_image = first_image[:, left_cut:1000]
        curr_image = curr_image[:, left_cut:1000]
        # 再合并成一张图
        # 创建一个新的黑色背景图像
        new_img = np.zeros((max(800, 800), 800, 3), np.uint8)

        # 将两张图像拼接到新图像上
        new_img[:800, :400, :] = first_image
        new_img[:800, 400:800, :] = curr_image
        # cv2.imshow("First image", first_image)
        cv2.imshow("The " + str(start_index + i + 1) + "th image(red=current ; white=first)", new_img)
        cv2.waitKey(delay_time)  # 等待用户操作，里面等待参数是毫秒，我们填写0，代表是永远，等待用户操作
        if i == len(file_names_arr) - start_index - 1:
            cv2.waitKey(0)
        cv2.destroyAllWindows()  # 销毁所有窗口


def video_stable(images_path, mask, max_pts, is_image_out, is_transform_out, transform_path):
    # 获取文件夹下的文件名数组并排序
    file_names_arr = os.listdir(images_path)
    file_names_arr.sort(key=lambda x: int(x.replace("s_", "").split('.')[0]))

    # 获取图片数量
    image_count = len(file_names_arr)

    # 预定义改正矩阵
    transforms = np.zeros((image_count - 1, 3), np.float32)

    # 预定义特征点对的矩阵
    pts_arr = np.zeros((image_count - 1, max_pts, 2, 2), np.float32)

    # 得到第一张图片
    first_im_name = file_names_arr[0]
    first_im_path = os.path.join(images_path, first_im_name)
    first_image = cv2.imread(first_im_path)
    first_image_gray = cv2.cvtColor(first_image, cv2.COLOR_BGR2GRAY)
    # 得到图片的宽度和高度
    w = first_image.shape[1]
    h = first_image.shape[0]
    # 找到第一帧的特征点,找的时候用mask2掩膜，即只找数字12附近的特征点
    first_pts = cv2.goodFeaturesToTrack(first_image_gray,
                                        maxCorners=max_pts,
                                        qualityLevel=0.01,
                                        minDistance=30,
                                        mask=mask,
                                        blockSize=3)

    # 稳像前的循环,此循环遍历除了第一帧以外的所有帧，检测了特征点并通过光流计算出运动曲线
    for i in range(image_count - 1):
        # 读取当前帧（从第二帧开始）
        curr_image_path = os.path.join(images_path, file_names_arr[i + 1])
        curr_image = cv2.imread(curr_image_path)
        # 转成灰度图
        curr_image_gray = cv2.cvtColor(curr_image, cv2.COLOR_BGR2GRAY)
        # 计算光流,得到第一帧特征点对应的当前帧上的特征点 (i.e. track feature points)
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(first_image_gray, curr_image_gray, first_pts, None)
        # 可靠性检验
        assert first_pts.shape == curr_pts.shape

        # 筛出有效点对
        idx = np.where(status == 1)[0]
        first_pts = first_pts[idx]
        curr_pts = curr_pts[idx]
        # 用Ransac再次筛选
        F, mask = cv2.findFundamentalMat(first_pts, curr_pts, cv2.RANSAC, 0.1)

        # 筛选过的点对坐标
        first_pts_filtered = first_pts[mask.ravel() == 1]
        curr_pts_filtered = curr_pts[mask.ravel() == 1]

        # 存入特征点矩阵
        for j in range(first_pts_filtered.shape[0]):
            pts_arr[i, j, 0, :] = first_pts_filtered[j, 0, :]
            pts_arr[i, j, 1, :] = curr_pts_filtered[j, 0, :]
        # 得到变换矩阵m
        m = cv2.estimateRigidTransform(first_pts_filtered, curr_pts_filtered,
                                       fullAffine=False)
        # 提取坐标改变量dx和dy
        dx = m[0, 2]
        dy = m[1, 2]

        # 提取角度改变量da
        da = np.arctan2(m[1, 0], m[0, 0])

        # 组成运动矩阵
        transforms[i] = [dx, dy, da]

        # 打印进度
        print("改正矩阵已写入第 " + str(i + 1) + "/" + str(image_count - 1) + " 行- 特征点对数 : " + str(
            len(first_pts_filtered)))

    # 输出特征点对的二进制文件备用
    np.save("pts_data/pts.npy", pts_arr)
    # 把运动矩阵取负作为图像改正矩阵
    transforms_smooth = 0 - transforms

    # 这是第二个循环,稳像过程就在这个循环中进行
    for i in range(image_count - 1):
        # 判断是否需要输出图像，不需要则循环跳过
        if is_image_out is 0:
            break
        # 读取下一张图片
        curr_image_path = os.path.join(images_path, file_names_arr[i + 1])
        curr_image = cv2.imread(curr_image_path)

        # 取出改正矩阵中的数据，制作当前帧的仿射变换参数
        dx = transforms_smooth[i, 0]
        dy = transforms_smooth[i, 1]
        da = transforms_smooth[i, 2]

        m = np.zeros((2, 3), np.float32)
        m[0, 0] = np.cos(da)
        m[0, 1] = -np.sin(da)
        m[1, 0] = np.sin(da)
        m[1, 1] = np.cos(da)
        m[0, 2] = dx
        m[1, 2] = dy

        # 对当前帧进行仿射变换
        image_stabilized = cv2.warpAffine(curr_image, m, (w, h))
        # 输出图片
        cv2.imwrite("images_out/s_" + str(3001 + i) + ".bmp", image_stabilized)
        # 打印进度
        print("输出第" + str(i + 2) + "张图像")

    # 以下是附加功能，根据参数指定是否运行
    # 输出运动矩阵的excel
    if is_transform_out:
        data_to_excel(transforms, transform_path)
        print("已输出运动矩阵图表,路径：" + transform_path)
