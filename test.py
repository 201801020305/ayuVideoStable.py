import numpy as np
import video_stab_ransac2
import unittest


class MyClassTest(unittest.TestCase):

    def test_video_stable(self):
        # 设置特征点的范围掩膜
        pts_mask = np.zeros([1720, 2304], dtype=np.uint8)
        pts_mask[112:321, 2060:2222] = 255
        pts_mask[523:647, 2200:2304] = 255
        pts_mask[1554:1708, 2120:2259] = 255

        # 去抖(参数说明：图像路径、掩膜、最大特征点数、是否输出图像、是否输出运动参数，输出路径）
        video_stab_ransac2.video_stable(images_path="../images", mask=pts_mask, max_pts=10, is_image_out=1,
                                        is_transform_out=1, transform_path="excels_out/transforms.xlsx")

    def test_draw_all_flow(self):
        # 特征点绘制
        # 参数说明：图像路径、特征点文件路径、起始绘制图像、窗口持续时间
        # P.S.
        #   1.对于特征点文件pts.npy，在运行video_stable稳像函数之后会自动生成，无需更改，路径放在这里是为了防止你误删了这个文件夹
        #   2.delay_time设为0时，窗口持续时间为永久
        video_stab_ransac2.draw_all_flow(images_path="../images_out", pts_data="pts_data/pts.npy",
                                         start_image_name="s_3099.bmp",
                                         delay_time=500)

    def test_psnr(self):
        # 设置平台的范围掩膜
        plat_mask = np.zeros([1720, 2304], dtype=np.uint8)
        plat_mask[0:1720, 2020:2304] = 255
        # 计算psnr（参数说明：图像路径、掩膜、输出文件名）
        video_stab_ransac2.psnr_iamges(images_path="../images_out", mask=plat_mask, out_path="excels_out/psnr_2.xlsx")


if __name__ == "__main__":
    unittest.main()
