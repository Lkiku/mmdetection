import numpy as np
import random
import cv2
from scipy.special import comb
import xml.etree.ElementTree as ET
import os

class LocationScaleAugmentation(object):
    def __init__(self, vrange=(0., 255.), background_threshold=1.5, nPoints=4, nTimes=100000):
        self.nPoints=nPoints
        self.nTimes=nTimes
        self.vrange=vrange
        self.background_threshold=background_threshold
        self._get_polynomial_array()

    def _get_polynomial_array(self):
        def bernstein_poly(i, n, t):
            return comb(n, i) * (t ** (n - i)) * (1 - t) ** i
        t = np.linspace(0.0, 1.0, self.nTimes)
        self.polynomial_array = np.array([bernstein_poly(i, self.nPoints - 1, t) for i in range(0, self.nPoints)]).astype(np.float32)

    def get_bezier_curve(self,points):
        xPoints = np.array([p[0] for p in points])
        yPoints = np.array([p[1] for p in points])
        xvals = np.dot(xPoints, self.polynomial_array)
        yvals = np.dot(yPoints, self.polynomial_array)
        return xvals, yvals

    def non_linear_transformation(self, inputs, inverse=False, inverse_prop=0.5):
        start_point,end_point=inputs.min(),inputs.max()
        xPoints = [start_point, end_point]
        yPoints = [start_point, end_point]
        for _ in range(self.nPoints-2):
            xPoints.insert(1, random.uniform(xPoints[0], xPoints[-1]))
            yPoints.insert(1, random.uniform(yPoints[0], yPoints[-1]))
        xvals, yvals = self.get_bezier_curve([[x, y] for x, y in zip(xPoints, yPoints)])
        if inverse and random.random()<=inverse_prop:
            xvals = np.sort(xvals)
        else:
            xvals, yvals = np.sort(xvals), np.sort(yvals)
        return np.interp(inputs, xvals, yvals)

    def add_ultrasound_noise(self, image, noise_level=0.1):
        # 确保输入是浮点型
        image = image.astype(np.float32)
        speckle = np.random.normal(0, noise_level, image.shape).astype(np.float32)
        noisy_image = image * (1 + speckle)
        return np.clip(noisy_image, self.vrange[0], self.vrange[1])

    def simulate_attenuation(self, image, alpha=0.1):
        # 确保输入是浮点型
        image = image.astype(np.float32)
        h, w = image.shape[:2]
        # 确保alpha在合理范围内
        alpha = np.clip(alpha, 0, 0.5)
        attenuation_map = np.linspace(1.0, max(1.0 - alpha, 0.5), h)[:, np.newaxis]
        if len(image.shape) == 3:
            attenuation_map = np.repeat(attenuation_map[:, :, np.newaxis], 3, axis=2)
        return np.clip(image * attenuation_map, self.vrange[0], self.vrange[1])

    def location_scale_transformation(self, inputs, slide_limit=30):
        # 增加尺度变化范围以适应超声图像的特点
        scale = np.array(max(min(random.gauss(1, 0.1), 1.1), 0.9), dtype=np.float32)
        # 增加位置变化范围
        location = np.array(random.gauss(0, 0.35), dtype=np.float32)
        location = np.clip(location, self.vrange[0] - np.percentile(inputs, slide_limit), 
                          self.vrange[1] - np.percentile(inputs, 100 - slide_limit))
        return np.clip(inputs*scale + location, self.vrange[0], self.vrange[1])

    def Global_Location_Scale_Augmentation(self, image):
        # 转换为灰度图进行处理
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        # 添加超声特有的噪声
        gray = self.add_ultrasound_noise(gray)
        
        # 对灰度图进行非线性变换和位置尺度变换
        gray = self.non_linear_transformation(gray, inverse=False)
        gray = self.location_scale_transformation(gray)
        
        # 计算变换前后的差值
        diff = gray - cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        # 将差值应用到所有通道
        output = image.astype(np.float32)
        for i in range(image.shape[2]):
            output[:,:,i] = np.clip(output[:,:,i] + diff, self.vrange[0], self.vrange[1])
        
        # 添加衰减效应
        output = self.simulate_attenuation(output)
        
        return output.astype(np.uint8)

    def Local_Location_Scale_Augmentation(self, image, mask):
        # 确保输入数据类型正确
        image = image.astype(np.float32)
        mask = mask.astype(np.int32)
        output_image = np.zeros_like(image, dtype=np.float32)
        original_image = image.copy()

        # 处理空mask的情况
        if np.max(mask) == 0:
            return self.Global_Location_Scale_Augmentation(image)
        
        # 对背景区域进行增强
        background_region = (mask == 0)
        if background_region.any():
            output_image[background_region] = self.location_scale_transformation(
                self.non_linear_transformation(image[background_region], inverse=True, inverse_prop=0.8))
        
        # 对每个目标区域进行增强
        for c in range(1, np.max(mask) + 1):
            current_region = (mask == c)
            if not current_region.any():
                continue
            
            # 创建边界模糊效果
            kernel_size = random.choice([3, 5, 7])
            local_mask = current_region.astype(np.float32)
            blurred_mask = cv2.GaussianBlur(local_mask, (kernel_size, kernel_size), 0)
            
            # 应用增强
            enhanced = self.location_scale_transformation(
                self.non_linear_transformation(image[current_region], inverse=True, inverse_prop=0.3))
            
            # 使用模糊的mask进行混合
            if len(image.shape) == 3:
                blurred_mask = blurred_mask[:, :, np.newaxis]
            
            output_image[current_region] = (
                enhanced * blurred_mask[current_region] + 
                original_image[current_region] * (1 - blurred_mask[current_region])
            )
        
        # 应用背景阈值
        if self.background_threshold >= self.vrange[0]:
            alpha = np.clip((image - self.background_threshold) / 10.0, 0, 1)
            output_image = output_image * alpha + image * (1 - alpha)
        
        # 添加超声特有的噪声
        output_image = self.add_ultrasound_noise(output_image)
        
        return np.clip(output_image, self.vrange[0], self.vrange[1]).astype(np.uint8)

def create_mask_from_bboxes(image_shape, bboxes):
    """
    从检测框创建mask，相同类别的区域具有相同的mask值
    Args:
        image_shape: 图像形状 (H, W)
        bboxes: 检测框列表 [[x1, y1, x2, y2, class_name], ...]
    Returns:
        mask: 与图像同大小的mask，背景为0，每个类别的区域为1,2,3...
    """
    mask = np.zeros(image_shape[:2], dtype=np.int32)
    # 创建类别到ID的映射
    class_to_id = {}
    current_id = 1
    
    for bbox in bboxes:
        x1, y1, x2, y2, class_name = bbox
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        
        # 如果是新类别，分配新的ID
        if class_name not in class_to_id:
            class_to_id[class_name] = current_id
            current_id += 1
            
        mask[y1:y2, x1:x2] = class_to_id[class_name]
    
    return mask

def read_xml_annotations(xml_path):
    """
    从VOC格式的XML文件中读取目标检测框和类别
    Returns:
        bboxes: 检测框列表 [[x1, y1, x2, y2, class_name], ...]
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    bboxes = []
    
    for obj in root.findall('object'):
        bbox = obj.find('bndbox')
        class_name = obj.find('name').text
        x1 = float(bbox.find('xmin').text) - 1
        y1 = float(bbox.find('ymin').text) - 1
        x2 = float(bbox.find('xmax').text) - 1
        y2 = float(bbox.find('ymax').text) - 1
        bboxes.append([x1, y1, x2, y2, class_name])
    
    return bboxes

def mixup_images(img1, img2, alpha=0.5):
    """
    使用Mixup方法融合两张图片
    Args:
        img1: 第一张图片
        img2: 第二张图片
        alpha: 混合权重
    Returns:
        混合后的图片
    """
    return (img1 * alpha + img2 * (1 - alpha)).astype(np.uint8)

if __name__ == "__main__":
    # 读取图像
    
    image_path = "./data/VOCH1/JPEGImages/1.2.156.112601.1.4.960051513.3155.1645669497.123193.jpg"
    xml_path = image_path.replace('JPEGImages', 'Annotations').replace('.jpg', '.xml')
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    # 从XML文件读取检测框
    bboxes = read_xml_annotations(xml_path)
    if not bboxes:
        print(f"警告: 在{xml_path}中没有找到任何检测框")
    
    # 从检测框创建mask
    mask = create_mask_from_bboxes(image.shape, bboxes)
    
    # 创建 LocationScaleAugmentation 对象
    aug = LocationScaleAugmentation()

    # 应用全局增强和局部增强
    image = image.astype(np.float32)
    global_aug_image = aug.Global_Location_Scale_Augmentation(image)
    local_aug_image = aug.Local_Location_Scale_Augmentation(image, mask)
    
    # Mixup融合两种增强结果
    mixed_image = mixup_images(global_aug_image, local_aug_image, alpha=0.5)
    
    # 保存结果
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    print(f"保存结果到 {output_dir}")
    cv2.imwrite(os.path.join(output_dir, "original_image.png"), image)
    cv2.imwrite(os.path.join(output_dir, "global_augmented_image.png"), global_aug_image)
    cv2.imwrite(os.path.join(output_dir, "local_augmented_image.png"), local_aug_image)
    cv2.imwrite(os.path.join(output_dir, "mixed_augmented_image.png"), mixed_image)
    cv2.imwrite(os.path.join(output_dir, "mask.png"), (mask * 50).astype(np.uint8))
