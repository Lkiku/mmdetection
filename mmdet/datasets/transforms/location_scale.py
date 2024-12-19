import numpy as np
import random
import cv2
from scipy.special import comb

from mmdet.registry import TRANSFORMS
from mmcv.transforms import BaseTransform

@TRANSFORMS.register_module()
class LocationScaleAugmentation(BaseTransform):
    """Location Scale Augmentation for ultrasound images.
    
    Args:
        vrange (tuple[float]): Value range for image. Defaults to (0., 255.).
        background_threshold (float): Threshold for background. Defaults to 1.5.
        nPoints (int): Number of points for bezier curve. Defaults to 4.
        nTimes (int): Number of times for interpolation. Defaults to 100000.
        prob (float): Probability of applying the transform. Defaults to 1.0.
        global_prob (float): Probability of applying global augmentation. Defaults to 0.5.
        local_prob (float): Probability of applying local augmentation. Defaults to 0.5.
    """

    def __init__(self,
                 vrange=(0., 255.),
                 background_threshold=1.5,
                 nPoints=4,
                 nTimes=100000,
                 prob=1.0,
                 global_prob=0.5,
                 local_prob=0.5):
        super().__init__()
        self.vrange = vrange
        self.background_threshold = background_threshold
        self.nPoints = nPoints
        self.nTimes = nTimes
        self.prob = prob
        self.global_prob = global_prob
        self.local_prob = local_prob
        self._get_polynomial_array()

    def _get_polynomial_array(self):
        """Initialize the polynomial array for bezier curve."""
        def bernstein_poly(i, n, t):
            return comb(n, i) * (t ** (n - i)) * (1 - t) ** i
        t = np.linspace(0.0, 1.0, self.nTimes)
        self.polynomial_array = np.array(
            [bernstein_poly(i, self.nPoints - 1, t) for i in range(0, self.nPoints)]
        ).astype(np.float32)

    def get_bezier_curve(self, points):
        """Get bezier curve from control points."""
        xPoints = np.array([p[0] for p in points])
        yPoints = np.array([p[1] for p in points])
        xvals = np.dot(xPoints, self.polynomial_array)
        yvals = np.dot(yPoints, self.polynomial_array)
        return xvals, yvals

    def non_linear_transformation(self, inputs, inverse=False, inverse_prop=0.5):
        """Apply non-linear transformation using bezier curve."""
        start_point, end_point = inputs.min(), inputs.max()
        xPoints = [start_point, end_point]
        yPoints = [start_point, end_point]
        for _ in range(self.nPoints-2):
            xPoints.insert(1, random.uniform(xPoints[0], xPoints[-1]))
            yPoints.insert(1, random.uniform(yPoints[0], yPoints[-1]))
        xvals, yvals = self.get_bezier_curve([[x, y] for x, y in zip(xPoints, yPoints)])
        if inverse and random.random() <= inverse_prop:
            xvals = np.sort(xvals)
        else:
            xvals, yvals = np.sort(xvals), np.sort(yvals)
        return np.interp(inputs, xvals, yvals)

    def add_ultrasound_noise(self, image, noise_level=0.1):
        """Add ultrasound-specific speckle noise."""
        image = image.astype(np.float32)
        speckle = np.random.normal(0, noise_level, image.shape).astype(np.float32)
        noisy_image = image * (1 + speckle)
        return np.clip(noisy_image, self.vrange[0], self.vrange[1])

    def simulate_attenuation(self, image, alpha=0.1):
        """Simulate ultrasound attenuation effect."""
        image = image.astype(np.float32)
        h, w = image.shape[:2]
        alpha = np.clip(alpha, 0, 0.5)
        attenuation_map = np.linspace(1.0, max(1.0 - alpha, 0.5), h)[:, np.newaxis]
        if len(image.shape) == 3:
            attenuation_map = np.repeat(attenuation_map[:, :, np.newaxis], 3, axis=2)
        return np.clip(image * attenuation_map, self.vrange[0], self.vrange[1])

    def location_scale_transformation(self, inputs, slide_limit=30):
        """Apply location and scale transformation."""
        scale = np.array(max(min(random.gauss(1, 0.1), 1.1), 0.9), dtype=np.float32)
        location = np.array(random.gauss(0, 0.35), dtype=np.float32)
        location = np.clip(location, 
                          self.vrange[0] - np.percentile(inputs, slide_limit),
                          self.vrange[1] - np.percentile(inputs, 100 - slide_limit))
        return np.clip(inputs*scale + location, self.vrange[0], self.vrange[1])

    def Global_Location_Scale_Augmentation(self, image):
        """Apply global augmentation to the image."""
        # 保存原始形状
        original_shape = image.shape
        
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
            
        # 确保数值范围正确
        gray = gray.astype(np.float32)
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        
        # 添加超声特有的噪声
        gray = self.add_ultrasound_noise(gray)
        
        # 对灰度图进行非线性变换和位置尺度变换
        gray = self.non_linear_transformation(gray, inverse=False)
        gray = self.location_scale_transformation(gray)
        
        # 如果是彩色图像，将变换应用到每个通道
        if len(original_shape) == 3:
            output = image.astype(np.float32)
            original_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
            diff = gray - original_gray
            
            # 对每个通道应用相同的变换
            for i in range(original_shape[2]):
                output[:,:,i] = np.clip(output[:,:,i] + diff, 0, 255)
                
            # 添加衰减效应
            output = self.simulate_attenuation(output)
        else:
            # 对于灰度图直接使用处理后的结果
            output = gray
            output = self.simulate_attenuation(output)
            
        return output.astype(np.uint8)

    def Local_Location_Scale_Augmentation(self, image, mask):
        """Apply local augmentation to the image using mask."""
        # 保存原始形状
        original_shape = image.shape
        
        # 转换为灰度图进行处理
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            is_color = True
        else:
            gray = image.copy()
            is_color = False
            
        # 确保数值范围正确
        gray = gray.astype(np.float32)
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        
        # 创建输出图像
        if is_color:
            output = image.astype(np.float32)
        else:
            output = gray.copy()
            
        # 处理空mask的情况
        if np.max(mask) == 0:
            return self.Global_Location_Scale_Augmentation(image)
            
        # 创建全局mask用于平滑过渡
        global_mask = np.zeros_like(gray, dtype=np.float32)
        
        # 对每个区域进行增强
        for c in range(np.max(mask) + 1):  # 包含背景(0)和所有目标
            current_region = (mask == c)
            if not current_region.any():
                continue
                
            # 创建模糊的mask
            kernel_size = random.choice([3, 5, 7])
            local_mask = current_region.astype(np.float32)
            blurred_mask = cv2.GaussianBlur(local_mask, (kernel_size, kernel_size), 0)
            
            # 对当前区域进行增强
            region_gray = gray.copy()
            region_gray[~current_region] = 0
            
            # 应用变换
            enhanced = self.non_linear_transformation(region_gray[current_region], 
                                                   inverse=(c > 0), 
                                                   inverse_prop=0.3 if c > 0 else 0.8)
            enhanced = self.location_scale_transformation(enhanced)
            
            # 更新全局mask
            global_mask[current_region] = blurred_mask[current_region]
            
            # 应用增强
            if is_color:
                diff = enhanced - gray[current_region]
                for i in range(original_shape[2]):
                    output[:,:,i][current_region] = np.clip(
                        image[:,:,i][current_region] + diff * blurred_mask[current_region],
                        0, 255
                    )
            else:
                output[current_region] = np.clip(
                    enhanced * blurred_mask[current_region] + 
                    gray[current_region] * (1 - blurred_mask[current_region]),
                    0, 255
                )
        
        # 添加超声特有的噪声
        output = self.add_ultrasound_noise(output)
        
        # 添加衰减效应
        output = self.simulate_attenuation(output)
        
        return output.astype(np.uint8)

    def transform(self, results: dict) -> dict:
        """Apply the transform."""
        if random.random() > self.prob:
            return results
            
        img = results['img']
        if img is None:
            return results
            
        # Create mask from bounding boxes
        mask = np.zeros(img.shape[:2], dtype=np.int32)
        if 'gt_bboxes' in results:
            bboxes = results['gt_bboxes'].tensor.numpy() 
            for i, bbox in enumerate(bboxes):
                x1, y1, x2, y2 = map(int, bbox[:4]) 
                mask[y1:y2, x1:x2] = i + 1
        
        # Apply augmentations
        augmented_img = None
        
        if random.random() < self.global_prob:
            augmented_img = self.Global_Location_Scale_Augmentation(img)
            
        if random.random() < self.local_prob:
            local_aug = self.Local_Location_Scale_Augmentation(img, mask)
            if augmented_img is None:
                augmented_img = local_aug
            else:
                # 使用加权混合
                weight = random.uniform(0.4, 0.6)
                augmented_img = cv2.addWeighted(augmented_img, weight, local_aug, 1-weight, 0)
        
        # 如果没有应用任何增强，返回原图
        if augmented_img is None:
            augmented_img = img
            
        results['img'] = augmented_img
        return results
