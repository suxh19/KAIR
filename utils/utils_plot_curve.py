"""
基于 OpenCV 的训练曲线绘制工具

用于在训练过程中实时生成和保存训练指标曲线图
"""

import os
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple


class TrainingCurvePlotter:
    """使用 OpenCV 绘制训练曲线的工具类"""
    
    def __init__(self, save_dir: str, width: int = 1200, height: int = 800):
        """
        初始化绘图器
        
        Args:
            save_dir: 图像保存目录
            width: 图像宽度
            height: 图像高度
        """
        self.save_dir = save_dir
        self.width = width
        self.height = height
        
        # 边距设置
        self.margin_left = 100
        self.margin_right = 50
        self.margin_top = 50
        self.margin_bottom = 80
        
        # 训练数据存储
        self.train_iters: List[int] = []
        self.train_logs: Dict[str, List[float]] = {
            'G_loss': [],
            'F_loss': [],
            'D_loss': [],
            'D_real': [],
            'D_fake': []
        }
        
        # 测试数据存储
        self.test_iters: List[int] = []
        self.test_psnr: List[float] = []
        
        # 颜色定义 (BGR格式)
        self.colors = {
            'G_loss': (0, 165, 255),    # 橙色
            'F_loss': (0, 255, 0),       # 绿色
            'D_loss': (255, 0, 0),       # 蓝色
            'D_real': (255, 165, 0),     # 浅蓝
            'D_fake': (128, 0, 128),     # 紫色
            'psnr': (0, 140, 255),       # 深橙色
        }
        
        # 确保保存目录存在
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
    def add_training_data(self, iteration: int, logs: Dict[str, float]) -> None:
        """
        添加训练数据点
        
        Args:
            iteration: 当前迭代次数
            logs: 包含训练指标的字典
        """
        self.train_iters.append(iteration)
        for key in self.train_logs:
            if key in logs:
                self.train_logs[key].append(float(logs[key]))
            else:
                # 如果日志中没有该键，添加 NaN
                self.train_logs[key].append(float('nan'))
    
    def add_test_data(self, iteration: int, psnr: float) -> None:
        """
        添加测试数据点
        
        Args:
            iteration: 当前迭代次数
            psnr: 平均 PSNR 值
        """
        self.test_iters.append(iteration)
        self.test_psnr.append(psnr)
    
    def _create_canvas(self) -> np.ndarray:
        """创建白色画布"""
        canvas = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
        return canvas
    
    def _draw_axes(self, canvas: np.ndarray, title: str,
                   x_min: float, x_max: float, y_min: float, y_max: float,
                   x_label: str = "Iteration", y_label: str = "Value") -> None:
        """
        绘制坐标轴
        
        Args:
            canvas: 画布
            title: 图表标题
            x_min, x_max: X轴范围
            y_min, y_max: Y轴范围
            x_label, y_label: 轴标签
        """
        plot_width = self.width - self.margin_left - self.margin_right
        plot_height = self.height - self.margin_top - self.margin_bottom
        
        # 绘制坐标轴线
        # X轴
        cv2.line(canvas, 
                 (self.margin_left, self.height - self.margin_bottom),
                 (self.width - self.margin_right, self.height - self.margin_bottom),
                 (0, 0, 0), 2)
        # Y轴
        cv2.line(canvas,
                 (self.margin_left, self.margin_top),
                 (self.margin_left, self.height - self.margin_bottom),
                 (0, 0, 0), 2)
        
        # 绘制标题
        cv2.putText(canvas, title, 
                    (self.width // 2 - len(title) * 8, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # 绘制X轴标签
        cv2.putText(canvas, x_label,
                    (self.width // 2 - len(x_label) * 5, self.height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # 绘制Y轴标签（垂直）
        for i, char in enumerate(y_label):
            cv2.putText(canvas, char,
                        (15, self.height // 2 - len(y_label) * 8 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # 绘制X轴刻度
        num_x_ticks = 6
        for i in range(num_x_ticks):
            x_val = x_min + (x_max - x_min) * i / (num_x_ticks - 1) if num_x_ticks > 1 else x_min
            x_pos = self.margin_left + int(plot_width * i / (num_x_ticks - 1)) if num_x_ticks > 1 else self.margin_left
            
            # 刻度线
            cv2.line(canvas,
                     (x_pos, self.height - self.margin_bottom),
                     (x_pos, self.height - self.margin_bottom + 5),
                     (0, 0, 0), 1)
            
            # 刻度值
            tick_text = f"{int(x_val)}"
            cv2.putText(canvas, tick_text,
                        (x_pos - len(tick_text) * 4, self.height - self.margin_bottom + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            
            # 网格线
            cv2.line(canvas,
                     (x_pos, self.margin_top),
                     (x_pos, self.height - self.margin_bottom),
                     (220, 220, 220), 1)
        
        # 绘制Y轴刻度
        num_y_ticks = 6
        for i in range(num_y_ticks):
            y_val = y_min + (y_max - y_min) * i / (num_y_ticks - 1) if num_y_ticks > 1 else y_min
            y_pos = self.height - self.margin_bottom - int(plot_height * i / (num_y_ticks - 1)) if num_y_ticks > 1 else self.height - self.margin_bottom
            
            # 刻度线
            cv2.line(canvas,
                     (self.margin_left - 5, y_pos),
                     (self.margin_left, y_pos),
                     (0, 0, 0), 1)
            
            # 刻度值
            tick_text = f"{y_val:.2e}" if abs(y_val) < 0.01 or abs(y_val) > 1000 else f"{y_val:.2f}"
            cv2.putText(canvas, tick_text,
                        (5, y_pos + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
            
            # 网格线
            cv2.line(canvas,
                     (self.margin_left, y_pos),
                     (self.width - self.margin_right, y_pos),
                     (220, 220, 220), 1)
    
    def _draw_curve(self, canvas: np.ndarray, x_data: List[int], y_data: List[float],
                    x_min: float, x_max: float, y_min: float, y_max: float,
                    color: Tuple[int, int, int], label: str) -> None:
        """
        在画布上绘制曲线
        
        Args:
            canvas: 画布
            x_data, y_data: 数据点
            x_min, x_max, y_min, y_max: 坐标范围
            color: 曲线颜色 (BGR)
            label: 曲线标签
        """
        if len(x_data) < 2:
            return
        
        plot_width = self.width - self.margin_left - self.margin_right
        plot_height = self.height - self.margin_top - self.margin_bottom
        
        x_range = x_max - x_min if x_max != x_min else 1
        y_range = y_max - y_min if y_max != y_min else 1
        
        points = []
        for x, y in zip(x_data, y_data):
            if np.isnan(y):
                continue
            px = self.margin_left + int((x - x_min) / x_range * plot_width)
            py = self.height - self.margin_bottom - int((y - y_min) / y_range * plot_height)
            # 裁剪到绘图区域内
            px = max(self.margin_left, min(self.width - self.margin_right, px))
            py = max(self.margin_top, min(self.height - self.margin_bottom, py))
            points.append((px, py))
        
        # 绘制曲线
        for i in range(len(points) - 1):
            cv2.line(canvas, points[i], points[i + 1], color, 2)
    
    def _draw_legend(self, canvas: np.ndarray, labels: List[str], 
                     colors: List[Tuple[int, int, int]]) -> None:
        """
        绘制图例
        
        Args:
            canvas: 画布
            labels: 标签列表
            colors: 颜色列表
        """
        legend_x = self.width - self.margin_right - 120
        legend_y = self.margin_top + 20
        
        for i, (label, color) in enumerate(zip(labels, colors)):
            y_pos = legend_y + i * 25
            # 绘制颜色方块
            cv2.rectangle(canvas, (legend_x, y_pos - 8), (legend_x + 20, y_pos + 4), color, -1)
            # 绘制标签文字
            cv2.putText(canvas, label, (legend_x + 30, y_pos + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    def _get_data_range(self, data_list: List[List[float]], 
                        margin_ratio: float = 0.1) -> Tuple[float, float]:
        """
        计算数据范围（带边距）
        
        Args:
            data_list: 多组数据
            margin_ratio: 边距比例
            
        Returns:
            (min_val, max_val)
        """
        all_data = []
        for data in data_list:
            all_data.extend([v for v in data if not np.isnan(v)])
        
        if not all_data:
            return 0, 1
        
        min_val = min(all_data)
        max_val = max(all_data)
        data_range = max_val - min_val if max_val != min_val else abs(max_val) * 0.1 or 1
        
        return min_val - data_range * margin_ratio, max_val + data_range * margin_ratio
    
    def save_curves(self, current_step: int) -> None:
        """
        保存所有曲线图
        
        Args:
            current_step: 当前步数
        """
        if not self.train_iters:
            return
        
        x_min = min(self.train_iters)
        x_max = max(self.train_iters)
        
        # 1. 保存训练损失曲线
        self._save_loss_curves(current_step, x_min, x_max)
        
        # 2. 保存判别器值曲线
        self._save_discriminator_curves(current_step, x_min, x_max)
        
        # 3. 保存PSNR曲线
        self._save_psnr_curve(current_step)
    
    def _save_loss_curves(self, current_step: int, x_min: float, x_max: float) -> None:
        """保存训练损失曲线"""
        canvas = self._create_canvas()
        
        loss_keys = ['G_loss', 'F_loss', 'D_loss']
        loss_data = [self.train_logs[k] for k in loss_keys]
        y_min, y_max = self._get_data_range(loss_data)
        
        self._draw_axes(canvas, "Training Losses", x_min, x_max, y_min, y_max,
                        "Iteration", "Loss")
        
        for key in loss_keys:
            if self.train_logs[key]:
                self._draw_curve(canvas, self.train_iters, self.train_logs[key],
                                x_min, x_max, y_min, y_max,
                                self.colors[key], key)
        
        self._draw_legend(canvas, loss_keys, [self.colors[k] for k in loss_keys])
        
        save_path = os.path.join(self.save_dir, 'training_losses.png')
        cv2.imwrite(save_path, canvas)
    
    def _save_discriminator_curves(self, current_step: int, x_min: float, x_max: float) -> None:
        """保存判别器值曲线"""
        canvas = self._create_canvas()
        
        disc_keys = ['D_real', 'D_fake']
        disc_data = [self.train_logs[k] for k in disc_keys]
        y_min, y_max = self._get_data_range(disc_data)
        
        self._draw_axes(canvas, "Discriminator Outputs", x_min, x_max, y_min, y_max,
                        "Iteration", "Value")
        
        for key in disc_keys:
            if self.train_logs[key]:
                self._draw_curve(canvas, self.train_iters, self.train_logs[key],
                                x_min, x_max, y_min, y_max,
                                self.colors[key], key)
        
        self._draw_legend(canvas, disc_keys, [self.colors[k] for k in disc_keys])
        
        save_path = os.path.join(self.save_dir, 'discriminator_values.png')
        cv2.imwrite(save_path, canvas)
    
    def _save_psnr_curve(self, current_step: int) -> None:
        """保存PSNR曲线"""
        if not self.test_iters:
            return
        
        canvas = self._create_canvas()
        
        x_min = min(self.test_iters)
        x_max = max(self.test_iters)
        y_min, y_max = self._get_data_range([self.test_psnr])
        
        self._draw_axes(canvas, "Test Average PSNR", x_min, x_max, y_min, y_max,
                        "Iteration", "PSNR (dB)")
        
        self._draw_curve(canvas, self.test_iters, self.test_psnr,
                         x_min, x_max, y_min, y_max,
                         self.colors['psnr'], 'PSNR')
        
        # 在数据点位置绘制标记和数值
        if len(self.test_iters) > 0:
            plot_width = self.width - self.margin_left - self.margin_right
            plot_height = self.height - self.margin_top - self.margin_bottom
            x_range = x_max - x_min if x_max != x_min else 1
            y_range = y_max - y_min if y_max != y_min else 1
            
            for x, y in zip(self.test_iters, self.test_psnr):
                px = self.margin_left + int((x - x_min) / x_range * plot_width)
                py = self.height - self.margin_bottom - int((y - y_min) / y_range * plot_height)
                # 绘制圆点
                cv2.circle(canvas, (px, py), 4, self.colors['psnr'], -1)
                # 标注数值
                cv2.putText(canvas, f"{y:.1f}",
                            (px - 15, py - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        self._draw_legend(canvas, ['PSNR'], [self.colors['psnr']])
        
        save_path = os.path.join(self.save_dir, 'test_psnr.png')
        cv2.imwrite(save_path, canvas)
