import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn
from utils.maths import *
from typing import List

class ReluSegmentNetwork(nn.Module):
    def __init__(self, x_points, y_points):
        """
        Initialize the ReLU segment network.

        Parameters:
            x_points (list): List of x breakpoints [x1, x2, ..., xn].
            y_points (list): List of y values at the breakpoints [y1, y2, ..., yn].
        """
        super(ReluSegmentNetwork, self).__init__()
        self.x_points = torch.tensor(x_points, dtype=torch.float32)
        self.y_points = torch.tensor(y_points, dtype=torch.float32)
        self.slopes = self.calculate_slopes()

    def calculate_slopes(self):
        """
        Calculate the slopes for each segment based on x and y points.

        Returns:
            Tensor: Slopes for each segment.
        """
        slopes = [
            get_slope_by_2_points((self.x_points[i].item(), self.y_points[i].item()),
                                  (self.x_points[i + 1].item(), self.y_points[i + 1].item()))
            for i in range(len(self.x_points) - 1)
        ]
        return torch.tensor(slopes, dtype=torch.float32)

    def forward(self, x):
        """
        Forward pass to compute the piecewise linear function and individual segment outputs.

        Parameters:
            x (Tensor): Input x values.

        Returns:
            Tuple[Tensor, List[Tensor]]: The base value at x1 and a list of outputs for each segment.
        """
        base_value = self.y_points[0]  # f*n(x1)
        segment_outputs = []
        for i in range(len(self.slopes)):
            sign_k = torch.sign(self.slopes[i])
            abs_k = torch.abs(self.slopes[i])
            x_i, x_next = self.x_points[i], self.x_points[i + 1]
            O_i = sign_k * F.relu(abs_k * (F.relu(x - x_i) - F.relu(x - x_next)))
            segment_outputs.append(O_i)
        return base_value + sum(segment_outputs)
    

class FixedWidthReluNetwork(nn.Module):
    def __init__(self, x_points, y_points):
        """
        Initialize the fixed-width ReLU network.

        Parameters:
            x_points (list): List of x breakpoints [x1, x2, ..., xn].
            y_points (list): List of y values at the breakpoints [y1, y2, ..., yn].
        """
        super(FixedWidthReluNetwork, self).__init__()
        self.x_points = torch.tensor(x_points, dtype=torch.float32)
        self.y_points = torch.tensor(y_points, dtype=torch.float32)
        self.slopes = self.calculate_slopes()

    def calculate_slopes(self):
        """
        Calculate the slopes for each segment based on x and y points.

        Returns:
            Tensor: Slopes for each segment.
        """
        slopes = []
        for i in range(len(self.x_points) - 1):
            dx = self.x_points[i + 1] - self.x_points[i]
            dy = self.y_points[i + 1] - self.y_points[i]
            slopes.append(dy / dx)
        return torch.tensor(slopes, dtype=torch.float32)

    def forward(self, x):
        """
        Forward pass for the fixed-width architecture.

        Parameters:
            x (Tensor): Input x values.

        Returns:
            Tensor: Final output of the function.
        """
        f_x = torch.full_like(x, self.y_points[0])  # Initialize with f*n(x1)
        for i in range(len(self.slopes)):
            x_i, x_next = self.x_points[i], self.x_points[i + 1]
            sigma_x_minus_xi = F.relu(x - x_i)
            sigma_x_minus_xnext = F.relu(x - x_next)
            O_i = self.slopes[i] * (sigma_x_minus_xi - sigma_x_minus_xnext)
            f_x += O_i
        return f_x


class ReluSegmentNetwork2D(nn.Module):
    def __init__(self, x_points, y_points, z_grid):
        super(ReluSegmentNetwork2D, self).__init__()
        self.x_points = torch.tensor(x_points, dtype=torch.float32)
        self.y_points = torch.tensor(y_points, dtype=torch.float32)
        self.z_grid = torch.tensor(z_grid, dtype=torch.float32)

    def forward(self, x, y):
        result = torch.zeros_like(x)
        
        for i in range(len(self.x_points) - 1):
            for j in range(len(self.y_points) - 1):
                x0, x1 = self.x_points[i], self.x_points[i + 1]
                y0, y1 = self.y_points[j], self.y_points[j + 1]
                
                char_x = (F.relu(x - x0) - F.relu(x - x1)) / (x1 - x0)
                char_y = (F.relu(y - y0) - F.relu(y - y1)) / (y1 - y0)
                char_func = char_x * char_y
                
                # Bilinear-interpolation
                z00 = self.z_grid[i][j]
                z10 = self.z_grid[i + 1][j]
                z01 = self.z_grid[i][j + 1]
                z11 = self.z_grid[i + 1][j + 1]
                
                dx = x1 - x0
                dy = y1 - y0
                
                x_rel = (x - x0) / dx
                y_rel = (y - y0) / dy

                bilinear = (z00 * (1 - x_rel) * (1 - y_rel) + 
                            z10 * x_rel * (1 - y_rel) + 
                            z01 * (1 - x_rel) * y_rel + 
                            z11 * x_rel * y_rel)
                
                result += bilinear * char_func
        
        return result