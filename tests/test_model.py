import torch
from model.simple_model import FFN_Network
from model.piecewise_linear_fn import *

def test_relu_forward_shape():
    model = FFN_Network(1, 2, 16)
    x = torch.randn(32, 1)
    y = model(x)
    assert y.shape == (32, 1)

def test_relu_forward_method():
    model = FFN_Network(1, 1, 8)
    x = torch.randn(4, 1)
    y = model.forward(x)
    assert y.shape == (4, 1)

def test_relu_segment_network_forward_shape():
    x_points = [0.0, 1.0, 2.0]
    y_points = [1.0, 2.0, 4.0]
    model = ReluSegmentNetwork(x_points, y_points)

    x_input = torch.tensor([0.5, 1.5, 2.5], dtype=torch.float32)
    output = model.forward(x_input)
    assert output.shape == x_input.shape

def test_relu_segment_network_forward_values():
    x_points = [0.0, 1.0, 2.0]
    y_points = [1.0, 2.0, 4.0]
    model = ReluSegmentNetwork(x_points, y_points)

    # Test with a point that falls in the first segment
    x_input = torch.tensor([0.5], dtype=torch.float32)
    output = model.forward(x_input)

    # Expected output is the value at the segment corresponding to 0.5
    expected = 1 + (2 - 1) * 0.5  # = 1.5
    assert torch.allclose(output, torch.tensor([expected], dtype=torch.float32), atol=1e-4)

def test_relu_segment_network_2D_forward_shape():
    x_points = np.array([0.0, 1.0])
    y_points = np.array([0.0, 1.0])
    z_grid = np.array([[1.0, 2.0], [3.0, 4.0]])

    model = ReluSegmentNetwork2D(x_points, y_points, z_grid)

    x_input = torch.tensor([0.25, 0.5, 0.75, 0.9], dtype=torch.float32)
    y_input = torch.tensor([0.25, 0.5, 0.75, 0.9], dtype=torch.float32)

    output = model.forward(x_input, y_input)

    assert output.shape == x_input.shape
    assert torch.is_tensor(output)

