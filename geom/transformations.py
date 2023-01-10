import torch

def quat2mat(quaternions: torch.Tensor) -> torch.Tensor:
    # Check that the input is a tensor of quaternions with shape (n, 4)
    if quaternions.dim() != 2 or quaternions.size(1) != 4:
        raise ValueError("Expected a tensor of quaternions with shape (n, 4)")

    # Get the batch size from the shape of the input tensor
    batch_size = quaternions.size(0)

    # Extract the scalar part and the vector part of the quaternions
    w, x, y, z = quaternions.unbind(1)

    # Compute the rotation matrix elements
    xx = x * x
    xy = x * y
    xz = x * z
    xw = x * w

    yy = y * y
    yz = y * z
    yw = y * w

    zz = z * z
    zw = z * w

    # Stack the computed elements to form the rotation matrices
    rotation_matrices = torch.stack([
        w * w + xx - yy - zz, 2 * (xy + zw), 2 * (xz - yw),
        2 * (xy - zw), w * w - xx + yy - zz, 2 * (yz + xw),
        2 * (xz + yw), 2 * (yz - xw), w * w - xx - yy + zz
    ], dim=1).view(batch_size, 3, 3)

    return rotation_matrices

def quaternion_to_rotation_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    # Check that the input is a tensor of quaternions with shape (n, 4)
    if quaternions.dim() != 2 or quaternions.size(1) != 4:
        raise ValueError("Expected a tensor of quaternions with shape (n, 4)")

    # Get the batch size from the shape of the input tensor
    batch_size = quaternions.size(0)

    # Compute the rotation matrices by using the quat2mat function from the torch.quaternion module
    rotation_matrices = quat2mat(quaternions)

    # Reshape the rotation matrices to have shape (n, 3, 3)
    return rotation_matrices.view(batch_size, 3, 3)


def sample_r3(n, device):
    # Sample tensors from normal distribution with zero mean and unit variance on specified device
    tensors = torch.randn(n, 3, device=device)
    return tensors


def sample_so3(n, device):
    # Generate n random numbers from a uniform distribution in the range [0, 1)
    u = torch.rand(n, 3, device=device)

    # Scale the second and third element of u by 2 * pi
    u[:, 1:] *= torch.tensor(2 * torch.pi, device=device)

    # Compute a and b as the square root of 1 - u[:, 0] and u[:, 0], respectively
    a = (1 - u[:, 0]) ** .5
    b = (u[:, 0]) ** .5

    # Initialize a tensor of quaternions with zeros
    q_n = torch.zeros([n, 4], device=device)

    # Compute the scalar and vector parts of the quaternions from a, b, and u[:, 1:]
    q_n[:, 0] = a * torch.sin(u[:, 1])
    q_n[:, 1] = a * torch.cos(u[:, 1])
    q_n[:, 2] = b * torch.sin(u[:, 2])
    q_n[:, 3] = b * torch.cos(u[:, 2])

    # Convert the quaternions to rotation matrices using the quat2mat function from the torch.quaternion module
    R = quat2mat(q_n)

    # Return the rotation matrices
    return R

def test_sample_so3():
    # Set the number of samples and the device to use
    n = 10
    device = torch.device("cpu")

    # Sample n rotation matrices from SO(3)
    R = sample_so3(n, device)

    # Check that the shape of R is (n, 3, 3)
    assert R.shape == (n, 3, 3)

    # Check that the determinant of each rotation matrix is 1
    assert torch.allclose(R.det(), torch.ones(n))

    # Check that each rotation matrix is orthogonal
    assert torch.allclose(R.matmul(R.transpose(-2, -1)), torch.eye(3).expand(n, -1, -1))

    # Check that each rotation matrix has a unit Frobenius norm
    assert torch.allclose(R.norm("fro", dim=(1, 2)), torch.ones(n))

def zyz_euler_angles_from_rotation_matrix(Rs, device):
    if Rs.ndim == 2:
        assert Rs.shape[0] == Rs.shape[1] == 3
        n = 1
        Rs = Rs.reshape(1, 3, 3)
    elif Rs.ndim == 3:
        assert Rs.shape[1] == Rs.shape[2] == 3
        n = Rs.shape[0]
    abg = torch.zeros((n, 3), device=device)
    abg[:, 0] = torch.atan2(Rs[:, 1, 2], Rs[:, 0, 2])
    abg[:, 1] = torch.atan2((Rs[:, 0, 2] ** 2 + Rs[:, 1, 2] ** 2) ** .5, Rs[:, 2, 2])
    abg[:, 2] = torch.atan2(Rs[:, 2, 1], - Rs[:, 2, 0])
    return abg