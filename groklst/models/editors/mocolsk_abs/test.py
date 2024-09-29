import torch

# 创建示例 tensors
tensor1 = torch.randn(2, 32)  # 形状 (B, C)
tensor2 = torch.randn(2, 50, 32)  # 形状 (B, N, C)

# 转置 tensor2 以便于矩阵乘法
tensor2_transposed = tensor2.transpose(1, 2)  # 形状 (B, C, N)

# 进行矩阵乘法
result = torch.matmul(tensor1, tensor2_transposed)  # 形状 (B, C, N)

# 对结果进行 squeeze 操作，以去掉 N 维度
result_squeezed,_ = result.sum(dim=2)  # 形状 (B, C)

print(result_squeezed.shape)
