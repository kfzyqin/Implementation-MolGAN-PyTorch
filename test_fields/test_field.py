import torch.tensor as tensor
import torch

a_tensor = tensor([[11, 12, 13, 14],
        [21, 22, 23, 24],
        [31, 32, 33, 34],
        [41, 42, 43, 44]])

b_tensor = tensor([[1, 1, 1, 1],
        [2, 2, 2, 2],
        [3, 3, 3, 3],
        [4, 4, 4, 4]])

# 'ik, kj -> ij'语义解释如下：
# 输入a_tensor: 2维数组，下标为ik,
# 输入b_tensor: 2维数组，下标为kj,
# 输出output：2维数组，下标为ij。
# 隐含语义：输入a,b下标中相同的k，是求和的下标，对应上面的例子2的公式
output = torch.einsum('ik, jk -> ij', a_tensor, b_tensor)

print(output)

