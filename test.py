from torch_scatter import segment_csr
import torch

src = torch.randn(10, 6, 64)
indptr = torch.tensor([0, 2, 5, 6])
indptr = indptr.view(1, -1)  # Broadcasting in the first and last dim.
print(indptr)

out = segment_csr(src, indptr, reduce="sum")

print(out.size())