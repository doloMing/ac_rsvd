import sys

import torch


torch.ops.load_library(sys.argv[1])

matrix = torch.diag(
    torch.tensor([1.0, 0.2, 0.04, 0.008, 0.0016], dtype=torch.float64)
)
u, singular_values, v, counters, diagnostics, stop_reason = (
    torch.ops.ac_rsvd.run_ac_rsvd(matrix, 1e-2, 1e-6, 3, 7, 0)
)

print("rank:", singular_values.numel())
print("error:", torch.linalg.norm(matrix - (u * singular_values) @ v.T).item())
print("counters:", counters.tolist())
print("diagnostics:", diagnostics.tolist())
print("stop reason:", stop_reason)
