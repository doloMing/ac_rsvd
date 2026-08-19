import sys

import torch


torch.ops.load_library(sys.argv[1])

matrix = torch.zeros((8, 6), dtype=torch.float64)
matrix[range(6), range(6)] = torch.tensor(
    [10.0, 5.0, 2.0, 0.5, 0.1, 0.01], dtype=torch.float64
)


def error(result):
    u, singular_values, v = result[:3]
    return torch.linalg.norm(matrix - (u * singular_values) @ v.T).item()


ac_result = torch.ops.ac_rsvd.run_ac_rsvd(matrix, 0.2, 0.1, 2, 7, 0)
if error(ac_result) > 0.2:
    raise RuntimeError("AC-RSVD binding missed the tolerance")
if ac_result[3][4].item() != 1:
    raise RuntimeError("AC-RSVD binding lost the adjoint count")

mf_result = torch.ops.ac_rsvd.run_randqb_mf_fro(matrix, 1.0, 2, 7, 0)
if error(mf_result) > 1.0:
    raise RuntimeError("randQB_MF_Fro binding missed the tolerance")

frobenius_norm = torch.linalg.norm(matrix).item()
ei_result = torch.ops.ac_rsvd.run_randqb_ei(
    matrix, 0.2, frobenius_norm, 2, 7, 0
)
if error(ei_result) > 0.2:
    raise RuntimeError("randQB_EI binding missed the tolerance")

try:
    torch.ops.ac_rsvd.run_ac_rsvd(matrix.float(), 0.2, 0.1, 2, 7, 0)
except RuntimeError as error_message:
    if "float64" not in str(error_message):
        raise
else:
    raise RuntimeError("binding accepted a float32 matrix")

print("Torch binding tests passed")
