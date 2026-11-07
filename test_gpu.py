
import torch

print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
print("XPU Available:", torch.xpu.is_available())

if torch.xpu.is_available():
    print("GPU Name:", torch.xpu.get_device_name(0))

    x = torch.randn(2000, 2000, device="xpu")
    y = torch.randn(2000, 2000, device="xpu")

    result = x @ y
    torch.xpu.synchronize()

    print("GPU Computation: SUCCESS")
    print("Device:", result.device)
else:
    print("Intel GPU not detected by PyTorch.")
