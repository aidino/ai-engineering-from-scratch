# GPU Setup & Cloud

> Training trên CPU thì ổn để học. Training thật sự thì cần GPU.

- **Loại:** Build
- **Ngôn ngữ:** Python
- **Yêu cầu trước:** Phase 0, Lesson 01
- **Thời gian:** ~45 phút

## Mục tiêu học tập

- Kiểm tra GPU local có sẵn không bằng `nvidia-smi` và CUDA API của PyTorch
- Cấu hình Google Colab với T4 GPU để thử nghiệm miễn phí trên cloud
- Benchmark phép nhân matrix trên CPU vs GPU và đo speedup
- Ước tính model lớn nhất vừa với VRAM của bạn dùng quy tắc fp16

## Vấn đề

Hầu hết các bài học ở phase 1-3 chạy tốt trên CPU. Nhưng khi bạn bắt đầu training CNN, transformer, hay LLM (phase 4 trở đi), bạn cần GPU acceleration. Một lần training mất 8 tiếng trên CPU chỉ mất 10 phút trên GPU.

Bạn có ba lựa chọn: GPU local, GPU cloud, hoặc Google Colab (miễn phí).

## Khái niệm

```
Các lựa chọn của bạn:

1. Local NVIDIA GPU
   Chi phí: $0 (bạn đã có sẵn)
   Setup: Cài CUDA + cuDNN
   Phù hợp cho: Dùng thường xuyên, dataset lớn

2. Google Colab (free tier)
   Chi phí: $0
   Setup: Không cần
   Phù hợp cho: Thử nghiệm nhanh, không có GPU ở nhà

3. Cloud GPU (Lambda, RunPod, Vast.ai)
   Chi phí: $0.20-2.00/giờ
   Setup: SSH + cài đặt
   Phù hợp cho: Training nghiêm túc, model lớn
```

## Bắt tay vào làm

### Lựa chọn 1: Local NVIDIA GPU

Kiểm tra xem bạn có GPU không:

```bash
nvidia-smi
```

Cài PyTorch với CUDA:

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### Lựa chọn 2: Google Colab

1. Vào [colab.research.google.com](https://colab.research.google.com)
2. Runtime > Change runtime type > T4 GPU
3. Chạy `!nvidia-smi` để kiểm tra

Upload notebook từ khóa học này lên Colab.

### Lựa chọn 3: Cloud GPU

Với Lambda Labs, RunPod, hoặc Vast.ai:

```bash
ssh user@your-gpu-instance

pip install torch torchvision torchaudio
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### Không có GPU? Không sao.

Hầu hết các bài học chạy được trên CPU. Những bài cần GPU sẽ ghi rõ và có link Colab đi kèm.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")
```

## Bắt tay vào làm: Benchmark GPU vs CPU

```python
import torch
import time

size = 5000

a_cpu = torch.randn(size, size)
b_cpu = torch.randn(size, size)

start = time.time()
c_cpu = a_cpu @ b_cpu
cpu_time = time.time() - start
print(f"CPU: {cpu_time:.3f}s")

if torch.cuda.is_available():
    a_gpu = a_cpu.to("cuda")
    b_gpu = b_cpu.to("cuda")

    torch.cuda.synchronize()
    start = time.time()
    c_gpu = a_gpu @ b_gpu
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    print(f"GPU: {gpu_time:.3f}s")
    print(f"Speedup: {cpu_time / gpu_time:.0f}x")
```

## Bài tập

1. Chạy benchmark ở trên và so sánh thời gian CPU vs GPU
2. Nếu bạn không có GPU, chạy trên Google Colab rồi so sánh
3. Kiểm tra VRAM GPU bạn có bao nhiêu và ước tính model lớn nhất bạn có thể chạy (quy tắc: 2 byte mỗi parameter cho fp16)

## Thuật ngữ quan trọng

| Thuật ngữ | Cách mọi người hay nói | Ý nghĩa thực sự |
|-----------|----------------------|-----------------|
| CUDA | "GPU programming" | Platform tính toán song song của NVIDIA cho phép chạy code trên GPU |
| VRAM | "GPU memory" | Video RAM trên GPU, tách biệt với system RAM. Giới hạn kích thước model. |
| fp16 | "Half precision" | Floating point 16-bit, dùng một nửa bộ nhớ so với fp32 mà độ chính xác giảm không đáng kể |
| Tensor Core | "Fast matrix hardware" | Các core GPU chuyên dụng cho phép nhân matrix, nhanh hơn core thường 4-8 lần |
