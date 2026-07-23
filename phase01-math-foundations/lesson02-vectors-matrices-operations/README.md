# Vectors, Matrices & Operations

> Mọi neural network thực chất chỉ là matrix multiplication cộng thêm vài bước nữa.

- **Loại bài:** Build (Xây dựng)
- **Ngôn ngữ:** Python, Julia
- **Kiến thức cần có:** Phase 1, Lesson 01 (Trực giác về Linear Algebra)
- **Thời gian:** ~60 phút

## Mục tiêu học tập

- Xây dựng một class Matrix với các phép toán element-wise, matrix multiplication, transpose, determinant, và inverse
- Phân biệt element-wise multiplication với matrix multiplication và giải thích khi nào dùng cái nào
- Viết code cho một dense neural network layer (`relu(W @ x + b)`) chỉ dùng class Matrix tự xây
- Giải thích quy tắc broadcasting và cách bias addition hoạt động trong các neural network framework

## Vấn đề

Bạn muốn xây một neural network. Bạn đọc code và thấy dòng này:

```
output = activation(weights @ input + bias)
```

Dấu `@` đó là matrix multiplication. `weights` là một matrix. `input` là một vector. Nếu bạn không biết những phép toán đó làm gì, dòng này là phép thuật. Nếu bạn biết, nó chính là toàn bộ forward pass của một layer chỉ trong ba phép toán.

Mỗi ảnh mà model xử lý là một matrix chứa các pixel value. Mỗi word embedding là một vector. Mỗi layer của mỗi neural network là một matrix transformation. Bạn không thể xây dựng hệ thống AI mà không thành thạo các matrix operation — giống như bạn không thể viết code mà không hiểu biến.

Bài học này sẽ xây dựng sự thành thạo đó từ đầu.

## Khái niệm

### Vectors: danh sách số có thứ tự

Một vector là một danh sách các số có hướng (direction) và độ lớn (magnitude). Trong AI, vector dùng để biểu diễn data point, feature, hoặc parameter.

```
v = [3, 4]        -- một vector 2D
w = [1, 0, -2]    -- một vector 3D
```

Một vector 2D `[3, 4]` chỉ đến tọa độ (3, 4) trên mặt phẳng. Độ dài (magnitude) của nó là 5 (tam giác 3-4-5).

### Matrices: bảng số dạng lưới

Một matrix là một lưới 2D gồm hàng và cột. Một matrix m x n có m hàng và n cột.

```
A = | 1  2  3 |     -- matrix 2x3 (2 hàng, 3 cột)
    | 4  5  6 |
```

Trong neural network, weight matrix biến đổi input vector thành output vector. Một layer có 784 input và 128 output sẽ dùng weight matrix kích thước 128x784.

### Tại sao shape quan trọng

Matrix multiplication có quy tắc nghiêm ngặt: `(m x n) @ (n x p) = (m x p)`. Các chiều bên trong (inner dimensions) phải khớp nhau.

```
(128 x 784) @ (784 x 1) = (128 x 1)
  weights       input       output

Inner dimensions: 784 = 784  -- hợp lệ
```

Nếu bạn bị lỗi shape mismatch trong PyTorch, đây là lý do.

### Bản đồ các phép toán

| Phép toán | Làm gì | Ứng dụng trong neural network |
|-----------|--------|-------------------------------|
| Addition | Cộng từng phần tử tương ứng | Cộng bias vào output |
| Scalar multiply | Nhân mỗi phần tử với một số | Learning rate * gradients |
| Matrix multiply | Biến đổi vector | Forward pass của layer |
| Transpose | Đổi hàng thành cột | Backpropagation |
| Determinant | Tóm tắt thành một con số | Kiểm tra tính khả nghịch |
| Inverse | Đảo ngược phép biến đổi | Giải hệ phương trình tuyến tính |
| Identity | Matrix "không làm gì" | Khởi tạo, residual connection |

### Element-wise so với matrix multiplication

Sự khác biệt này hay làm người mới nhầm lẫn.

Element-wise: nhân từng vị trí tương ứng. Hai matrix phải cùng shape.

```
| 1  2 |   | 5  6 |   | 5  12 |
| 3  4 | * | 7  8 | = | 21 32 |
```

Matrix multiplication: tính dot product giữa hàng và cột. Inner dimensions phải khớp.

```
| 1  2 |   | 5  6 |   | 1*5+2*7  1*6+2*8 |   | 19  22 |
| 3  4 | @ | 7  8 | = | 3*5+4*7  3*6+4*8 | = | 43  50 |
```

Phép toán khác nhau, kết quả khác nhau, quy tắc khác nhau.

### Broadcasting

Khi bạn cộng một bias vector vào một matrix output, shape không khớp nhau. Broadcasting sẽ "kéo giãn" mảng nhỏ hơn cho vừa.

```
| 1  2  3 |   +   [10, 20, 30]
| 4  5  6 |

Broadcasting kéo giãn vector theo các hàng:

| 1  2  3 |   | 10  20  30 |   | 11  22  33 |
| 4  5  6 | + | 10  20  30 | = | 14  25  36 |
```

Mọi framework hiện đại đều làm điều này tự động. Hiểu nó sẽ giúp bạn không bị bối rối khi shape trông có vẻ sai nhưng code vẫn chạy.

```figure
vector-projection
```

## Xây dựng

### Bước 1: Class Vector

```python
class Vector:
    def __init__(self, data):
        self.data = list(data)
        self.size = len(self.data)

    def __repr__(self):
        return f"Vector({self.data})"

    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.data, other.data)])

    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self.data, other.data)])

    def __mul__(self, scalar):
        return Vector([x * scalar for x in self.data])

    def dot(self, other):
        return sum(a * b for a, b in zip(self.data, other.data))

    def magnitude(self):
        return sum(x ** 2 for x in self.data) ** 0.5
```

### Bước 2: Class Matrix với các phép toán cơ bản

```python
class Matrix:
    def __init__(self, data):
        self.data = [list(row) for row in data]
        self.rows = len(self.data)
        self.cols = len(self.data[0])
        self.shape = (self.rows, self.cols)

    def __repr__(self):
        rows_str = "\n  ".join(str(row) for row in self.data)
        return f"Matrix({self.shape}):\n  {rows_str}"

    def __add__(self, other):
        return Matrix([
            [self.data[i][j] + other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def __sub__(self, other):
        return Matrix([
            [self.data[i][j] - other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def scalar_multiply(self, scalar):
        return Matrix([
            [self.data[i][j] * scalar for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def element_wise_multiply(self, other):
        return Matrix([
            [self.data[i][j] * other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def matmul(self, other):
        return Matrix([
            [
                sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
                for j in range(other.cols)
            ]
            for i in range(self.rows)
        ])

    def transpose(self):
        return Matrix([
            [self.data[j][i] for j in range(self.rows)]
            for i in range(self.cols)
        ])

    def determinant(self):
        if self.shape == (1, 1):
            return self.data[0][0]
        if self.shape == (2, 2):
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
        det = 0
        for j in range(self.cols):
            minor = Matrix([
                [self.data[i][k] for k in range(self.cols) if k != j]
                for i in range(1, self.rows)
            ])
            det += ((-1) ** j) * self.data[0][j] * minor.determinant()
        return det

    def inverse_2x2(self):
        det = self.determinant()
        if det == 0:
            raise ValueError("Matrix is singular, no inverse exists")
        return Matrix([
            [self.data[1][1] / det, -self.data[0][1] / det],
            [-self.data[1][0] / det, self.data[0][0] / det]
        ])

    @staticmethod
    def identity(n):
        return Matrix([
            [1 if i == j else 0 for j in range(n)]
            for i in range(n)
        ])
```

### Bước 3: Xem nó hoạt động

```python
A = Matrix([[1, 2], [3, 4]])
B = Matrix([[5, 6], [7, 8]])

print("A + B =", (A + B).data)
print("A @ B =", A.matmul(B).data)
print("A^T =", A.transpose().data)
print("det(A) =", A.determinant())
print("A^-1 =", A.inverse_2x2().data)

I = Matrix.identity(2)
print("A @ A^-1 =", A.matmul(A.inverse_2x2()).data)
```

### Bước 4: Kết nối với neural network

```python
import random

inputs = Matrix([[0.5], [0.8], [0.2]])
weights = Matrix([
    [random.uniform(-1, 1) for _ in range(3)]
    for _ in range(2)
])
bias = Matrix([[0.1], [0.1]])

def relu_matrix(m):
    return Matrix([[max(0, val) for val in row] for row in m.data])

pre_activation = weights.matmul(inputs) + bias
output = relu_matrix(pre_activation)

print(f"Input shape: {inputs.shape}")
print(f"Weight shape: {weights.shape}")
print(f"Output shape: {output.shape}")
print(f"Output: {output.data}")
```

Đây là một dense layer: `output = relu(W @ x + b)`. Mọi dense layer trong mọi neural network đều làm đúng điều này.

## Sử dụng

NumPy làm tất cả những gì ở trên với ít code hơn và nhanh hơn hàng trăm lần.

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("A + B =\n", A + B)
print("A * B (element-wise) =\n", A * B)
print("A @ B (matrix multiply) =\n", A @ B)
print("A^T =\n", A.T)
print("det(A) =", np.linalg.det(A))
print("A^-1 =\n", np.linalg.inv(A))
print("I =\n", np.eye(2))

inputs = np.random.randn(3, 1)
weights = np.random.randn(2, 3)
bias = np.array([[0.1], [0.1]])
output = np.maximum(0, weights @ inputs + bias)

print(f"\nNeural network layer: {weights.shape} @ {inputs.shape} = {output.shape}")
print(f"Output:\n{output}")
```

Toán tử `@` trong Python gọi hàm `__matmul__`. NumPy triển khai nó bằng các BLAS routine tối ưu viết bằng C và Fortran. Cùng phép toán, nhanh hơn 100 lần.

Broadcasting trong NumPy:

```python
matrix = np.array([[1, 2, 3], [4, 5, 6]])
bias = np.array([10, 20, 30])
print(matrix + bias)
```

NumPy tự động broadcast bias 1D vào cả hai hàng. Đây là cách bias addition hoạt động trong mọi neural network framework.

## Đưa vào sử dụng

Bài học này tạo ra một prompt để dạy matrix operation thông qua trực giác hình học. Xem `outputs/prompt-matrix-operations.md`.

Class Matrix xây ở đây là nền tảng cho mini neural network framework mà chúng ta sẽ xây trong Phase 3, Lesson 10.

## Bài tập

1. **Kiểm tra inverse.** Nhân `A @ A.inverse_2x2()` và xác nhận bạn được identity matrix. Thử với ba matrix 2x2 khác nhau. Chuyện gì xảy ra khi determinant bằng 0?

2. **Viết inverse cho 3x3.** Mở rộng class Matrix để tính inverse cho matrix 3x3 bằng phương pháp adjugate. Kiểm tra kết quả với `np.linalg.inv` của NumPy.

3. **Xây neural network hai layer.** Chỉ dùng class Matrix của bạn (không dùng NumPy), tạo một neural network hai layer: input (3) -> hidden (4) -> output (2). Khởi tạo weight ngẫu nhiên, chạy forward pass, và kiểm tra tất cả shape có đúng không.

## Thuật ngữ chính

| Thuật ngữ | Người ta hay nói | Thực sự có nghĩa là |
|-----------|-----------------|---------------------|
| Vector | "Một mũi tên" | Danh sách số có thứ tự. Trong AI: một điểm trong không gian nhiều chiều. |
| Matrix | "Một bảng số" | Một linear transformation. Nó ánh xạ vector từ không gian này sang không gian khác. |
| Matrix multiply | "Chỉ là nhân các số" | Dot product giữa mỗi hàng của matrix thứ nhất với mỗi cột của matrix thứ hai. Thứ tự quan trọng. |
| Transpose | "Lật nó lại" | Đổi hàng thành cột. Biến matrix m x n thành n x m. Rất quan trọng trong backpropagation. |
| Determinant | "Một con số gì đó từ matrix" | Đo mức độ matrix co giãn diện tích (2D) hoặc thể tích (3D). Bằng 0 nghĩa là phép biến đổi đã "nén bẹp" mất một chiều. |
| Inverse | "Đảo ngược matrix" | Matrix mà khi nhân vào sẽ đảo ngược phép biến đổi. Chỉ tồn tại khi determinant khác 0. |
| Identity matrix | "Matrix nhàm chán" | Matrix tương đương với nhân cho 1. Dùng trong residual connection (ResNets). |
| Broadcasting | "Tự sửa shape thần kỳ" | Kéo giãn mảng nhỏ hơn để khớp với mảng lớn hơn bằng cách lặp lại theo chiều còn thiếu. |
| Element-wise | "Phép nhân bình thường" | Nhân từng vị trí tương ứng. Hai mảng phải cùng shape (hoặc broadcastable). |

## Đọc thêm

- [3Blue1Brown: Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra) — trực giác trực quan cho mọi phép toán trong bài này
- [NumPy documentation on broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) — quy tắc chính xác mà NumPy tuân theo
- [Stanford CS229 Linear Algebra Review](http://cs229.stanford.edu/section/cs229-linalg.pdf) — tài liệu tham khảo ngắn gọn về linear algebra cho ML
