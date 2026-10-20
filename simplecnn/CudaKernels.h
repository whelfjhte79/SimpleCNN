// SimpleCNN - CUDA 커널 소스
//
// 커널은 실행 중에 NVRTC로 "지금 꽂혀있는 GPU" 아키텍처에 맞춰 컴파일된다.
// 그래서 nvcc나 .cu 빌드 설정 없이 #include "SimpleCNN.h" 만으로 CUDA가 동작한다.
// (MSVC 문자열 길이 제한 때문에 여러 조각으로 나눠 두고 실행 시 이어 붙인다)
#ifndef SIMPLE_CNN_CUDA_KERNELS_H
#define SIMPLE_CNN_CUDA_KERNELS_H

namespace cnn {
namespace cuda {

	// 공통: 활성화 함수 코드는 ACTIVATION enum 순서와 같다.
	static const char* kKernelCommon = R"CUDA(
#define ACT_NONE 0
#define ACT_SIGMOID 2
#define ACT_RELU 3
#define ACT_LEAKY_RELU 4
#define ACT_TANH 5
#define ACT_ELU 7

__device__ float actForward(float x, int act) {
	if (act == ACT_RELU) {
		if (x < 0.0f)
			return 0.0f;
		return x;
	}
	if (act == ACT_SIGMOID)
		return 1.0f / (1.0f + expf(-x));
	if (act == ACT_TANH)
		return tanhf(x);
	if (act == ACT_LEAKY_RELU) {
		if (x < 0.0f)
			return SCNN_LEAKY * x;
		return x;
	}
	if (act == ACT_ELU) {
		if (x < 0.0f)
			return SCNN_ELU * (expf(x) - 1.0f);
		return x;
	}
	return x;
}

// derivative from the output y (no need to keep the input)
__device__ float actGrad(float y, int act) {
	if (act == ACT_RELU) {
		if (y > 0.0f)
			return 1.0f;
		return 0.0f;
	}
	if (act == ACT_SIGMOID)
		return y * (1.0f - y);
	if (act == ACT_TANH)
		return 1.0f - y * y;
	if (act == ACT_LEAKY_RELU) {
		if (y > 0.0f)
			return 1.0f;
		return SCNN_LEAKY;
	}
	if (act == ACT_ELU) {
		if (y > 0.0f)
			return 1.0f;
		return y + SCNN_ELU;
	}
	return 1.0f;
}

extern "C" __global__ void fill_kernel(float* x, float value, int n) {
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
		x[i] = value;
}

extern "C" __global__ void scale_kernel(float* x, float s, int n) {
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
		x[i] *= s;
}

extern "C" __global__ void act_forward_kernel(const float* x, float* y, int n, int act) {
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
		y[i] = actForward(x[i], act);
}

extern "C" __global__ void act_backward_kernel(const float* y, float* dy, int n, int act) {
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
		dy[i] *= actGrad(y[i], act);
}

extern "C" __global__ void softmax_rows_kernel(float* y, int rows, int cols) {
	for (int r = blockIdx.x * blockDim.x + threadIdx.x; r < rows; r += blockDim.x * gridDim.x) {
		float* row = y + (size_t)r * cols;
		float maxValue = row[0];
		for (int j = 1; j < cols; ++j)
			maxValue = fmaxf(maxValue, row[j]);
		float sum = 0.0f;
		for (int j = 0; j < cols; ++j) {
			row[j] = expf(row[j] - maxValue);
			sum += row[j];
		}
		for (int j = 0; j < cols; ++j)
			row[j] /= sum;
	}
}

extern "C" __global__ void softmax_backward_kernel(const float* y, float* dy, int rows, int cols) {
	for (int r = blockIdx.x * blockDim.x + threadIdx.x; r < rows; r += blockDim.x * gridDim.x) {
		const float* yr = y + (size_t)r * cols;
		float* dr = dy + (size_t)r * cols;
		float dot = 0.0f;
		for (int j = 0; j < cols; ++j)
			dot += yr[j] * dr[j];
		for (int j = 0; j < cols; ++j)
			dr[j] = yr[j] * (dr[j] - dot);
	}
}

extern "C" __global__ void bias_act_rows_kernel(float* y, const float* bias, int rows, int cols, int act) {
	int n = rows * cols;
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
		y[i] = actForward(y[i] + bias[i % cols], act);
}

// FC bias gradient: column sums
extern "C" __global__ void colsum_kernel(const float* src, float* dst, int rows, int cols) {
	for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < cols; j += blockDim.x * gridDim.x) {
		float sum = 0.0f;
		for (int r = 0; r < rows; ++r)
			sum += src[(size_t)r * cols + j];
		dst[j] += sum;
	}
}

// Conv bias gradient: one block reduces one row in shared memory
extern "C" __global__ void rowsum_kernel(const float* src, float* dst, int rows, int cols) {
	__shared__ float partial[256];
	int r = blockIdx.x;
	float sum = 0.0f;
	if (r < rows) {
		const float* row = src + (size_t)r * cols;
		for (int j = threadIdx.x; j < cols; j += blockDim.x)
			sum += row[j];
	}
	partial[threadIdx.x] = sum;
	__syncthreads();
	for (int step = blockDim.x / 2; step > 0; step /= 2) {
		if ((int)threadIdx.x < step)
			partial[threadIdx.x] += partial[threadIdx.x + step];
		__syncthreads();
	}
	if (threadIdx.x == 0 && r < rows)
		dst[r] += partial[0];
}
)CUDA";

	// 합성곱: im2col / col2im / 레이아웃 변환
	static const char* kKernelConv = R"CUDA(
// x: [nb, C, H, W]  ->  col: [C*K*K, nb*OH*OW]
extern "C" __global__ void im2col_kernel(const float* x, float* col, int C, int H, int W, int K, int S, int P, int OH, int OW, int nb) {
	int outPlane = OH * OW;
	int cols = nb * outPlane;
	int total = C * K * K * cols;
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += blockDim.x * gridDim.x) {
		int r = i / cols;
		int colIndex = i % cols;
		int b = colIndex / outPlane;
		int p = colIndex % outPlane;
		int oy = p / OW;
		int ox = p % OW;
		int c = r / (K * K);
		int kk = r % (K * K);
		int iy = oy * S - P + kk / K;
		int ix = ox * S - P + kk % K;
		float value = 0.0f;
		if (iy >= 0 && iy < H && ix >= 0 && ix < W)
			value = x[(((size_t)b * C + c) * H + iy) * W + ix];
		col[i] = value;
	}
}

// gather col back to the image. one thread per input pixel, so no atomics
extern "C" __global__ void col2im_kernel(const float* col, float* dx, int C, int H, int W, int K, int S, int P, int OH, int OW, int nb) {
	int outPlane = OH * OW;
	int cols = nb * outPlane;
	int total = nb * C * H * W;
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += blockDim.x * gridDim.x) {
		int ix = i % W;
		int iy = (i / W) % H;
		int c = (i / (W * H)) % C;
		int b = i / (W * H * C);
		float sum = 0.0f;
		for (int ky = 0; ky < K; ++ky) {
			int ty = iy + P - ky;
			if (ty < 0 || ty % S != 0)
				continue;
			int oy = ty / S;
			if (oy >= OH)
				continue;
			for (int kx = 0; kx < K; ++kx) {
				int tx = ix + P - kx;
				if (tx < 0 || tx % S != 0)
					continue;
				int ox = tx / S;
				if (ox >= OW)
					continue;
				int r = (c * K + ky) * K + kx;
				sum += col[(size_t)r * cols + b * outPlane + oy * OW + ox];
			}
		}
		dx[i] = sum;
	}
}

// GEMM result [Cout, nb*P] -> [nb, Cout, P] with bias + activation fused
extern "C" __global__ void cnhw_to_nchw_kernel(const float* src, const float* bias, float* dst, int Cout, int nb, int P, int act) {
	int total = Cout * nb * P;
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += blockDim.x * gridDim.x) {
		int p = i % P;
		int co = (i / P) % Cout;
		int b = i / (P * Cout);
		dst[i] = actForward(src[(size_t)co * nb * P + (size_t)b * P + p] + bias[co], act);
	}
}

extern "C" __global__ void nchw_to_cnhw_kernel(const float* src, float* dst, int Cout, int nb, int P) {
	int total = Cout * nb * P;
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += blockDim.x * gridDim.x) {
		int p = i % P;
		int b = (i / P) % nb;
		int co = i / (P * nb);
		dst[i] = src[((size_t)b * Cout + co) * P + p];
	}
}

extern "C" __global__ void pad_forward_kernel(const float* x, float* y, int planes, int H, int W, int P) {
	int OH = H + 2 * P;
	int OW = W + 2 * P;
	int total = planes * OH * OW;
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += blockDim.x * gridDim.x) {
		int ox = i % OW;
		int oy = (i / OW) % OH;
		int plane = i / (OW * OH);
		int iy = oy - P;
		int ix = ox - P;
		float value = 0.0f;
		if (iy >= 0 && iy < H && ix >= 0 && ix < W)
			value = x[((size_t)plane * H + iy) * W + ix];
		y[i] = value;
	}
}

extern "C" __global__ void pad_backward_kernel(const float* dy, float* dx, int planes, int H, int W, int P) {
	int OH = H + 2 * P;
	int OW = W + 2 * P;
	int total = planes * H * W;
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += blockDim.x * gridDim.x) {
		int ix = i % W;
		int iy = (i / W) % H;
		int plane = i / (W * H);
		dx[i] = dy[((size_t)plane * OH + iy + P) * OW + ix + P];
	}
}

// mode: 0 = Max, 1 = Average, 2 = Min. idx keeps the chosen position inside the plane
extern "C" __global__ void pool_forward_kernel(const float* x, float* y, float* idx, int planes, int H, int W, int OH, int OW, int size, int stride, int mode) {
	int total = planes * OH * OW;
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += blockDim.x * gridDim.x) {
		int ox = i % OW;
		int oy = (i / OW) % OH;
		int plane = i / (OW * OH);
		const float* src = x + (size_t)plane * H * W;
		int y0 = oy * stride;
		int x0 = ox * stride;
		int best = y0 * W + x0;
		float bestValue = src[best];
		float sum = 0.0f;
		for (int ky = 0; ky < size; ++ky) {
			for (int kx = 0; kx < size; ++kx) {
				int pos = (y0 + ky) * W + x0 + kx;
				float v = src[pos];
				sum += v;
				if (mode == 0 && v > bestValue) {
					bestValue = v;
					best = pos;
				}
				if (mode == 2 && v < bestValue) {
					bestValue = v;
					best = pos;
				}
			}
		}
		if (mode == 1)
			bestValue = sum / (float)(size * size);
		y[i] = bestValue;
		idx[i] = (float)best;
	}
}

// dx must be zeroed first (windows overlap when stride < size, hence atomicAdd)
extern "C" __global__ void pool_backward_kernel(const float* dy, const float* idx, float* dx, int planes, int H, int W, int OH, int OW, int size, int stride, int mode) {
	int total = planes * OH * OW;
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += blockDim.x * gridDim.x) {
		int ox = i % OW;
		int oy = (i / OW) % OH;
		int plane = i / (OW * OH);
		float* dst = dx + (size_t)plane * H * W;
		if (mode == 1) {
			float g = dy[i] / (float)(size * size);
			for (int ky = 0; ky < size; ++ky)
				for (int kx = 0; kx < size; ++kx)
					atomicAdd(&dst[(oy * stride + ky) * W + ox * stride + kx], g);
		}
		else {
			atomicAdd(&dst[(int)idx[i]], dy[i]);
		}
	}
}

extern "C" __global__ void maxout_forward_kernel(const float* z, float* y, float* idx, int rows, int units, int pieces) {
	int total = rows * units;
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += blockDim.x * gridDim.x) {
		const float* src = z + (size_t)i * pieces;
		int best = 0;
		for (int p = 1; p < pieces; ++p) {
			if (src[p] > src[best])
				best = p;
		}
		y[i] = src[best];
		idx[i] = (float)best;
	}
}

extern "C" __global__ void maxout_backward_kernel(const float* dy, const float* idx, float* dz, int rows, int units, int pieces) {
	int total = rows * units;
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += blockDim.x * gridDim.x) {
		float* dst = dz + (size_t)i * pieces;
		for (int p = 0; p < pieces; ++p)
			dst[p] = 0.0f;
		dst[(int)idx[i]] = dy[i];
	}
}
)CUDA";

	// 행렬 곱 (cuBLAS 가 없을 때 사용하는 타일링 GEMM) + 손실/옵티마이저
	static const char* kKernelMath = R"CUDA(
// row-major C = alpha * op(A) * op(B) + beta * C
// one block = 64x64 tile, one thread accumulates 4x4 results in registers
extern "C" __global__ void gemm_kernel(int transA, int transB, int M, int N, int K, float alpha, const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc) {
	__shared__ float As[8][64];
	__shared__ float Bs[8][64];
	int tid = threadIdx.x;
	int tx = tid % 16;
	int ty = tid / 16;
	int rowBase = blockIdx.y * 64;
	int colBase = blockIdx.x * 64;
	float acc[4][4];
	for (int i = 0; i < 4; ++i)
		for (int j = 0; j < 4; ++j)
			acc[i][j] = 0.0f;

	for (int k0 = 0; k0 < K; k0 += 8) {
		for (int e = tid; e < 512; e += 256) {
			int m = e / 8;
			int k = e % 8;
			int gm = rowBase + m;
			int gk = k0 + k;
			float value = 0.0f;
			if (gm < M && gk < K) {
				if (transA)
					value = A[(size_t)gk * lda + gm];
				else
					value = A[(size_t)gm * lda + gk];
			}
			As[k][m] = value;
		}
		for (int e = tid; e < 512; e += 256) {
			int k = e / 64;
			int n = e % 64;
			int gn = colBase + n;
			int gk = k0 + k;
			float value = 0.0f;
			if (gn < N && gk < K) {
				if (transB)
					value = B[(size_t)gn * ldb + gk];
				else
					value = B[(size_t)gk * ldb + gn];
			}
			Bs[k][n] = value;
		}
		__syncthreads();
		for (int k = 0; k < 8; ++k) {
			float a[4];
			float b[4];
			for (int i = 0; i < 4; ++i)
				a[i] = As[k][ty + 16 * i];
			for (int j = 0; j < 4; ++j)
				b[j] = Bs[k][tx + 16 * j];
			for (int i = 0; i < 4; ++i)
				for (int j = 0; j < 4; ++j)
					acc[i][j] += a[i] * b[j];
		}
		__syncthreads();
	}

	for (int i = 0; i < 4; ++i) {
		int r = rowBase + ty + 16 * i;
		if (r >= M)
			continue;
		for (int j = 0; j < 4; ++j) {
			int c = colBase + tx + 16 * j;
			if (c >= N)
				continue;
			float value = alpha * acc[i][j];
			if (beta != 0.0f)
				value += beta * C[(size_t)r * ldc + c];
			C[(size_t)r * ldc + c] = value;
		}
	}
}

// loss: 0 MSE, 1 RMSE, 2 BinaryCE, 3 CategoricalCE, 4 SparseCategoricalCE
// fused = 1: gradient already merged with the last Softmax/Sigmoid, i.e. (y - t)
extern "C" __global__ void loss_kernel(const float* y, const float* labels, float* grad, float* lossOut, int rows, int cols, int loss, int fused) {
	float invN = 1.0f / (float)rows;
	for (int r = blockIdx.x * blockDim.x + threadIdx.x; r < rows; r += blockDim.x * gridDim.x) {
		const float* yr = y + (size_t)r * cols;
		float* gr = grad + (size_t)r * cols;
		float label = labels[r];
		float total = 0.0f;
		for (int j = 0; j < cols; ++j) {
			float t = 0.0f;
			if (cols == 1)
				t = label;
			else if (j == (int)label)
				t = 1.0f;
			float v = yr[j];
			float g = 0.0f;
			if (loss == 0 || loss == 1) {
				total += (v - t) * (v - t);
				g = 2.0f * (v - t) * invN / (float)cols;
			}
			else if (loss == 2) {
				total -= (t * logf(v + 1e-7f) + (1.0f - t) * logf(1.0f - v + 1e-7f)) / (float)cols;
				if (fused)
					g = (v - t) * invN / (float)cols;
				else
					g = (v - t) / ((v + 1e-7f) * (1.0f - v + 1e-7f)) * invN / (float)cols;
			}
			else {
				if (t > 0.0f)
					total -= logf(v + 1e-7f);
				if (fused)
					g = (v - t) * invN;
				else
					g = -t / (v + 1e-7f) * invN;
			}
			gr[j] = g;
		}
		lossOut[r] = total;
	}
}

extern "C" __global__ void argmax_rows_kernel(const float* y, float* out, int rows, int cols) {
	for (int r = blockIdx.x * blockDim.x + threadIdx.x; r < rows; r += blockDim.x * gridDim.x) {
		const float* yr = y + (size_t)r * cols;
		int best = 0;
		for (int j = 1; j < cols; ++j) {
			if (yr[j] > yr[best])
				best = j;
		}
		if (cols == 1) {
			best = 0;
			if (yr[0] > 0.5f)
				best = 1;
		}
		out[r] = (float)best;
	}
}

// type: 0 SGD, 1 Momentum, 2 Adagrad, 3 RMSProp, 4 Adam, 5 Nadam
extern "C" __global__ void optimizer_kernel(float* w, const float* g, float* m, float* v, int n, int type, float lr, float beta1, float beta2, float eps, float bc1, float bc2, float bc1Next) {
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
		float grad = g[i];
		if (type == 0) {
			w[i] -= lr * grad;
		}
		else if (type == 1) {
			m[i] = beta1 * m[i] - lr * grad;
			w[i] += m[i];
		}
		else if (type == 2) {
			v[i] += grad * grad;
			w[i] -= lr * grad / (sqrtf(v[i]) + eps);
		}
		else if (type == 3) {
			v[i] = beta2 * v[i] + (1.0f - beta2) * grad * grad;
			w[i] -= lr * grad / (sqrtf(v[i]) + eps);
		}
		else {
			m[i] = beta1 * m[i] + (1.0f - beta1) * grad;
			v[i] = beta2 * v[i] + (1.0f - beta2) * grad * grad;
			float mHat = m[i] / bc1;
			if (type == 5)
				mHat = beta1 * m[i] / bc1Next + (1.0f - beta1) * grad / bc1;
			float vHat = v[i] / bc2;
			w[i] -= lr * mHat / (sqrtf(vHat) + eps);
		}
	}
}
)CUDA";

	static const char* kKernelNames[] = {
		"fill_kernel", "scale_kernel", "act_forward_kernel", "act_backward_kernel",
		"softmax_rows_kernel", "softmax_backward_kernel", "bias_act_rows_kernel",
		"colsum_kernel", "rowsum_kernel", "im2col_kernel", "col2im_kernel",
		"cnhw_to_nchw_kernel", "nchw_to_cnhw_kernel", "pad_forward_kernel", "pad_backward_kernel",
		"pool_forward_kernel", "pool_backward_kernel", "maxout_forward_kernel", "maxout_backward_kernel",
		"gemm_kernel", "loss_kernel", "argmax_rows_kernel", "optimizer_kernel"
	};
}
}

#endif
