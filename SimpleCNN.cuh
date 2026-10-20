// SimpleCNN CUDA 커널
//
// 이 파일은 SimpleCNN.h 가 문자열로 읽어 들인 뒤, 실행 중에 NVRTC 로 "지금 꽂혀 있는 GPU" 에 맞게 컴파일한다.
// 그래서 nvcc 나 CUDA 빌드 설정 없이 #include "SimpleCNN.h" 만으로 GPU 가 동작한다.
//
// 맨 위 R"CUDA( 와 맨 아래 )CUDA" 는 C++ 문자열 표시이고, 그 사이는 그대로 CUDA 코드다.
// 중간의 )CUDA" R"CUDA( 는 MSVC 문자열 길이 제한(16KB) 때문에 나눈 자리일 뿐이다.
// 문자열 안은 NVRTC 가 읽으므로 주석도 영어(ASCII)로만 쓴다.
R"CUDA(
// activation codes = order of cnn::ACTIVATION
#define ACT_NONE 0
#define ACT_SIGMOID 2
#define ACT_RELU 3
#define ACT_LEAKY_RELU 4
#define ACT_TANH 5
#define ACT_ELU 7
#define LEAKY_SLOPE 0.01f
#define ELU_ALPHA 1.0f

// every kernel walks its data with a grid-stride loop
#define FOR_EACH(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

__device__ float actForward(float x, int act) {
	if (act == ACT_RELU)
		return fmaxf(x, 0.0f);
	if (act == ACT_SIGMOID)
		return 1.0f / (1.0f + expf(-x));
	if (act == ACT_TANH)
		return tanhf(x);
	if (act == ACT_LEAKY_RELU && x < 0.0f)
		return LEAKY_SLOPE * x;
	if (act == ACT_ELU && x < 0.0f)
		return ELU_ALPHA * (expf(x) - 1.0f);
	return x;
}

// derivative computed from the output y (the input does not have to be kept)
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
	if (act == ACT_LEAKY_RELU && y <= 0.0f)
		return LEAKY_SLOPE;
	if (act == ACT_ELU && y <= 0.0f)
		return y + ELU_ALPHA;
	return 1.0f;
}

extern "C" __global__ void fill_kernel(float* x, float value, int n) {
	FOR_EACH(i, n)
		x[i] = value;
}

extern "C" __global__ void scale_kernel(float* x, float s, int n) {
	FOR_EACH(i, n)
		x[i] *= s;
}

extern "C" __global__ void act_forward_kernel(const float* x, float* y, int n, int act) {
	FOR_EACH(i, n)
		y[i] = actForward(x[i], act);
}

extern "C" __global__ void act_backward_kernel(const float* y, float* dy, int n, int act) {
	FOR_EACH(i, n)
		dy[i] *= actGrad(y[i], act);
}

extern "C" __global__ void softmax_kernel(float* y, int rows, int cols) {
	FOR_EACH(r, rows) {
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
	FOR_EACH(r, rows) {
		const float* yr = y + (size_t)r * cols;
		float* dr = dy + (size_t)r * cols;
		float dot = 0.0f;
		for (int j = 0; j < cols; ++j)
			dot += yr[j] * dr[j];
		for (int j = 0; j < cols; ++j)
			dr[j] = yr[j] * (dr[j] - dot);
	}
}

// FC: y = act(y + bias)
extern "C" __global__ void bias_act_kernel(float* y, const float* bias, int rows, int cols, int act) {
	FOR_EACH(i, rows * cols)
		y[i] = actForward(y[i] + bias[i % cols], act);
}

// FC bias gradient: dst[j] += sum over rows
extern "C" __global__ void colsum_kernel(const float* src, float* dst, int rows, int cols) {
	FOR_EACH(j, cols) {
		float sum = 0.0f;
		for (int r = 0; r < rows; ++r)
			sum += src[(size_t)r * cols + j];
		dst[j] += sum;
	}
}

// Conv bias gradient: dst[r] += sum of row r (one block per row, shared memory reduction)
extern "C" __global__ void rowsum_kernel(const float* src, float* dst, int rows, int cols) {
	__shared__ float partial[256];
	int r = blockIdx.x;
	float sum = 0.0f;
	for (int j = threadIdx.x; j < cols; j += blockDim.x)
		sum += src[(size_t)r * cols + j];
	partial[threadIdx.x] = sum;
	__syncthreads();
	for (int step = blockDim.x / 2; step > 0; step /= 2) {
		if ((int)threadIdx.x < step)
			partial[threadIdx.x] += partial[threadIdx.x + step];
		__syncthreads();
	}
	if (threadIdx.x == 0)
		dst[r] += partial[0];
}
)CUDA" R"CUDA(
// x: [nb, C, H, W] -> col: [C*K*K, nb*OH*OW]
extern "C" __global__ void im2col_kernel(const float* x, float* col, int C, int H, int W, int K, int S, int P, int OH, int OW, int nb) {
	int plane = OH * OW;
	int cols = nb * plane;
	FOR_EACH(i, C * K * K * cols) {
		int r = i / cols;
		int b = (i % cols) / plane;
		int p = (i % cols) % plane;
		int c = r / (K * K);
		int iy = (p / OW) * S - P + (r % (K * K)) / K;
		int ix = (p % OW) * S - P + (r % (K * K)) % K;
		float value = 0.0f;
		if (iy >= 0 && iy < H && ix >= 0 && ix < W)
			value = x[(((size_t)b * C + c) * H + iy) * W + ix];
		col[i] = value;
	}
}

// col -> image. one thread per input pixel gathers its values, so no atomics are needed
extern "C" __global__ void col2im_kernel(const float* col, float* dx, int C, int H, int W, int K, int S, int P, int OH, int OW, int nb) {
	int plane = OH * OW;
	int cols = nb * plane;
	FOR_EACH(i, nb * C * H * W) {
		int ix = i % W;
		int iy = (i / W) % H;
		int c = (i / (W * H)) % C;
		int b = i / (W * H * C);
		float sum = 0.0f;
		for (int ky = 0; ky < K; ++ky) {
			int ty = iy + P - ky;
			if (ty < 0 || ty % S != 0 || ty / S >= OH)
				continue;
			for (int kx = 0; kx < K; ++kx) {
				int tx = ix + P - kx;
				if (tx < 0 || tx % S != 0 || tx / S >= OW)
					continue;
				int r = (c * K + ky) * K + kx;
				sum += col[(size_t)r * cols + b * plane + (ty / S) * OW + tx / S];
			}
		}
		dx[i] = sum;
	}
}

// GEMM result [Cout, nb*P] -> output [nb, Cout, P], adding bias and activation on the way
extern "C" __global__ void conv_output_kernel(const float* src, const float* bias, float* dst, int Cout, int nb, int P, int act) {
	FOR_EACH(i, Cout * nb * P) {
		int p = i % P;
		int co = (i / P) % Cout;
		int b = i / (P * Cout);
		dst[i] = actForward(src[(size_t)co * nb * P + (size_t)b * P + p] + bias[co], act);
	}
}

// gradient [nb, Cout, P] -> [Cout, nb*P] (the layout GEMM wants)
extern "C" __global__ void conv_grad_kernel(const float* src, float* dst, int Cout, int nb, int P) {
	FOR_EACH(i, Cout * nb * P) {
		int p = i % P;
		int b = (i / P) % nb;
		int co = i / (P * nb);
		dst[i] = src[((size_t)b * Cout + co) * P + p];
	}
}

extern "C" __global__ void pad_forward_kernel(const float* x, float* y, int planes, int H, int W, int P) {
	int OH = H + 2 * P;
	int OW = W + 2 * P;
	FOR_EACH(i, planes * OH * OW) {
		int iy = (i / OW) % OH - P;
		int ix = i % OW - P;
		int plane = i / (OW * OH);
		float value = 0.0f;
		if (iy >= 0 && iy < H && ix >= 0 && ix < W)
			value = x[((size_t)plane * H + iy) * W + ix];
		y[i] = value;
	}
}

extern "C" __global__ void pad_backward_kernel(const float* dy, float* dx, int planes, int H, int W, int P) {
	int OW = W + 2 * P;
	int OH = H + 2 * P;
	FOR_EACH(i, planes * H * W) {
		int ix = i % W;
		int iy = (i / W) % H;
		int plane = i / (W * H);
		dx[i] = dy[((size_t)plane * OH + iy + P) * OW + ix + P];
	}
}

// mode: 0 Max, 1 Average, 2 Min. index keeps the chosen position inside the plane
extern "C" __global__ void pool_forward_kernel(const float* x, float* y, float* index, int planes, int H, int W, int OH, int OW, int size, int stride, int mode) {
	FOR_EACH(i, planes * OH * OW) {
		const float* src = x + (size_t)(i / (OW * OH)) * H * W;
		int y0 = ((i / OW) % OH) * stride;
		int x0 = (i % OW) * stride;
		int best = y0 * W + x0;
		float sum = 0.0f;
		for (int ky = 0; ky < size; ++ky) {
			for (int kx = 0; kx < size; ++kx) {
				int pos = (y0 + ky) * W + x0 + kx;
				sum += src[pos];
				if (mode == 0 && src[pos] > src[best])
					best = pos;
				if (mode == 2 && src[pos] < src[best])
					best = pos;
			}
		}
		y[i] = src[best];
		if (mode == 1)
			y[i] = sum / (float)(size * size);
		index[i] = (float)best;
	}
}

// dx must be zero first. windows overlap when stride < size, hence atomicAdd
extern "C" __global__ void pool_backward_kernel(const float* dy, const float* index, float* dx, int planes, int H, int W, int OH, int OW, int size, int stride, int mode) {
	FOR_EACH(i, planes * OH * OW) {
		float* dst = dx + (size_t)(i / (OW * OH)) * H * W;
		if (mode != 1) {
			atomicAdd(&dst[(int)index[i]], dy[i]);
			continue;
		}
		int y0 = ((i / OW) % OH) * stride;
		int x0 = (i % OW) * stride;
		float g = dy[i] / (float)(size * size);
		for (int ky = 0; ky < size; ++ky)
			for (int kx = 0; kx < size; ++kx)
				atomicAdd(&dst[(y0 + ky) * W + x0 + kx], g);
	}
}

// Maxout: each unit takes the larger of its 2 linear outputs
extern "C" __global__ void maxout_forward_kernel(const float* z, float* y, float* index, int n) {
	FOR_EACH(i, n) {
		int best = 0;
		if (z[2 * i + 1] > z[2 * i])
			best = 1;
		y[i] = z[2 * i + best];
		index[i] = (float)best;
	}
}

extern "C" __global__ void maxout_backward_kernel(const float* dy, const float* index, float* dz, int n) {
	FOR_EACH(i, n) {
		dz[2 * i] = 0.0f;
		dz[2 * i + 1] = 0.0f;
		dz[2 * i + (int)index[i]] = dy[i];
	}
}
)CUDA" R"CUDA(
// row-major C = alpha * op(A) * op(B) + beta * C  (used when cuBLAS is not installed)
// one block computes a 64x64 tile, each of the 256 threads accumulates 4x4 results in registers
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
			int m = rowBase + e / 8;
			int k = k0 + e % 8;
			float value = 0.0f;
			if (m < M && k < K && transA)
				value = A[(size_t)k * lda + m];
			if (m < M && k < K && !transA)
				value = A[(size_t)m * lda + k];
			As[e % 8][e / 8] = value;
		}
		for (int e = tid; e < 512; e += 256) {
			int k = k0 + e / 64;
			int n = colBase + e % 64;
			float value = 0.0f;
			if (n < N && k < K && transB)
				value = B[(size_t)n * ldb + k];
			if (n < N && k < K && !transB)
				value = B[(size_t)k * ldb + n];
			Bs[e / 64][e % 64] = value;
		}
		__syncthreads();
		for (int k = 0; k < 8; ++k)
			for (int i = 0; i < 4; ++i)
				for (int j = 0; j < 4; ++j)
					acc[i][j] += As[k][ty + 16 * i] * Bs[k][tx + 16 * j];
		__syncthreads();
	}

	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 4; ++j) {
			int r = rowBase + ty + 16 * i;
			int c = colBase + tx + 16 * j;
			if (r >= M || c >= N)
				continue;
			float value = alpha * acc[i][j];
			if (beta != 0.0f)
				value += beta * C[(size_t)r * ldc + c];
			C[(size_t)r * ldc + c] = value;
		}
	}
}

// loss: 0 MSE, 1 RMSE, 2 BinaryCE, 3 CategoricalCE, 4 SparseCategoricalCE
// fused = 1: the last Softmax/Sigmoid derivative is merged in, so the gradient is simply (y - t)
extern "C" __global__ void loss_kernel(const float* y, const float* labels, float* grad, float* loss, float* predicted, int rows, int cols, int type, int fused) {
	FOR_EACH(r, rows) {
		const float* yr = y + (size_t)r * cols;
		float* gr = grad + (size_t)r * cols;
		float total = 0.0f;
		int best = 0;
		for (int j = 0; j < cols; ++j) {
			float t = 0.0f;
			if (cols == 1)
				t = labels[r];
			if (cols > 1 && j == (int)labels[r])
				t = 1.0f;
			float v = yr[j];
			float g = 0.0f;
			if (type <= 1) {
				total += (v - t) * (v - t);
				g = 2.0f * (v - t) / cols;
			}
			else if (type == 2) {
				total -= (t * logf(v + 1e-7f) + (1.0f - t) * logf(1.0f - v + 1e-7f)) / cols;
				g = (v - t) / ((v + 1e-7f) * (1.0f - v + 1e-7f)) / cols;
				if (fused)
					g = (v - t) / cols;
			}
			else {
				if (t > 0.0f)
					total -= logf(v + 1e-7f);
				g = -t / (v + 1e-7f);
				if (fused)
					g = v - t;
			}
			gr[j] = g / rows;
			if (v > yr[best])
				best = j;
		}
		if (cols == 1)
			best = (int)(yr[0] > 0.5f);
		loss[r] = total;
		predicted[r] = (float)best;
	}
}

// type: 0 SGD, 1 Momentum, 2 Adagrad, 3 RMSProp, 4 Adam, 5 Nadam
// bc1 = 1 - beta1^t, bc2 = 1 - beta2^t, bc1Next = 1 - beta1^(t+1)
extern "C" __global__ void optimizer_kernel(float* w, const float* g, float* m, float* v, int n, int type, float lr, float beta1, float beta2, float eps, float bc1, float bc2, float bc1Next) {
	FOR_EACH(i, n) {
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
			w[i] -= lr * mHat / (sqrtf(v[i] / bc2) + eps);
		}
	}
}
)CUDA"
