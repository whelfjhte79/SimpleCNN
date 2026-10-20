// SimpleCNN - 연산 모음
//
//  cpu:: 네이티브 C++ 구현 (OpenMP 가 켜져 있으면 멀티코어)
//    gemm      : 캐시 타일링 행렬곱. 합성곱/FC 의 핵심 연산
//    im2col    : 합성곱을 행렬곱 하나로 바꾸는 펼치기
//    convFFT   : 주파수 영역 곱으로 합성곱 (큰 커널에서 유리)
//    convDirect: 정의 그대로의 합성곱 (검증/참고용)
//  gpu:: 같은 연산의 CUDA 버전 (CudaKernels.h 커널 호출)
#ifndef SIMPLE_CNN_OPS_H
#define SIMPLE_CNN_OPS_H

#include "Core.h"
#include "Tensor.h"

#include <map>

namespace cnn {
namespace cpu {

	// ---------------------------------------------------------------- GEMM
	inline float dot(const float* a, const float* b, int n) {
		float s[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
		int k = 0;
		for (; k + 8 <= n; k += 8) {
			for (int u = 0; u < 8; ++u)
				s[u] += a[k + u] * b[k + u];
		}
		float sum = (s[0] + s[1]) + (s[2] + s[3]) + (s[4] + s[5]) + (s[6] + s[7]);
		for (; k < n; ++k)
			sum += a[k] * b[k];
		return sum;
	}

	// 행 우선 C = alpha * op(A) * op(B) + beta * C
	// C 를 (32 x 256) 타일로 나눠 스레드에 분배하고, 안쪽 루프는 연속 메모리만 읽도록 배치했다.
	inline void gemm(bool transA, bool transB, int M, int N, int K, float alpha, const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc) {
		const int TM = 32;
		const int TN = 256;
		const int TK = 128;
		int tilesM = (M + TM - 1) / TM;
		int tilesN = (N + TN - 1) / TN;
		parallelFor(tilesM * tilesN, [&](int t) {
			int i0 = (t / tilesN) * TM;
			int j0 = (t % tilesN) * TN;
			int i1 = (std::min)(i0 + TM, M);
			int j1 = (std::min)(j0 + TN, N);
			for (int i = i0; i < i1; ++i) {
				float* c = C + (size_t)i * ldc;
				if (beta == 0.0f) {
					for (int j = j0; j < j1; ++j)
						c[j] = 0.0f;
				}
				else if (beta != 1.0f) {
					for (int j = j0; j < j1; ++j)
						c[j] *= beta;
				}
			}
			if (!transB) {
				for (int k0 = 0; k0 < K; k0 += TK) {
					int k1 = (std::min)(k0 + TK, K);
					for (int i = i0; i < i1; ++i) {
						float* c = C + (size_t)i * ldc;
						for (int k = k0; k < k1; ++k) {
							float a = 0.0f;
							if (transA)
								a = A[(size_t)k * lda + i];
							else
								a = A[(size_t)i * lda + k];
							// ReLU 뒤에는 0 이 많아서 건너뛰는 것만으로도 꽤 빨라진다.
							if (a == 0.0f)
								continue;
							a *= alpha;
							const float* b = B + (size_t)k * ldb;
							for (int j = j0; j < j1; ++j)
								c[j] += a * b[j];
						}
					}
				}
				return;
			}
			if (!transA) {
				for (int i = i0; i < i1; ++i) {
					const float* a = A + (size_t)i * lda;
					float* c = C + (size_t)i * ldc;
					for (int j = j0; j < j1; ++j)
						c[j] += alpha * dot(a, B + (size_t)j * ldb, K);
				}
				return;
			}
			for (int i = i0; i < i1; ++i) {
				float* c = C + (size_t)i * ldc;
				for (int j = j0; j < j1; ++j) {
					float sum = 0.0f;
					for (int k = 0; k < K; ++k)
						sum += A[(size_t)k * lda + i] * B[(size_t)j * ldb + k];
					c[j] += alpha * sum;
				}
			}
		});
	}

	// ---------------------------------------------------------------- 활성화
	inline float activate(float x, ACTIVATION act) {
		switch (act) {
		case ACTIVATION::ReLU:
			if (x < 0.0f)
				return 0.0f;
			return x;
		case ACTIVATION::Sigmoid:
			return 1.0f / (1.0f + std::exp(-x));
		case ACTIVATION::TanH:
			return std::tanh(x);
		case ACTIVATION::Leaky_ReLU:
			if (x < 0.0f)
				return kLeakySlope * x;
			return x;
		case ACTIVATION::ELU:
			if (x < 0.0f)
				return kEluAlpha * (std::exp(x) - 1.0f);
			return x;
		default:
			return x;
		}
	}
	// 출력 y 만으로 미분을 계산 (입력을 따로 저장하지 않는다)
	inline float activateGrad(float y, ACTIVATION act) {
		switch (act) {
		case ACTIVATION::ReLU:
			if (y > 0.0f)
				return 1.0f;
			return 0.0f;
		case ACTIVATION::Sigmoid:
			return y * (1.0f - y);
		case ACTIVATION::TanH:
			return 1.0f - y * y;
		case ACTIVATION::Leaky_ReLU:
			if (y > 0.0f)
				return 1.0f;
			return kLeakySlope;
		case ACTIVATION::ELU:
			if (y > 0.0f)
				return 1.0f;
			return y + kEluAlpha;
		default:
			return 1.0f;
		}
	}
	inline void activate(const float* x, float* y, size_t n, ACTIVATION act) {
		if (act == ACTIVATION::None || act == ACTIVATION::Softmax || act == ACTIVATION::Maxout) {
			if (x != y)
				std::memcpy(y, x, n * sizeof(float));
			return;
		}
		for (size_t i = 0; i < n; ++i)
			y[i] = activate(x[i], act);
	}
	inline void activateBackward(const float* y, float* dy, size_t n, ACTIVATION act) {
		if (act == ACTIVATION::None || act == ACTIVATION::Softmax || act == ACTIVATION::Maxout)
			return;
		for (size_t i = 0; i < n; ++i)
			dy[i] *= activateGrad(y[i], act);
	}
	inline void softmaxRows(float* y, int rows, int cols) {
		for (int r = 0; r < rows; ++r) {
			float* row = y + (size_t)r * cols;
			float maxValue = *std::max_element(row, row + cols);
			float sum = 0.0f;
			for (int j = 0; j < cols; ++j) {
				row[j] = std::exp(row[j] - maxValue);
				sum += row[j];
			}
			for (int j = 0; j < cols; ++j)
				row[j] /= sum;
		}
	}
	inline void softmaxBackward(const float* y, float* dy, int rows, int cols) {
		for (int r = 0; r < rows; ++r) {
			const float* yr = y + (size_t)r * cols;
			float* dr = dy + (size_t)r * cols;
			float d = dot(yr, dr, cols);
			for (int j = 0; j < cols; ++j)
				dr[j] = yr[j] * (dr[j] - d);
		}
	}

	// ---------------------------------------------------------------- im2col / col2im
	// x: [C, H, W] 한 장  ->  col: [C*K*K, OH*OW]
	inline void im2col(const float* x, int C, int H, int W, int K, int S, int P, int OH, int OW, float* col) {
		for (int c = 0; c < C; ++c) {
			for (int ky = 0; ky < K; ++ky) {
				for (int kx = 0; kx < K; ++kx) {
					float* row = col + (size_t)((c * K + ky) * K + kx) * OH * OW;
					for (int oy = 0; oy < OH; ++oy) {
						float* dst = row + oy * OW;
						int iy = oy * S - P + ky;
						if (iy < 0 || iy >= H) {
							std::fill(dst, dst + OW, 0.0f);
							continue;
						}
						const float* src = x + ((size_t)c * H + iy) * W;
						for (int ox = 0; ox < OW; ++ox) {
							int ix = ox * S - P + kx;
							if (ix >= 0 && ix < W)
								dst[ox] = src[ix];
							else
								dst[ox] = 0.0f;
						}
					}
				}
			}
		}
	}
	// col 을 이미지로 되돌리며 겹치는 위치는 더한다. dx 는 0 으로 시작해야 한다.
	inline void col2im(const float* col, int C, int H, int W, int K, int S, int P, int OH, int OW, float* dx) {
		for (int c = 0; c < C; ++c) {
			for (int ky = 0; ky < K; ++ky) {
				for (int kx = 0; kx < K; ++kx) {
					const float* row = col + (size_t)((c * K + ky) * K + kx) * OH * OW;
					for (int oy = 0; oy < OH; ++oy) {
						int iy = oy * S - P + ky;
						if (iy < 0 || iy >= H)
							continue;
						float* dst = dx + ((size_t)c * H + iy) * W;
						const float* src = row + oy * OW;
						for (int ox = 0; ox < OW; ++ox) {
							int ix = ox * S - P + kx;
							if (ix >= 0 && ix < W)
								dst[ix] += src[ox];
						}
					}
				}
			}
		}
	}

	// ---------------------------------------------------------------- FFT
	struct FFTPlan {
		int n = 0;
		std::vector<int> reverse;
		std::vector<float> cosTable;
		std::vector<float> sinTable;
	};
	inline const FFTPlan& fftPlan(int n) {
		static thread_local std::map<int, FFTPlan> plans;
		FFTPlan& plan = plans[n];
		if (plan.n == n)
			return plan;
		plan.n = n;
		plan.reverse.resize(n);
		int bits = 0;
		while ((1 << bits) < n)
			++bits;
		for (int i = 0; i < n; ++i) {
			int r = 0;
			for (int b = 0; b < bits; ++b) {
				if (i & (1 << b))
					r |= 1 << (bits - 1 - b);
			}
			plan.reverse[i] = r;
		}
		plan.cosTable.resize(n / 2 + 1);
		plan.sinTable.resize(n / 2 + 1);
		const double pi = 3.14159265358979323846;
		for (int i = 0; i <= n / 2; ++i) {
			plan.cosTable[i] = (float)std::cos(2.0 * pi * i / n);
			plan.sinTable[i] = (float)std::sin(2.0 * pi * i / n);
		}
		return plan;
	}
	// 반복형 radix-2 FFT (제자리). inverse 는 정규화(1/n)를 하지 않는다.
	inline void fft(float* re, float* im, int n, bool inverse) {
		const FFTPlan& plan = fftPlan(n);
		for (int i = 0; i < n; ++i) {
			int j = plan.reverse[i];
			if (i < j) {
				std::swap(re[i], re[j]);
				std::swap(im[i], im[j]);
			}
		}
		float sign = -1.0f;
		if (inverse)
			sign = 1.0f;
		for (int len = 2; len <= n; len <<= 1) {
			int half = len / 2;
			int step = n / len;
			for (int i = 0; i < n; i += len) {
				for (int k = 0; k < half; ++k) {
					float wr = plan.cosTable[k * step];
					float wi = sign * plan.sinTable[k * step];
					int a = i + k;
					int b = a + half;
					float tr = re[b] * wr - im[b] * wi;
					float ti = re[b] * wi + im[b] * wr;
					re[b] = re[a] - tr;
					im[b] = im[a] - ti;
					re[a] += tr;
					im[a] += ti;
				}
			}
		}
	}
	// 2차원 FFT: 행 → 열. tmp 는 길이 2*rows 이상
	inline void fft2d(float* re, float* im, int rows, int cols, bool inverse, float* tmp) {
		for (int r = 0; r < rows; ++r)
			fft(re + (size_t)r * cols, im + (size_t)r * cols, cols, inverse);
		float* colRe = tmp;
		float* colIm = tmp + rows;
		for (int c = 0; c < cols; ++c) {
			for (int r = 0; r < rows; ++r) {
				colRe[r] = re[(size_t)r * cols + c];
				colIm[r] = im[(size_t)r * cols + c];
			}
			fft(colRe, colIm, rows, inverse);
			for (int r = 0; r < rows; ++r) {
				re[(size_t)r * cols + c] = colRe[r];
				im[(size_t)r * cols + c] = colIm[r];
			}
		}
	}
	inline int nextPow2(int v) {
		int p = 1;
		while (p < v)
			p <<= 1;
		return p;
	}
	// FFT 합성곱에 필요한 메모리 (커널 스펙트럼 + 스레드별 버퍼)
	inline size_t fftWorkspaceBytes(int Cin, int Cout, int H, int W, int P) {
		size_t plane = (size_t)nextPow2(H + 2 * P) * nextPow2(W + 2 * P);
		size_t kernels = (size_t)Cout * Cin * plane * 2;
		size_t perThread = ((size_t)Cin + 1) * plane * 2 + plane * 2;
		return (kernels + perThread * threadCount()) * sizeof(float);
	}

	// FFT 합성곱. 상관(cross-correlation)이므로 커널을 뒤집어 넣고,
	// 원형 합성곱 결과에서 (K-1) 만큼 밀린 위치가 valid 출력이다.
	inline void convFFT(const float* x, int N, int Cin, int H, int W, const float* weight, const float* bias, int Cout, int K, int S, int P, int OH, int OW, ACTIVATION act, float* y, std::vector<float>& buffer) {
		int PH = nextPow2(H + 2 * P);
		int PW = nextPow2(W + 2 * P);
		size_t plane = (size_t)PH * PW;
		int threads = threadCount();
		size_t kernelFloats = (size_t)Cout * Cin * plane * 2;
		size_t perThread = ((size_t)Cin + 1) * plane * 2 + (size_t)PH * 2;
		buffer.resize(kernelFloats + perThread * threads);
		float* kernelRe = buffer.data();
		float* kernelIm = kernelRe + (size_t)Cout * Cin * plane;

		parallelFor(Cout * Cin, [&](int index) {
			float* re = kernelRe + (size_t)index * plane;
			float* im = kernelIm + (size_t)index * plane;
			std::fill(re, re + plane, 0.0f);
			std::fill(im, im + plane, 0.0f);
			const float* w = weight + (size_t)index * K * K;
			for (int ky = 0; ky < K; ++ky)
				for (int kx = 0; kx < K; ++kx)
					re[ky * PW + kx] = w[(K - 1 - ky) * K + (K - 1 - kx)];
			std::vector<float> tmp(PH * 2);
			fft2d(re, im, PH, PW, false, tmp.data());
		});

		float scale = 1.0f / (float)plane;
		parallelFor(N, [&](int n) {
			float* base = kernelRe + kernelFloats + perThread * threadIndex();
			float* inRe = base;
			float* inIm = inRe + (size_t)Cin * plane;
			float* accRe = inIm + (size_t)Cin * plane;
			float* accIm = accRe + plane;
			float* tmp = accIm + plane;
			for (int c = 0; c < Cin; ++c) {
				float* re = inRe + (size_t)c * plane;
				float* im = inIm + (size_t)c * plane;
				std::fill(re, re + plane, 0.0f);
				std::fill(im, im + plane, 0.0f);
				const float* src = x + ((size_t)n * Cin + c) * H * W;
				for (int iy = 0; iy < H; ++iy)
					std::memcpy(re + (size_t)(iy + P) * PW + P, src + (size_t)iy * W, W * sizeof(float));
				fft2d(re, im, PH, PW, false, tmp);
			}
			for (int co = 0; co < Cout; ++co) {
				std::fill(accRe, accRe + plane, 0.0f);
				std::fill(accIm, accIm + plane, 0.0f);
				for (int c = 0; c < Cin; ++c) {
					const float* ar = inRe + (size_t)c * plane;
					const float* ai = inIm + (size_t)c * plane;
					const float* br = kernelRe + ((size_t)co * Cin + c) * plane;
					const float* bi = kernelIm + ((size_t)co * Cin + c) * plane;
					for (size_t i = 0; i < plane; ++i) {
						accRe[i] += ar[i] * br[i] - ai[i] * bi[i];
						accIm[i] += ar[i] * bi[i] + ai[i] * br[i];
					}
				}
				fft2d(accRe, accIm, PH, PW, true, tmp);
				float* dst = y + ((size_t)n * Cout + co) * OH * OW;
				for (int oy = 0; oy < OH; ++oy) {
					const float* row = accRe + (size_t)(oy * S + K - 1) * PW + K - 1;
					for (int ox = 0; ox < OW; ++ox)
						dst[oy * OW + ox] = activate(row[ox * S] * scale + bias[co], act);
				}
			}
		});
	}

	// 정의 그대로의 합성곱 (정답 확인용)
	inline void convDirect(const float* x, int N, int Cin, int H, int W, const float* weight, const float* bias, int Cout, int K, int S, int P, int OH, int OW, ACTIVATION act, float* y) {
		parallelFor(N * Cout, [&](int index) {
			int n = index / Cout;
			int co = index % Cout;
			float* dst = y + (size_t)index * OH * OW;
			for (int oy = 0; oy < OH; ++oy) {
				for (int ox = 0; ox < OW; ++ox) {
					float sum = bias[co];
					for (int c = 0; c < Cin; ++c) {
						const float* src = x + ((size_t)n * Cin + c) * H * W;
						const float* w = weight + ((size_t)co * Cin + c) * K * K;
						for (int ky = 0; ky < K; ++ky) {
							int iy = oy * S - P + ky;
							if (iy < 0 || iy >= H)
								continue;
							for (int kx = 0; kx < K; ++kx) {
								int ix = ox * S - P + kx;
								if (ix >= 0 && ix < W)
									sum += src[iy * W + ix] * w[ky * K + kx];
							}
						}
					}
					dst[oy * OW + ox] = activate(sum, act);
				}
			}
		});
	}

	// ---------------------------------------------------------------- 풀링
	inline void poolForward(const float* x, float* y, float* index, int planes, int H, int W, int OH, int OW, int size, int stride, POOLING mode) {
		float area = (float)(size * size);
		parallelFor(planes, [&](int p) {
			const float* src = x + (size_t)p * H * W;
			float* dst = y + (size_t)p * OH * OW;
			float* idx = index + (size_t)p * OH * OW;
			for (int oy = 0; oy < OH; ++oy) {
				for (int ox = 0; ox < OW; ++ox) {
					int best = oy * stride * W + ox * stride;
					float bestValue = src[best];
					float sum = 0.0f;
					for (int ky = 0; ky < size; ++ky) {
						for (int kx = 0; kx < size; ++kx) {
							int pos = (oy * stride + ky) * W + ox * stride + kx;
							float v = src[pos];
							sum += v;
							if (mode == POOLING::Max && v > bestValue) {
								bestValue = v;
								best = pos;
							}
							if (mode == POOLING::Min && v < bestValue) {
								bestValue = v;
								best = pos;
							}
						}
					}
					if (mode == POOLING::Average)
						bestValue = sum / area;
					dst[oy * OW + ox] = bestValue;
					idx[oy * OW + ox] = (float)best;
				}
			}
		});
	}
	inline void poolBackward(const float* dy, const float* index, float* dx, int planes, int H, int W, int OH, int OW, int size, int stride, POOLING mode) {
		float area = (float)(size * size);
		parallelFor(planes, [&](int p) {
			const float* src = dy + (size_t)p * OH * OW;
			const float* idx = index + (size_t)p * OH * OW;
			float* dst = dx + (size_t)p * H * W;
			std::fill(dst, dst + (size_t)H * W, 0.0f);
			for (int oy = 0; oy < OH; ++oy) {
				for (int ox = 0; ox < OW; ++ox) {
					float g = src[oy * OW + ox];
					if (mode == POOLING::Average) {
						for (int ky = 0; ky < size; ++ky)
							for (int kx = 0; kx < size; ++kx)
								dst[(oy * stride + ky) * W + ox * stride + kx] += g / area;
					}
					else {
						dst[(int)idx[oy * OW + ox]] += g;
					}
				}
			}
		});
	}
}

namespace gpu {
	inline cuda::Runtime& rt() { return cuda::Runtime::instance(); }

	inline void activate(const float* x, float* y, int n, ACTIVATION act) {
		rt().launch("act_forward_kernel", n, x, y, n, (int)act);
	}
	inline void activateBackward(const float* y, float* dy, int n, ACTIVATION act) {
		if (act == ACTIVATION::None || act == ACTIVATION::Softmax || act == ACTIVATION::Maxout)
			return;
		rt().launch("act_backward_kernel", n, y, dy, n, (int)act);
	}
	inline void softmaxRows(float* y, int rows, int cols) {
		rt().launch("softmax_rows_kernel", rows, y, rows, cols);
	}
	inline void softmaxBackward(const float* y, float* dy, int rows, int cols) {
		rt().launch("softmax_backward_kernel", rows, y, dy, rows, cols);
	}
	inline void biasActRows(float* y, const float* bias, int rows, int cols, ACTIVATION act) {
		rt().launch("bias_act_rows_kernel", (size_t)rows * cols, y, bias, rows, cols, (int)act);
	}
	inline void colSum(const float* src, float* dst, int rows, int cols) {
		rt().launch("colsum_kernel", cols, src, dst, rows, cols);
	}
	inline void rowSum(const float* src, float* dst, int rows, int cols) {
		rt().launchGrid("rowsum_kernel", (unsigned)rows, 1, 256, src, dst, rows, cols);
	}
	inline void im2col(const float* x, float* col, int C, int H, int W, int K, int S, int P, int OH, int OW, int nb) {
		size_t total = (size_t)C * K * K * nb * OH * OW;
		rt().launch("im2col_kernel", total, x, col, C, H, W, K, S, P, OH, OW, nb);
	}
	inline void col2im(const float* col, float* dx, int C, int H, int W, int K, int S, int P, int OH, int OW, int nb) {
		size_t total = (size_t)nb * C * H * W;
		rt().launch("col2im_kernel", total, col, dx, C, H, W, K, S, P, OH, OW, nb);
	}
	inline void cnhwToNchw(const float* src, const float* bias, float* dst, int Cout, int nb, int P, ACTIVATION act) {
		rt().launch("cnhw_to_nchw_kernel", (size_t)Cout * nb * P, src, bias, dst, Cout, nb, P, (int)act);
	}
	inline void nchwToCnhw(const float* src, float* dst, int Cout, int nb, int P) {
		rt().launch("nchw_to_cnhw_kernel", (size_t)Cout * nb * P, src, dst, Cout, nb, P);
	}
	inline void padForward(const float* x, float* y, int planes, int H, int W, int P) {
		rt().launch("pad_forward_kernel", (size_t)planes * (H + 2 * P) * (W + 2 * P), x, y, planes, H, W, P);
	}
	inline void padBackward(const float* dy, float* dx, int planes, int H, int W, int P) {
		rt().launch("pad_backward_kernel", (size_t)planes * H * W, dy, dx, planes, H, W, P);
	}
	inline int poolMode(POOLING mode) {
		if (mode == POOLING::Average)
			return 1;
		if (mode == POOLING::Min)
			return 2;
		return 0;
	}
	inline void poolForward(const float* x, float* y, float* index, int planes, int H, int W, int OH, int OW, int size, int stride, POOLING mode) {
		rt().launch("pool_forward_kernel", (size_t)planes * OH * OW, x, y, index, planes, H, W, OH, OW, size, stride, poolMode(mode));
	}
	inline void poolBackward(const float* dy, const float* index, float* dx, int planes, int H, int W, int OH, int OW, int size, int stride, POOLING mode) {
		rt().launch("pool_backward_kernel", (size_t)planes * OH * OW, dy, index, dx, planes, H, W, OH, OW, size, stride, poolMode(mode));
	}
	inline void maxoutForward(const float* z, float* y, float* index, int rows, int units, int pieces) {
		rt().launch("maxout_forward_kernel", (size_t)rows * units, z, y, index, rows, units, pieces);
	}
	inline void maxoutBackward(const float* dy, const float* index, float* dz, int rows, int units, int pieces) {
		rt().launch("maxout_backward_kernel", (size_t)rows * units, dy, index, dz, rows, units, pieces);
	}
}
}

#endif
