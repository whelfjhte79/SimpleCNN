// SimpleCNN - 레이어 (블럭)
//
// 모든 레이어는 같은 모양의 블럭이다.
//   build    : 앞 블럭의 출력 모양을 받아 자기 출력 모양을 정한다 (입력 크기를 사용자가 몰라도 됨)
//   forward  : x → y
//   backward : dy → dx (+ 자기 파라미터의 기울기)
// 새 레이어를 만들 때도 이 세 가지만 구현하면 CNN 에 그대로 끼워진다.
#ifndef SIMPLE_CNN_LAYERS_H
#define SIMPLE_CNN_LAYERS_H

#include "Core.h"
#include "Ops.h"
#include "Tensor.h"

namespace cnn {

	// 학습되는 값 하나 (가중치 또는 바이어스)
	struct Param {
		Tensor value;
		Tensor grad;
		Tensor m; // 옵티마이저 상태 (필요한 옵티마이저일 때만 할당)
		Tensor v;

		size_t size() const { return this->value.size(); }
		void moveTo(Device device) {
			this->value.moveTo(device);
			this->m.moveTo(device);
			this->v.moveTo(device);
			this->grad.resize(this->value.shape, device);
		}
	};

	// 레이어들이 함께 쓰는 실행 정보
	struct Context {
		Device device = Device::CPU;
		bool needInputGrad = true;   // 첫 레이어는 입력 기울기가 필요 없다
		bool fusedOutput = false;    // 손실 함수가 Softmax/Sigmoid 미분까지 합쳐서 넘겨준 경우
		CONV_ALGO convAlgo = CONV_ALGO::Auto;
		size_t workspaceLimit = (size_t)512 << 20;
		bool verbose = true;
		Tensor workspace;            // im2col 등 임시 버퍼. 모든 레이어가 하나를 같이 쓴다.
	};

	class Layer {
	public:
		LAYER layerType = LAYER::None;
		ACTIVATION act = ACTIVATION::None;
		Shape inShape;
		Shape outShape;

		virtual ~Layer() {}
		virtual std::string name() const = 0;
		// 저장 파일에 들어가는 한 줄 설정 (예: "Conv 3 16 1 1 ReLU")
		virtual std::string config() const = 0;
		virtual Shape build(const Shape& in, std::mt19937& rng) = 0;
		virtual void forward(const Tensor& x, Tensor& y, Context& ctx) = 0;
		virtual void backward(const Tensor& x, const Tensor& y, Tensor& dy, Tensor& dx, Context& ctx) = 0;
		virtual std::vector<Param*> params() { return std::vector<Param*>(); }
		virtual std::string detail() const { return ""; }

		size_t paramCount() {
			size_t total = 0;
			for (Param* p : this->params())
				total += p->size();
			return total;
		}
		void toDevice(Device device) {
			for (Param* p : this->params())
				p->moveTo(device);
		}
	protected:
		// He(ReLU 계열) / Xavier 균등분포 초기화. 이미 불러온 값이 있으면 건드리지 않는다.
		static void initParam(Param& p, size_t count, int fanIn, int fanOut, ACTIVATION act, std::mt19937& rng, bool isBias) {
			if (p.value.size() == count)
				return;
			std::vector<float> host(count, 0.0f);
			if (!isBias) {
				float limit = std::sqrt(6.0f / (float)(fanIn + fanOut));
				if (act == ACTIVATION::ReLU || act == ACTIVATION::Leaky_ReLU || act == ACTIVATION::ELU)
					limit = std::sqrt(6.0f / (float)fanIn);
				std::uniform_real_distribution<float> dist(-limit, limit);
				for (size_t i = 0; i < count; ++i)
					host[i] = dist(rng);
			}
			p.value.resize(count, Device::CPU);
			p.value.upload(host.data(), count);
			p.grad.resize(count, Device::CPU);
		}
		static void addBiasActivate(float* y, const float* bias, int rows, int cols, ACTIVATION act) {
			parallelFor(rows, [&](int r) {
				float* row = y + (size_t)r * cols;
				for (int j = 0; j < cols; ++j)
					row[j] = cpu::activate(row[j] + bias[j], act);
			});
		}
	};

	// ================================================================ Conv
	class Conv : public Layer {
	private:
		int kernel;
		int filters;
		int stride;
		int padding;
		CONV_ALGO selected = CONV_ALGO::Auto;
		std::vector<float> scratch;   // CPU: 스레드별 기울기 누적 / FFT 스펙트럼
		Param weight;                  // [filters, Cin*K*K]
		Param bias;                    // [filters]

		int cin() const { return this->inShape.c; }
		int colRows() const { return this->inShape.c * this->kernel * this->kernel; }
		int outPlane() const { return this->outShape.h * this->outShape.w; }
		bool pointwise() const { return this->kernel == 1 && this->stride == 1 && this->padding == 0; }
		// 스레드마다 im2col 버퍼를 따로 잡아도 메모리 한도 안에 들어오는지
		bool laneParallel(size_t extraFloatsPerLane, const Context& ctx) const {
			size_t lane = ((size_t)this->colRows() * this->outPlane() + extraFloatsPerLane) * sizeof(float);
			return lane * threadCount() <= ctx.workspaceLimit;
		}

		void forwardIm2col(const Tensor& x, Tensor& y, Context& ctx) {
			int N = x.shape.n;
			int K = this->colRows();
			int P = this->outPlane();
			size_t colSize = (size_t)K * P;
			bool lanes = this->laneParallel(0, ctx);
			int laneCount = 1;
			if (lanes)
				laneCount = threadCount();
			if (!this->pointwise())
				ctx.workspace.resize(colSize * laneCount, Device::CPU);
			const float* W = this->weight.value.data();
			const float* b = this->bias.value.data();
			auto one = [&](int n, int lane) {
				const float* xn = x.data() + (size_t)n * this->inShape.sample();
				float* yn = y.data() + (size_t)n * this->outShape.sample();
				const float* col = xn;
				if (!this->pointwise()) {
					float* c = ctx.workspace.data() + colSize * lane;
					cpu::im2col(xn, this->cin(), this->inShape.h, this->inShape.w, this->kernel, this->stride, this->padding, this->outShape.h, this->outShape.w, c);
					col = c;
				}
				cpu::gemm(false, false, this->filters, P, K, 1.0f, W, K, col, P, 0.0f, yn, P);
				for (int co = 0; co < this->filters; ++co) {
					float* row = yn + (size_t)co * P;
					for (int p = 0; p < P; ++p)
						row[p] = cpu::activate(row[p] + b[co], this->act);
				}
			};
			if (lanes) {
				parallelFor(N, [&](int n) { one(n, threadIndex()); });
				return;
			}
			for (int n = 0; n < N; ++n)
				one(n, 0);
		}
		void forwardCpu(const Tensor& x, Tensor& y, Context& ctx, CONV_ALGO algo) {
			if (algo == CONV_ALGO::FFT) {
				cpu::convFFT(x.data(), x.shape.n, this->cin(), this->inShape.h, this->inShape.w, this->weight.value.data(), this->bias.value.data(),
					this->filters, this->kernel, this->stride, this->padding, this->outShape.h, this->outShape.w, this->act, y.data(), this->scratch);
				return;
			}
			if (algo == CONV_ALGO::Direct) {
				cpu::convDirect(x.data(), x.shape.n, this->cin(), this->inShape.h, this->inShape.w, this->weight.value.data(), this->bias.value.data(),
					this->filters, this->kernel, this->stride, this->padding, this->outShape.h, this->outShape.w, this->act, y.data());
				return;
			}
			this->forwardIm2col(x, y, ctx);
		}
		// Auto: 실제 배치로 im2col 과 FFT 를 한 번씩 돌려 보고 빠른 쪽을 계속 쓴다.
		CONV_ALGO chooseAlgo(const Tensor& x, Tensor& y, Context& ctx) {
			size_t fftBytes = cpu::fftWorkspaceBytes(this->cin(), this->filters, this->inShape.h, this->inShape.w, this->padding);
			if (fftBytes > ctx.workspaceLimit || this->kernel < 3)
				return CONV_ALGO::Im2col;
			Timer timer;
			this->forwardCpu(x, y, ctx, CONV_ALGO::FFT);
			double fftTime = timer.seconds();
			timer.reset();
			this->forwardCpu(x, y, ctx, CONV_ALGO::Im2col);
			double gemmTime = timer.seconds();
			CONV_ALGO best = CONV_ALGO::Im2col;
			if (fftTime < gemmTime)
				best = CONV_ALGO::FFT;
			std::vector<float>().swap(this->scratch);
			if (ctx.verbose) {
				std::printf("  [Conv %dx%d, %d->%d] im2col %.2fms / FFT %.2fms -> %s\n", this->kernel, this->kernel, this->cin(), this->filters,
					gemmTime * 1000.0, fftTime * 1000.0, toString(best));
			}
			return best;
		}

		void forwardGpu(const Tensor& x, Tensor& y, Context& ctx) {
			int N = x.shape.n;
			int K = this->colRows();
			int P = this->outPlane();
			size_t perSample = ((size_t)K + this->filters) * P;
			int chunk = (int)(std::max)((size_t)1, (std::min)((size_t)N, ctx.workspaceLimit / (perSample * sizeof(float))));
			ctx.workspace.resize(perSample * chunk, Device::CUDA);
			float* col = ctx.workspace.data();
			float* tmp = col + (size_t)K * P * chunk;
			for (int b0 = 0; b0 < N; b0 += chunk) {
				int count = (std::min)(chunk, N - b0);
				int cols = count * P;
				gpu::im2col(x.data() + b0 * this->inShape.sample(), col, this->cin(), this->inShape.h, this->inShape.w, this->kernel, this->stride, this->padding, this->outShape.h, this->outShape.w, count);
				gpu::rt().gemm(false, false, this->filters, cols, K, 1.0f, this->weight.value.data(), K, col, cols, 0.0f, tmp, cols);
				gpu::cnhwToNchw(tmp, this->bias.value.data(), y.data() + b0 * this->outShape.sample(), this->filters, count, P, this->act);
			}
		}
		void backwardGpu(const Tensor& x, Tensor& dy, Tensor& dx, Context& ctx) {
			int N = x.shape.n;
			int K = this->colRows();
			int P = this->outPlane();
			size_t perSample = ((size_t)K + this->filters) * P;
			int chunk = (int)(std::max)((size_t)1, (std::min)((size_t)N, ctx.workspaceLimit / (perSample * sizeof(float))));
			ctx.workspace.resize(perSample * chunk, Device::CUDA);
			float* col = ctx.workspace.data();
			float* tmp = col + (size_t)K * P * chunk;
			this->bias.grad.zero();
			for (int b0 = 0; b0 < N; b0 += chunk) {
				int count = (std::min)(chunk, N - b0);
				int cols = count * P;
				float beta = 1.0f;
				if (b0 == 0)
					beta = 0.0f;
				gpu::nchwToCnhw(dy.data() + b0 * this->outShape.sample(), tmp, this->filters, count, P);
				gpu::im2col(x.data() + b0 * this->inShape.sample(), col, this->cin(), this->inShape.h, this->inShape.w, this->kernel, this->stride, this->padding, this->outShape.h, this->outShape.w, count);
				gpu::rt().gemm(false, true, this->filters, K, cols, 1.0f, tmp, cols, col, cols, beta, this->weight.grad.data(), K);
				gpu::rowSum(tmp, this->bias.grad.data(), this->filters, cols);
				if (!ctx.needInputGrad)
					continue;
				gpu::rt().gemm(true, false, K, cols, this->filters, 1.0f, this->weight.value.data(), K, tmp, cols, 0.0f, col, cols);
				gpu::col2im(col, dx.data() + b0 * this->inShape.sample(), this->cin(), this->inShape.h, this->inShape.w, this->kernel, this->stride, this->padding, this->outShape.h, this->outShape.w, count);
			}
		}
		void backwardCpu(const Tensor& x, Tensor& dy, Tensor& dx, Context& ctx) {
			int N = x.shape.n;
			int K = this->colRows();
			int P = this->outPlane();
			size_t colSize = (size_t)K * P;
			size_t gradSize = (size_t)this->filters * K + this->filters;
			bool lanes = this->laneParallel(gradSize, ctx);
			int laneCount = 1;
			if (lanes)
				laneCount = threadCount();
			ctx.workspace.resize(colSize * laneCount, Device::CPU);
			this->scratch.assign(gradSize * laneCount, 0.0f);
			const float* W = this->weight.value.data();
			auto one = [&](int n, int lane) {
				const float* xn = x.data() + (size_t)n * this->inShape.sample();
				const float* dyn = dy.data() + (size_t)n * this->outShape.sample();
				float* col = ctx.workspace.data() + colSize * lane;
				float* gW = this->scratch.data() + gradSize * lane;
				float* gB = gW + (size_t)this->filters * K;
				const float* src = xn;
				if (!this->pointwise()) {
					cpu::im2col(xn, this->cin(), this->inShape.h, this->inShape.w, this->kernel, this->stride, this->padding, this->outShape.h, this->outShape.w, col);
					src = col;
				}
				cpu::gemm(false, true, this->filters, K, P, 1.0f, dyn, P, src, P, 1.0f, gW, K);
				for (int co = 0; co < this->filters; ++co) {
					const float* row = dyn + (size_t)co * P;
					float sum = 0.0f;
					for (int p = 0; p < P; ++p)
						sum += row[p];
					gB[co] += sum;
				}
				if (!ctx.needInputGrad)
					return;
				float* dxn = dx.data() + (size_t)n * this->inShape.sample();
				if (this->pointwise()) {
					cpu::gemm(true, false, K, P, this->filters, 1.0f, W, K, dyn, P, 0.0f, dxn, P);
					return;
				}
				cpu::gemm(true, false, K, P, this->filters, 1.0f, W, K, dyn, P, 0.0f, col, P);
				std::fill(dxn, dxn + this->inShape.sample(), 0.0f);
				cpu::col2im(col, this->cin(), this->inShape.h, this->inShape.w, this->kernel, this->stride, this->padding, this->outShape.h, this->outShape.w, dxn);
			};
			if (lanes)
				parallelFor(N, [&](int n) { one(n, threadIndex()); });
			else {
				for (int n = 0; n < N; ++n)
					one(n, 0);
			}
			float* gW = this->weight.grad.data();
			float* gB = this->bias.grad.data();
			size_t wSize = (size_t)this->filters * K;
			std::fill(gW, gW + wSize, 0.0f);
			std::fill(gB, gB + this->filters, 0.0f);
			for (int lane = 0; lane < laneCount; ++lane) {
				const float* src = this->scratch.data() + gradSize * lane;
				for (size_t i = 0; i < wSize; ++i)
					gW[i] += src[i];
				for (int co = 0; co < this->filters; ++co)
					gB[co] += src[wSize + co];
			}
		}
	public:
		// filterRowCol: 필터 한 변 크기(3 이면 3x3), filterSize: 필터 개수(출력 채널)
		Conv(int filterRowCol = 3, int filterSize = 8, int stride = 1, int paddingSize = 1, ACTIVATION act = ACTIVATION::ReLU, CONV_ALGO algo = CONV_ALGO::Auto) {
			this->layerType = LAYER::Conv;
			this->kernel = filterRowCol;
			this->filters = filterSize;
			this->stride = (std::max)(1, stride);
			this->padding = (std::max)(0, paddingSize);
			this->act = act;
			this->selected = algo;
		}
		std::string name() const override { return "Conv"; }
		std::string config() const override {
			std::ostringstream os;
			os << "Conv " << this->kernel << " " << this->filters << " " << this->stride << " " << this->padding << " " << toString(this->act);
			return os.str();
		}
		std::string detail() const override {
			std::ostringstream os;
			os << this->kernel << "x" << this->kernel << " s" << this->stride << " p" << this->padding << " " << toString(this->act);
			if (this->selected != CONV_ALGO::Auto)
				os << " [" << toString(this->selected) << "]";
			return os.str();
		}
		Shape build(const Shape& in, std::mt19937& rng) override {
			this->inShape = in;
			int oh = (in.h + 2 * this->padding - this->kernel) / this->stride + 1;
			int ow = (in.w + 2 * this->padding - this->kernel) / this->stride + 1;
			if (oh <= 0 || ow <= 0) {
				throw std::runtime_error("Conv: 입력 " + in.str() + " 이 필터 " + std::to_string(this->kernel) + "x" + std::to_string(this->kernel) + " 보다 작습니다. Pooling 을 줄이거나 padding 을 늘리세요.");
			}
			this->outShape = Shape(1, this->filters, oh, ow);
			int fanIn = in.c * this->kernel * this->kernel;
			int fanOut = this->filters * this->kernel * this->kernel;
			initParam(this->weight, (size_t)this->filters * fanIn, fanIn, fanOut, this->act, rng, false);
			initParam(this->bias, (size_t)this->filters, fanIn, fanOut, this->act, rng, true);
			return this->outShape;
		}
		void forward(const Tensor& x, Tensor& y, Context& ctx) override {
			y.resize(Shape(x.shape.n, this->filters, this->outShape.h, this->outShape.w), ctx.device);
			if (ctx.device == Device::CUDA) {
				this->forwardGpu(x, y, ctx);
				return;
			}
			CONV_ALGO algo = this->selected;
			if (algo == CONV_ALGO::Auto)
				algo = ctx.convAlgo;
			if (algo == CONV_ALGO::Auto) {
				this->selected = this->chooseAlgo(x, y, ctx);
				return;
			}
			this->forwardCpu(x, y, ctx, algo);
		}
		void backward(const Tensor& x, const Tensor& y, Tensor& dy, Tensor& dx, Context& ctx) override {
			if (ctx.needInputGrad)
				dx.resize(x.shape, ctx.device);
			if (ctx.device == Device::CUDA) {
				if (!ctx.fusedOutput)
					gpu::activateBackward(y.data(), dy.data(), (int)y.size(), this->act);
				this->backwardGpu(x, dy, dx, ctx);
				return;
			}
			if (!ctx.fusedOutput)
				cpu::activateBackward(y.data(), dy.data(), y.size(), this->act);
			this->backwardCpu(x, dy, dx, ctx);
		}
		std::vector<Param*> params() override { return { &this->weight, &this->bias }; }
		void setAlgorithm(CONV_ALGO algo) { this->selected = algo; }
		CONV_ALGO algorithm() const { return this->selected; }
	};

	// ================================================================ Pooling
	class Pooling : public Layer {
	private:
		POOLING mode;
		int size;
		int stride;
		Tensor index; // Max/Min 이 고른 위치 (평면 내 인덱스). 마스크 전체를 들고 있지 않는다.
	public:
		Pooling(POOLING poolingName = POOLING::Max, int size = 2, int stride = 0) {
			this->layerType = LAYER::Pool;
			this->mode = poolingName;
			this->size = (std::max)(1, size);
			this->stride = stride;
			if (stride <= 0)
				this->stride = this->size;
		}
		std::string name() const override { return "Pooling"; }
		std::string config() const override {
			std::ostringstream os;
			os << "Pool " << toString(this->mode) << " " << this->size << " " << this->stride;
			return os.str();
		}
		std::string detail() const override {
			std::ostringstream os;
			os << toString(this->mode) << " " << this->size << "x" << this->size;
			return os.str();
		}
		Shape build(const Shape& in, std::mt19937& rng) override {
			this->inShape = in;
			if (in.h < this->size || in.w < this->size)
				throw std::runtime_error("Pooling: 입력 " + in.str() + " 이 풀링 크기보다 작습니다. Pooling 레이어 수를 줄이세요.");
			this->outShape = Shape(1, in.c, (in.h - this->size) / this->stride + 1, (in.w - this->size) / this->stride + 1);
			return this->outShape;
		}
		void forward(const Tensor& x, Tensor& y, Context& ctx) override {
			Shape out(x.shape.n, this->outShape.c, this->outShape.h, this->outShape.w);
			y.resize(out, ctx.device);
			this->index.resize(out, ctx.device);
			int planes = x.shape.n * x.shape.c;
			if (ctx.device == Device::CUDA)
				gpu::poolForward(x.data(), y.data(), this->index.data(), planes, this->inShape.h, this->inShape.w, out.h, out.w, this->size, this->stride, this->mode);
			else
				cpu::poolForward(x.data(), y.data(), this->index.data(), planes, this->inShape.h, this->inShape.w, out.h, out.w, this->size, this->stride, this->mode);
		}
		void backward(const Tensor& x, const Tensor& y, Tensor& dy, Tensor& dx, Context& ctx) override {
			if (!ctx.needInputGrad)
				return;
			dx.resize(x.shape, ctx.device);
			int planes = x.shape.n * x.shape.c;
			if (ctx.device == Device::CUDA) {
				dx.zero();
				gpu::poolBackward(dy.data(), this->index.data(), dx.data(), planes, this->inShape.h, this->inShape.w, y.shape.h, y.shape.w, this->size, this->stride, this->mode);
				return;
			}
			cpu::poolBackward(dy.data(), this->index.data(), dx.data(), planes, this->inShape.h, this->inShape.w, y.shape.h, y.shape.w, this->size, this->stride, this->mode);
		}
	};

	// ================================================================ Padding
	class Padding : public Layer {
	private:
		int pad;
	public:
		Padding(int paddingSize = 1) {
			this->layerType = LAYER::Padding;
			this->pad = (std::max)(0, paddingSize);
		}
		std::string name() const override { return "Padding"; }
		std::string config() const override { return "Padding " + std::to_string(this->pad); }
		std::string detail() const override { return std::to_string(this->pad); }
		Shape build(const Shape& in, std::mt19937& rng) override {
			this->inShape = in;
			this->outShape = Shape(1, in.c, in.h + 2 * this->pad, in.w + 2 * this->pad);
			return this->outShape;
		}
		void forward(const Tensor& x, Tensor& y, Context& ctx) override {
			y.resize(Shape(x.shape.n, this->outShape.c, this->outShape.h, this->outShape.w), ctx.device);
			int planes = x.shape.n * x.shape.c;
			int H = this->inShape.h;
			int W = this->inShape.w;
			if (ctx.device == Device::CUDA) {
				gpu::padForward(x.data(), y.data(), planes, H, W, this->pad);
				return;
			}
			int OW = W + 2 * this->pad;
			y.zero();
			parallelFor(planes, [&](int p) {
				const float* src = x.data() + (size_t)p * H * W;
				float* dst = y.data() + (size_t)p * this->outShape.h * OW;
				for (int iy = 0; iy < H; ++iy)
					std::memcpy(dst + (size_t)(iy + this->pad) * OW + this->pad, src + (size_t)iy * W, W * sizeof(float));
			});
		}
		void backward(const Tensor& x, const Tensor& y, Tensor& dy, Tensor& dx, Context& ctx) override {
			if (!ctx.needInputGrad)
				return;
			dx.resize(x.shape, ctx.device);
			int planes = x.shape.n * x.shape.c;
			int H = this->inShape.h;
			int W = this->inShape.w;
			if (ctx.device == Device::CUDA) {
				gpu::padBackward(dy.data(), dx.data(), planes, H, W, this->pad);
				return;
			}
			int OW = W + 2 * this->pad;
			parallelFor(planes, [&](int p) {
				const float* src = dy.data() + (size_t)p * this->outShape.h * OW;
				float* dst = dx.data() + (size_t)p * H * W;
				for (int iy = 0; iy < H; ++iy)
					std::memcpy(dst + (size_t)iy * W, src + (size_t)(iy + this->pad) * OW + this->pad, W * sizeof(float));
			});
		}
	};

	// ================================================================ Activation
	class Activation : public Layer {
	public:
		Activation(ACTIVATION act = ACTIVATION::ReLU) {
			this->layerType = LAYER::Act;
			this->act = act;
		}
		std::string name() const override { return "Activation"; }
		std::string config() const override { return std::string("Act ") + toString(this->act); }
		std::string detail() const override { return toString(this->act); }
		Shape build(const Shape& in, std::mt19937& rng) override {
			if (this->act == ACTIVATION::Maxout)
				throw std::runtime_error("Maxout 은 FullyConnected(유닛, ACTIVATION::Maxout) 으로 사용하세요.");
			this->inShape = in;
			this->outShape = in;
			return in;
		}
		void forward(const Tensor& x, Tensor& y, Context& ctx) override {
			y.resize(x.shape, ctx.device);
			int rows = x.shape.n;
			int cols = (int)x.shape.sample();
			if (ctx.device == Device::CUDA) {
				gpu::activate(x.data(), y.data(), (int)x.size(), this->act);
				if (this->act == ACTIVATION::Softmax)
					gpu::softmaxRows(y.data(), rows, cols);
				return;
			}
			cpu::activate(x.data(), y.data(), x.size(), this->act);
			if (this->act == ACTIVATION::Softmax)
				cpu::softmaxRows(y.data(), rows, cols);
		}
		void backward(const Tensor& x, const Tensor& y, Tensor& dy, Tensor& dx, Context& ctx) override {
			if (!ctx.needInputGrad)
				return;
			int rows = y.shape.n;
			int cols = (int)y.shape.sample();
			if (!ctx.fusedOutput) {
				if (ctx.device == Device::CUDA) {
					if (this->act == ACTIVATION::Softmax)
						gpu::softmaxBackward(y.data(), dy.data(), rows, cols);
					else
						gpu::activateBackward(y.data(), dy.data(), (int)y.size(), this->act);
				}
				else {
					if (this->act == ACTIVATION::Softmax)
						cpu::softmaxBackward(y.data(), dy.data(), rows, cols);
					else
						cpu::activateBackward(y.data(), dy.data(), y.size(), this->act);
				}
			}
			dx.copyFrom(dy);
		}
	};

	// ================================================================ Flatten
	// FullyConnected 는 입력을 알아서 펼치므로 꼭 필요하지는 않지만, 예전 코드처럼 써도 되도록 남겨둔다.
	class Flatten : public Layer {
	public:
		Flatten() { this->layerType = LAYER::Flatten; }
		std::string name() const override { return "Flatten"; }
		std::string config() const override { return "Flatten"; }
		Shape build(const Shape& in, std::mt19937& rng) override {
			this->inShape = in;
			this->outShape = Shape(1, (int)in.sample(), 1, 1);
			return this->outShape;
		}
		void forward(const Tensor& x, Tensor& y, Context& ctx) override {
			y.copyFrom(x);
			y.shape = Shape(x.shape.n, (int)x.shape.sample(), 1, 1);
		}
		void backward(const Tensor& x, const Tensor& y, Tensor& dy, Tensor& dx, Context& ctx) override {
			if (!ctx.needInputGrad)
				return;
			dx.copyFrom(dy);
			dx.shape = x.shape;
		}
	};

	// ================================================================ FullyConnected
	class FullyConnected : public Layer {
	private:
		int units;
		int pieces = 1;   // Maxout 은 유닛 하나당 선형 출력 2개 중 큰 값을 쓴다.
		Param weight;     // [입력, 유닛*pieces]
		Param bias;       // [유닛*pieces]
		Tensor z;         // Maxout 일 때만 사용하는 선형 출력
		Tensor zGrad;
		Tensor index;

		int inputs() const { return (int)this->inShape.sample(); }
		int rawUnits() const { return this->units * this->pieces; }

		void forwardCpu(const Tensor& x, Tensor& y, float* out, int rows) {
			int in = this->inputs();
			int raw = this->rawUnits();
			cpu::gemm(false, false, rows, raw, in, 1.0f, x.data(), in, this->weight.value.data(), raw, 0.0f, out, raw);
			ACTIVATION linearAct = this->act;
			if (this->act == ACTIVATION::Softmax || this->act == ACTIVATION::Maxout)
				linearAct = ACTIVATION::None;
			addBiasActivate(out, this->bias.value.data(), rows, raw, linearAct);
			if (this->act == ACTIVATION::Softmax)
				cpu::softmaxRows(out, rows, raw);
			if (this->act != ACTIVATION::Maxout)
				return;
			const float* src = out;
			float* dst = y.data();
			float* idx = this->index.data();
			for (int i = 0; i < rows * this->units; ++i) {
				int best = 0;
				for (int p = 1; p < this->pieces; ++p) {
					if (src[(size_t)i * this->pieces + p] > src[(size_t)i * this->pieces + best])
						best = p;
				}
				dst[i] = src[(size_t)i * this->pieces + best];
				idx[i] = (float)best;
			}
		}
		void forwardGpu(const Tensor& x, Tensor& y, float* out, int rows) {
			int in = this->inputs();
			int raw = this->rawUnits();
			gpu::rt().gemm(false, false, rows, raw, in, 1.0f, x.data(), in, this->weight.value.data(), raw, 0.0f, out, raw);
			ACTIVATION linearAct = this->act;
			if (this->act == ACTIVATION::Softmax || this->act == ACTIVATION::Maxout)
				linearAct = ACTIVATION::None;
			gpu::biasActRows(out, this->bias.value.data(), rows, raw, linearAct);
			if (this->act == ACTIVATION::Softmax)
				gpu::softmaxRows(out, rows, raw);
			if (this->act == ACTIVATION::Maxout)
				gpu::maxoutForward(out, y.data(), this->index.data(), rows, this->units, this->pieces);
		}
	public:
		FullyConnected(int denseSize = 10, ACTIVATION act = ACTIVATION::ReLU) {
			this->layerType = LAYER::FC;
			this->units = (std::max)(1, denseSize);
			this->act = act;
			if (act == ACTIVATION::Maxout)
				this->pieces = 2;
		}
		std::string name() const override { return "FullyConnected"; }
		std::string config() const override {
			std::ostringstream os;
			os << "FC " << this->units << " " << toString(this->act);
			return os.str();
		}
		std::string detail() const override { return toString(this->act); }
		int outputUnits() const { return this->units; }
		Shape build(const Shape& in, std::mt19937& rng) override {
			this->inShape = in;
			this->outShape = Shape(1, this->units, 1, 1);
			int fanIn = this->inputs();
			initParam(this->weight, (size_t)fanIn * this->rawUnits(), fanIn, this->rawUnits(), this->act, rng, false);
			initParam(this->bias, (size_t)this->rawUnits(), fanIn, this->rawUnits(), this->act, rng, true);
			return this->outShape;
		}
		void forward(const Tensor& x, Tensor& y, Context& ctx) override {
			int rows = x.shape.n;
			y.resize(Shape(rows, this->units, 1, 1), ctx.device);
			float* out = y.data();
			if (this->act == ACTIVATION::Maxout) {
				this->z.resize(Shape(rows, this->rawUnits(), 1, 1), ctx.device);
				this->index.resize(y.shape, ctx.device);
				out = this->z.data();
			}
			if (ctx.device == Device::CUDA)
				this->forwardGpu(x, y, out, rows);
			else
				this->forwardCpu(x, y, out, rows);
		}
		void backward(const Tensor& x, const Tensor& y, Tensor& dy, Tensor& dx, Context& ctx) override {
			int rows = x.shape.n;
			int in = this->inputs();
			int raw = this->rawUnits();
			bool cuda = ctx.device == Device::CUDA;
			float* g = dy.data();
			if (this->act == ACTIVATION::Maxout) {
				this->zGrad.resize(this->z.shape, ctx.device);
				g = this->zGrad.data();
				if (cuda) {
					gpu::maxoutBackward(dy.data(), this->index.data(), g, rows, this->units, this->pieces);
				}
				else {
					std::fill(g, g + this->zGrad.size(), 0.0f);
					for (int i = 0; i < rows * this->units; ++i)
						g[(size_t)i * this->pieces + (int)this->index.data()[i]] = dy.data()[i];
				}
			}
			else if (!ctx.fusedOutput) {
				if (cuda && this->act == ACTIVATION::Softmax)
					gpu::softmaxBackward(y.data(), g, rows, raw);
				else if (cuda)
					gpu::activateBackward(y.data(), g, (int)y.size(), this->act);
				else if (this->act == ACTIVATION::Softmax)
					cpu::softmaxBackward(y.data(), g, rows, raw);
				else
					cpu::activateBackward(y.data(), g, y.size(), this->act);
			}
			if (ctx.needInputGrad)
				dx.resize(x.shape, ctx.device);
			if (cuda) {
				gpu::rt().gemm(true, false, in, raw, rows, 1.0f, x.data(), in, g, raw, 0.0f, this->weight.grad.data(), raw);
				this->bias.grad.zero();
				gpu::colSum(g, this->bias.grad.data(), rows, raw);
				if (ctx.needInputGrad)
					gpu::rt().gemm(false, true, rows, in, raw, 1.0f, g, raw, this->weight.value.data(), raw, 0.0f, dx.data(), in);
				return;
			}
			cpu::gemm(true, false, in, raw, rows, 1.0f, x.data(), in, g, raw, 0.0f, this->weight.grad.data(), raw);
			float* gB = this->bias.grad.data();
			std::fill(gB, gB + raw, 0.0f);
			for (int r = 0; r < rows; ++r)
				for (int j = 0; j < raw; ++j)
					gB[j] += g[(size_t)r * raw + j];
			if (ctx.needInputGrad)
				cpu::gemm(false, true, rows, in, raw, 1.0f, g, raw, this->weight.value.data(), raw, 0.0f, dx.data(), in);
		}
		std::vector<Param*> params() override { return { &this->weight, &this->bias }; }
	};

	// 저장 파일의 한 줄 설정으로 레이어를 다시 만든다.
	inline ACTIVATION parseActivation(const std::string& text) {
		const ACTIVATION all[] = { ACTIVATION::None, ACTIVATION::Sigmoid, ACTIVATION::ReLU, ACTIVATION::Leaky_ReLU, ACTIVATION::TanH, ACTIVATION::Maxout, ACTIVATION::ELU, ACTIVATION::Softmax };
		for (ACTIVATION a : all) {
			if (text == toString(a))
				return a;
		}
		throw std::runtime_error("알 수 없는 활성화 함수: " + text);
	}
	inline Layer* createLayer(const std::string& line) {
		std::istringstream in(line);
		std::string type;
		in >> type;
		if (type == "Conv") {
			int k = 0;
			int f = 0;
			int s = 0;
			int p = 0;
			std::string act;
			in >> k >> f >> s >> p >> act;
			return new Conv(k, f, s, p, parseActivation(act));
		}
		if (type == "Pool") {
			std::string mode;
			int size = 2;
			int stride = 2;
			in >> mode >> size >> stride;
			POOLING pool = POOLING::Max;
			if (mode == "Average")
				pool = POOLING::Average;
			if (mode == "Min")
				pool = POOLING::Min;
			return new Pooling(pool, size, stride);
		}
		if (type == "Padding") {
			int p = 0;
			in >> p;
			return new Padding(p);
		}
		if (type == "Act") {
			std::string act;
			in >> act;
			return new Activation(parseActivation(act));
		}
		if (type == "Flatten")
			return new Flatten();
		if (type == "FC") {
			int units = 0;
			std::string act;
			in >> units >> act;
			return new FullyConnected(units, parseActivation(act));
		}
		throw std::runtime_error("알 수 없는 레이어: " + line);
	}
}

#endif
