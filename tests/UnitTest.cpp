// SimpleCNN 단위 테스트
//  1. GEMM 전치 조합이 단순 3중 루프와 같은지
//  2. 합성곱 im2col / FFT / Direct 결과가 같은지
//  3. 모든 레이어의 역전파가 수치 미분과 같은지 (gradient check)
//  4. 저장 → 불러오기 후 예측이 같은지
//  5. (GPU 가 있으면) CPU 와 CUDA 결과가 같은지
#include "SimpleCNN.h"

using namespace cnn;

static int failures = 0;
static int checks = 0;

static void expect(bool ok, const std::string& what) {
	++checks;
	if (ok)
		return;
	++failures;
	std::printf("  [실패] %s\n", what.c_str());
}

static std::vector<float> randomVector(size_t n, std::mt19937& rng, float scale = 1.0f) {
	std::uniform_real_distribution<float> dist(-scale, scale);
	std::vector<float> v(n);
	for (float& x : v)
		x = dist(rng);
	return v;
}

static float maxRelativeError(const std::vector<float>& a, const std::vector<float>& b) {
	float worst = 0.0f;
	for (size_t i = 0; i < a.size(); ++i) {
		float diff = std::fabs(a[i] - b[i]);
		float scale = (std::max)(1.0f, (std::max)(std::fabs(a[i]), std::fabs(b[i])));
		worst = (std::max)(worst, diff / scale);
	}
	return worst;
}

static void testGemm() {
	std::printf("GEMM\n");
	std::mt19937 rng(1);
	int M = 37;
	int N = 300;
	int K = 45;
	for (int ta = 0; ta < 2; ++ta) {
		for (int tb = 0; tb < 2; ++tb) {
			std::vector<float> A = randomVector((size_t)M * K, rng);
			std::vector<float> B = randomVector((size_t)K * N, rng);
			std::vector<float> C = randomVector((size_t)M * N, rng);
			std::vector<float> expected = C;
			int lda = K;
			if (ta)
				lda = M;
			int ldb = N;
			if (tb)
				ldb = K;
			for (int i = 0; i < M; ++i) {
				for (int j = 0; j < N; ++j) {
					double sum = 0.0;
					for (int k = 0; k < K; ++k) {
						float a = A[(size_t)i * lda + k];
						if (ta)
							a = A[(size_t)k * lda + i];
						float b = B[(size_t)k * ldb + j];
						if (tb)
							b = B[(size_t)j * ldb + k];
						sum += (double)a * b;
					}
					expected[(size_t)i * N + j] = (float)(0.5 * sum + 0.25 * C[(size_t)i * N + j]);
				}
			}
			cpu::gemm(ta != 0, tb != 0, M, N, K, 0.5f, A.data(), lda, B.data(), ldb, 0.25f, C.data(), N);
			float err = maxRelativeError(C, expected);
			expect(err < 1e-4f, "gemm transA=" + std::to_string(ta) + " transB=" + std::to_string(tb) + " 오차 " + std::to_string(err));
		}
	}
}

static std::vector<float> runConv(Conv& conv, CONV_ALGO algo, const Tensor& x, Context& ctx) {
	conv.setAlgorithm(algo);
	Tensor y;
	conv.forward(x, y, ctx);
	return y.toVector();
}

static void testConvAlgorithms() {
	std::printf("합성곱 알고리즘 (im2col / FFT / Direct)\n");
	struct Case { int c, h, w, k, f, s, p; };
	Case cases[] = { { 3, 17, 13, 3, 5, 1, 1 }, { 2, 20, 20, 5, 4, 2, 2 }, { 4, 9, 9, 1, 6, 1, 0 }, { 1, 28, 28, 7, 3, 1, 3 } };
	for (const Case& c : cases) {
		std::mt19937 rng(7);
		Conv conv(c.k, c.f, c.s, c.p, ACTIVATION::TanH);
		conv.build(Shape(1, c.c, c.h, c.w), rng);
		Tensor x(Shape(3, c.c, c.h, c.w), Device::CPU);
		std::vector<float> data = randomVector(x.size(), rng);
		x.upload(data.data(), data.size());
		Context ctx;
		ctx.verbose = false;
		std::vector<float> direct = runConv(conv, CONV_ALGO::Direct, x, ctx);
		std::vector<float> im2col = runConv(conv, CONV_ALGO::Im2col, x, ctx);
		std::vector<float> fft = runConv(conv, CONV_ALGO::FFT, x, ctx);
		std::string name = "k" + std::to_string(c.k) + " s" + std::to_string(c.s) + " p" + std::to_string(c.p);
		expect(maxRelativeError(direct, im2col) < 1e-4f, "im2col != direct (" + name + ")");
		expect(maxRelativeError(direct, fft) < 1e-3f, "FFT != direct (" + name + ") 오차 " + std::to_string(maxRelativeError(direct, fft)));
	}
}

// 손실 L = sum(y * r) 로 두고 dL/dx, dL/dw 를 수치 미분과 비교한다.
// Max/Min 풀링은 값이 가까우면 eps 만큼 흔들 때 선택이 바뀌므로 eps 를 작게 준다.
static void gradCheck(const std::string& name, Layer* layer, Shape in, int batch, float eps = 1e-2f) {
	std::unique_ptr<Layer> owner(layer);
	std::mt19937 rng(3);
	layer->build(in, rng);
	Context ctx;
	ctx.verbose = false;
	ctx.convAlgo = CONV_ALGO::Im2col;
	Tensor x(Shape(batch, in.c, in.h, in.w), Device::CPU);
	std::vector<float> xs = randomVector(x.size(), rng);
	x.upload(xs.data(), xs.size());
	Tensor y;
	layer->forward(x, y, ctx);
	std::vector<float> r = randomVector(y.size(), rng);
	Tensor dy(y.shape, Device::CPU);
	dy.upload(r.data(), r.size());
	Tensor dx;
	ctx.needInputGrad = true;
	layer->backward(x, y, dy, dx, ctx);

	auto loss = [&]() {
		Tensor out;
		layer->forward(x, out, ctx);
		std::vector<float> v = out.toVector();
		double sum = 0.0;
		for (size_t i = 0; i < v.size(); ++i)
			sum += (double)v[i] * r[i];
		return sum;
	};
	float worst = 0.0f;
	std::vector<float> analytic = dx.toVector();
	int stride = (std::max)(1, (int)x.size() / 40);
	for (size_t i = 0; i < x.size(); i += stride) {
		float keep = x.data()[i];
		x.data()[i] = keep + eps;
		double plus = loss();
		x.data()[i] = keep - eps;
		double minus = loss();
		x.data()[i] = keep;
		float numeric = (float)((plus - minus) / (2.0 * eps));
		worst = (std::max)(worst, std::fabs(numeric - analytic[i]) / (std::max)(1.0f, std::fabs(numeric)));
	}
	for (Param* p : layer->params()) {
		std::vector<float> grad = p->grad.toVector();
		int pstride = (std::max)(1, (int)p->size() / 40);
		for (size_t i = 0; i < p->size(); i += pstride) {
			float keep = p->value.data()[i];
			p->value.data()[i] = keep + eps;
			double plus = loss();
			p->value.data()[i] = keep - eps;
			double minus = loss();
			p->value.data()[i] = keep;
			float numeric = (float)((plus - minus) / (2.0 * eps));
			worst = (std::max)(worst, std::fabs(numeric - grad[i]) / (std::max)(1.0f, std::fabs(numeric)));
		}
	}
	expect(worst < 2e-2f, name + " 기울기 오차 " + std::to_string(worst));
	std::printf("  %-28s 최대 오차 %.5f\n", name.c_str(), worst);
}

static void testGradients() {
	std::printf("역전파 수치 검증\n");
	gradCheck("Conv 3x3 TanH", new Conv(3, 4, 1, 1, ACTIVATION::TanH), Shape(1, 2, 7, 7), 2);
	gradCheck("Conv 5x5 stride2 Sigmoid", new Conv(5, 3, 2, 2, ACTIVATION::Sigmoid), Shape(1, 3, 9, 9), 2);
	gradCheck("Conv 1x1 None", new Conv(1, 5, 1, 0, ACTIVATION::None), Shape(1, 3, 4, 4), 2);
	gradCheck("Conv 3x3 ELU", new Conv(3, 2, 1, 0, ACTIVATION::ELU), Shape(1, 2, 6, 6), 2);
	gradCheck("Pooling Max", new Pooling(POOLING::Max), Shape(1, 2, 6, 6), 2, 1e-4f);
	gradCheck("Pooling Min", new Pooling(POOLING::Min), Shape(1, 2, 6, 6), 2, 1e-4f);
	gradCheck("Pooling Average 3/2", new Pooling(POOLING::Average, 3, 2), Shape(1, 2, 7, 7), 2);
	gradCheck("Padding", new Padding(2), Shape(1, 2, 4, 5), 2);
	gradCheck("Flatten", new Flatten(), Shape(1, 2, 3, 3), 2);
	gradCheck("Activation TanH", new Activation(ACTIVATION::TanH), Shape(1, 2, 3, 3), 2);
	gradCheck("Activation Softmax", new Activation(ACTIVATION::Softmax), Shape(1, 5, 1, 1), 3);
	gradCheck("FC Sigmoid", new FullyConnected(6, ACTIVATION::Sigmoid), Shape(1, 2, 3, 3), 3);
	gradCheck("FC Softmax", new FullyConnected(5, ACTIVATION::Softmax), Shape(1, 7, 1, 1), 3);
	gradCheck("FC Maxout", new FullyConnected(4, ACTIVATION::Maxout), Shape(1, 6, 1, 1), 3);
	gradCheck("FC Leaky_ReLU", new FullyConnected(4, ACTIVATION::Leaky_ReLU), Shape(1, 6, 1, 1), 3);
}

static Dataset stripes(int count, unsigned seed) {
	std::mt19937 rng(seed);
	std::uniform_real_distribution<float> noise(0.0f, 0.3f);
	V4D images(count, V3D(1, V2D(10, V1D(10, 0.0f))));
	V1D labels(count);
	for (int i = 0; i < count; ++i) {
		int label = i % 3;
		labels[i] = (float)label;
		for (int y = 0; y < 10; ++y) {
			for (int x = 0; x < 10; ++x) {
				float v = noise(rng);
				if (label == 0 && x % 3 == 0)
					v += 0.7f;
				if (label == 1 && y % 3 == 0)
					v += 0.7f;
				if (label == 2 && (x + y) % 3 == 0)
					v += 0.7f;
				images[i][0][y][x] = v;
			}
		}
	}
	return Dataset::fromVectors(images, labels);
}

static void buildSmall(CNN& model) {
	model.add(new Conv(3, 6, 1, 1, ACTIVATION::ReLU));
	model.add(new Pooling(POOLING::Max));
	model.add(new Conv(3, 8, 1, 1, ACTIVATION::Leaky_ReLU));
	model.add(new FullyConnected(16, ACTIVATION::TanH));
	model.add(new FullyConnected(3, ACTIVATION::Softmax));
}

static void testTrainSaveLoad() {
	std::printf("학습 / 저장 / 불러오기\n");
	Dataset data = stripes(300, 5);
	CNN model;
	model.setVerbose(false);
	buildSmall(model);
	model.compile(OPTIMIZER::Adam, LOSS::CategoricalCrossentropy);
	History h = model.fit(data, 8, 20);
	float acc = model.evaluate(data);
	expect(h.loss.back() < h.loss.front(), "학습 손실이 줄지 않음");
	expect(acc > 0.9f, "학습 정확도 낮음 " + std::to_string(acc));
	std::printf("  정확도 %.2f%%, 손실 %.4f -> %.4f\n", acc * 100.0f, h.loss.front(), h.loss.back());

	model.save("unit_test_model.scnn");
	CNN loaded;
	loaded.setVerbose(false);
	loaded.load("unit_test_model.scnn");
	std::vector<int> a = model.predict(data);
	std::vector<int> b = loaded.predict(data);
	expect(a == b, "불러온 모델의 예측이 다름");
	std::remove("unit_test_model.scnn");

	const OPTIMIZER optimizers[] = { OPTIMIZER::Mini_SGD, OPTIMIZER::Momentum, OPTIMIZER::Adagrad, OPTIMIZER::RMSProp, OPTIMIZER::Nadam };
	for (OPTIMIZER opt : optimizers) {
		CNN m;
		m.setVerbose(false);
		buildSmall(m);
		m.compile(opt, LOSS::SparseCategoricalCrossentropy);
		History hh = m.fit(data, 4, 20);
		expect(hh.loss.back() < hh.loss.front(), "옵티마이저 " + std::to_string((int)opt) + " 손실이 줄지 않음");
	}
}

// GPU 가 있으면 같은 시드로 CPU 와 CUDA 를 돌려 결과를 비교한다.
static void testCudaMatchesCpu() {
	std::printf("CPU vs CUDA\n");
	if (!cuda::Runtime::instance().available()) {
		std::printf("  (건너뜀) %s\n", cuda::Runtime::instance().error().c_str());
		return;
	}
	std::printf("  %s\n", cuda::Runtime::instance().info().c_str());
	Dataset data = stripes(120, 9);
	CNN cpuModel(Device::CPU);
	CNN gpuModel(Device::CUDA);
	CNN* models[] = { &cpuModel, &gpuModel };
	std::vector<History> histories;
	for (CNN* m : models) {
		m->setVerbose(false);
		m->setConvAlgorithm(CONV_ALGO::Im2col);
		m->add(new Conv(3, 4, 1, 1, ACTIVATION::ReLU));
		m->add(new Pooling(POOLING::Average));
		m->add(new Padding(1));
		m->add(new Conv(3, 4, 2, 0, ACTIVATION::Sigmoid));
		m->add(new FullyConnected(12, ACTIVATION::Maxout));
		m->add(new FullyConnected(3, ACTIVATION::Softmax));
		m->compile(OPTIMIZER::Adam, LOSS::CategoricalCrossentropy);
		histories.push_back(m->fit(data, 3, 16));
	}
	for (size_t i = 0; i < histories[0].loss.size(); ++i) {
		float diff = std::fabs(histories[0].loss[i] - histories[1].loss[i]);
		expect(diff < 1e-3f, "epoch " + std::to_string(i + 1) + " 손실 차이 " + std::to_string(diff));
	}
	std::printf("  손실 CPU %.5f / CUDA %.5f\n", histories[0].loss.back(), histories[1].loss.back());
}

int main() {
	testGemm();
	testConvAlgorithms();
	testGradients();
	testTrainSaveLoad();
	testCudaMatchesCpu();
	std::printf("\n%d 개 검사 중 %d 개 실패\n", checks, failures);
	if (failures > 0)
		return 1;
	return 0;
}
