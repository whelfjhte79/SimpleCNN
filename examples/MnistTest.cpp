// SimpleCNN 실전 테스트: MNIST 손글씨 숫자 (학습 6만 장, 테스트 1만 장)
//
// 실행: MnistTest [cuda|cpu] [--epochs 3] [--train 60000] [--batch 64] [--algo auto|im2col|fft] [--image 내그림.png]
//   MnistTest cuda                      GPU 로 학습
//   MnistTest cpu --epochs 1 --train 10000
//   MnistTest cuda --image digit.png    학습 후 내가 그린 숫자 예측
//
// 하는 일
//   1. MNIST 를 내려받는다 (data/mnist, 처음 한 번만)
//   2. 레이어를 쌓고 학습한다
//   3. 테스트셋 정확도를 잰다
//   4. 모델을 저장하고 다시 불러온다
//   5. 테스트 이미지를 BMP 파일로 저장한 뒤, 그 파일을 읽어서 예측한다
#include "SimpleCNN.h"
#include "Download.h"

using namespace cnn;

struct Options {
	Device device = Device::CPU;
	int epochs = 3;
	int batch = 64;
	size_t trainLimit = 0;
	CONV_ALGO algo = CONV_ALGO::Auto;
	std::string image;
	std::string dataDir = "data";
};

static std::string lower(std::string text) {
	std::transform(text.begin(), text.end(), text.begin(), [](unsigned char ch) { return (char)std::tolower(ch); });
	return text;
}

static Options parse(int argc, char** argv) {
	Options opt;
	for (int i = 1; i < argc; ++i) {
		std::string arg = lower(argv[i]);
		std::string next;
		if (i + 1 < argc)
			next = argv[i + 1];
		if (arg == "cuda" || arg == "gpu" || arg == "--cuda") {
			opt.device = Device::CUDA;
		}
		else if (arg == "cpu" || arg == "--cpu") {
			opt.device = Device::CPU;
		}
		else if (arg == "--epochs" && !next.empty()) {
			opt.epochs = std::atoi(next.c_str());
			++i;
		}
		else if (arg == "--batch" && !next.empty()) {
			opt.batch = std::atoi(next.c_str());
			++i;
		}
		else if (arg == "--train" && !next.empty()) {
			opt.trainLimit = (size_t)std::atol(next.c_str());
			++i;
		}
		else if (arg == "--image" && !next.empty()) {
			opt.image = next;
			++i;
		}
		else if (arg == "--data" && !next.empty()) {
			opt.dataDir = next;
			++i;
		}
		else if (arg == "--algo" && !next.empty()) {
			std::string algo = lower(next);
			if (algo == "im2col")
				opt.algo = CONV_ALGO::Im2col;
			else if (algo == "fft")
				opt.algo = CONV_ALGO::FFT;
			else if (algo == "direct")
				opt.algo = CONV_ALGO::Direct;
			++i;
		}
	}
	return opt;
}

// MNIST 는 검은 배경에 흰 글씨다. 흰 종이에 검은 글씨로 그린 그림이면 뒤집어 준다.
static Image toMnistStyle(const Image& source) {
	Image gray = source.withChannels(1);
	double sum = 0.0;
	for (uint8_t v : gray.pixels)
		sum += v;
	if (sum / gray.pixels.size() > 127.0) {
		for (uint8_t& v : gray.pixels)
			v = (uint8_t)(255 - v);
	}
	return gray;
}

static int run(const Options& opt) {
	std::filesystem::path dataDir = toPath(opt.dataDir);
	std::filesystem::path mnistDir = dataDir / "mnist";
	if (!download::prepareMnist(mnistDir)) {
		std::printf("MNIST 를 내려받지 못했습니다. 인터넷 연결을 확인하거나 %s 폴더에 파일을 직접 넣어주세요.\n", pathString(mnistDir).c_str());
		return 1;
	}
	Dataset train = Dataset::loadMNIST(pathString(mnistDir / "train-images-idx3-ubyte"), pathString(mnistDir / "train-labels-idx1-ubyte"), opt.trainLimit);
	Dataset test = Dataset::loadMNIST(pathString(mnistDir / "t10k-images-idx3-ubyte"), pathString(mnistDir / "t10k-labels-idx1-ubyte"));
	std::printf("MNIST: 학습 %zu 장, 테스트 %zu 장 (데이터 메모리 %s)\n", train.size(), test.size(), formatBytes((double)(train.memoryBytes() + test.memoryBytes())).c_str());

	// 1. 블럭 쌓기 - 입력 크기는 데이터에서 자동으로 정해진다.
	CNN model(opt.device);
	model.add(new Conv(3, 16, 1, 1, ACTIVATION::ReLU));
	model.add(new Pooling(POOLING::Max));
	model.add(new Conv(3, 32, 1, 1, ACTIVATION::ReLU));
	model.add(new Pooling(POOLING::Max));
	model.add(new FullyConnected(128, ACTIVATION::ReLU));
	model.add(new FullyConnected(10, ACTIVATION::Softmax));
	model.setConvAlgorithm(opt.algo);
	model.compile(OPTIMIZER::Adam, LOSS::CategoricalCrossentropy);
	model.summary(train.shape(), opt.batch);

	// 2. 학습 (검증용으로 테스트셋 일부를 epoch 마다 확인)
	Dataset validation = test.subset(0, 2000);
	Timer timer;
	model.fit(train, opt.epochs, opt.batch, &validation);
	std::printf("학습 시간: %.1f 초\n", timer.seconds());

	// 3. 평가
	float accuracy = model.evaluate(test);

	// 4. 저장 → 새 모델로 불러오기 (레이어를 다시 쌓지 않아도 된다)
	std::string modelPath = pathString(dataDir / "mnist_model.scnn");
	model.save(modelPath);
	CNN loaded(opt.device);
	loaded.load(modelPath);
	loaded.evaluate(test);

	// 5. 이미지 파일로 저장했다가 다시 읽어서 예측
	std::filesystem::path sampleDir = dataDir / "samples";
	std::error_code ec;
	std::filesystem::create_directories(sampleDir, ec);
	std::vector<float> pixels(test.shape().sample());
	int correct = 0;
	std::printf("\n이미지 파일 예측 (%s)\n", pathString(sampleDir).c_str());
	for (int i = 0; i < 10; ++i) {
		test.sample(i, pixels.data());
		int label = (int)test.label(i);
		std::string file = pathString(sampleDir / ("digit_" + std::to_string(i) + "_label" + std::to_string(label) + ".bmp"));
		Image::fromCHW(pixels.data(), 1, 28, 28).saveBMP(file);
		int predicted = loaded.predict(file);
		std::string mark = "X";
		if (predicted == label) {
			mark = "O";
			++correct;
		}
		std::printf("  %-40s 정답 %d  예측 %d  %s\n", file.c_str(), label, predicted, mark.c_str());
	}
	std::printf("  파일 10장 중 %d 장 맞춤\n", correct);

	// 6. 내가 준 이미지
	if (!opt.image.empty()) {
		Image mine = Image::load(opt.image);
		if (mine.empty()) {
			std::printf("이미지를 읽을 수 없습니다: %s\n", opt.image.c_str());
		}
		else {
			std::vector<float> prob = loaded.predictProba(toMnistStyle(mine));
			int best = (int)(std::max_element(prob.begin(), prob.end()) - prob.begin());
			std::printf("\n%s → %d (확률 %.1f%%)\n", opt.image.c_str(), best, prob[best] * 100.0f);
		}
	}

	if (accuracy < 0.9f) {
		std::printf("\n정확도가 기대보다 낮습니다.\n");
		return 1;
	}
	std::printf("\n테스트 통과: 정확도 %.2f%%\n", accuracy * 100.0f);
	return 0;
}

int main(int argc, char** argv) {
	Options opt = parse(argc, argv);
	try {
		return run(opt);
	}
	catch (const std::exception& e) {
		std::printf("\n오류: %s\n", e.what());
		return 1;
	}
}
