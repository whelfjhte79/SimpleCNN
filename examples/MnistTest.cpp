// SimpleCNN 실전 테스트: MNIST 손글씨 숫자 (학습 6만 장, 테스트 1만 장) 또는 내 이미지 폴더
//
// 실행: MnistTest [cuda|cpu] [--epochs 3] [--train 60000] [--batch 64] [--algo auto|im2col|fft] [--image 내그림.png]
//       MnistTest [cuda|cpu] --folder 이미지폴더 [--size 32] [--channels 3] [--epochs 10] [--image 사진.png]
//   MnistTest cuda                      GPU 로 학습
//   MnistTest cpu --epochs 1 --train 10000
//   MnistTest cuda --image digit.png    학습 후 내가 그린 숫자 예측
//   MnistTest cuda --folder photos      photos/고양이/*.png, photos/강아지/*.bmp 처럼 클래스별 폴더의 이미지로 학습
//
// 하는 일 (MNIST)
//   1. MNIST 를 내려받는다 (data/mnist, 처음 한 번만)
//   2. 레이어를 쌓고 학습한다
//   3. 테스트셋 정확도를 잰다
//   4. 모델을 저장하고 다시 불러온다
//   5. 테스트 이미지를 BMP 파일로 저장한 뒤, 그 파일을 읽어서 예측한다
//
// 하는 일 (--folder)
//   PNG / BMP / PGM / PPM / RAW 이미지를 --size 크기로 맞춰 읽고, 80% 로 학습해서 나머지 20% 로 평가한다.
//   RAW 는 헤더가 없으므로 파일 이름에 크기를 넣는다. (예: cat_64x48x3.raw)
#include "SimpleCNN.h"

// ------------------------------------------------------------------------ 내려받기 도우미
// curl 로 받고 (Windows 10 이상 기본 포함), gzip 은 SimpleCNN.h 의 inflate 로 zlib 없이 푼다.
namespace download {

	// .gz 한 파일을 풀어 outPath 로 저장
	inline void gunzip(const std::filesystem::path& gzPath, const std::filesystem::path& outPath) {
		std::ifstream file(gzPath, std::ios::binary);
		std::vector<uint8_t> data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
		if (data.size() < 18 || data[0] != 0x1f || data[1] != 0x8b || data[2] != 8)
			throw std::runtime_error("gzip 파일이 아닙니다: " + gzPath.string());
		int flags = data[3];
		size_t pos = 10;
		if (flags & 4)
			pos += 2 + (data[pos] | (data[pos + 1] << 8));
		if (flags & 8) {
			while (data[pos] != 0)
				++pos;
			++pos;
		}
		if (flags & 16) {
			while (data[pos] != 0)
				++pos;
			++pos;
		}
		if (flags & 2)
			pos += 2;
		std::vector<uint8_t> output = cnn::inflate(data.data(), data.size(), pos);
		std::ofstream result(outPath, std::ios::binary);
		result.write(reinterpret_cast<const char*>(output.data()), (std::streamsize)output.size());
	}

	inline bool downloadFile(const std::string& url, const std::filesystem::path& path) {
		std::string command = "curl -L --fail --silent --show-error -o \"" + path.string() + "\" \"" + url + "\"";
		int result = std::system(command.c_str());
		std::error_code ec;
		return result == 0 && std::filesystem::exists(path, ec) && std::filesystem::file_size(path, ec) > 0;
	}

	// MNIST 4개 파일을 dir 에 준비한다. 이미 있으면 건너뛴다.
	inline bool prepareMnist(const std::filesystem::path& dir) {
		const char* names[] = { "train-images-idx3-ubyte", "train-labels-idx1-ubyte", "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte" };
		const char* mirrors[] = { "https://storage.googleapis.com/cvdf-datasets/mnist/", "https://ossci-datasets.s3.amazonaws.com/mnist/" };
		std::error_code ec;
		std::filesystem::create_directories(dir, ec);
		for (const char* name : names) {
			std::filesystem::path target = dir / name;
			if (std::filesystem::exists(target, ec))
				continue;
			std::filesystem::path gz = dir / (std::string(name) + ".gz");
			bool ok = false;
			for (const char* mirror : mirrors) {
				std::printf("내려받는 중: %s%s.gz\n", mirror, name);
				if (downloadFile(std::string(mirror) + name + ".gz", gz)) {
					ok = true;
					break;
				}
			}
			if (!ok)
				return false;
			gunzip(gz, target);
			std::filesystem::remove(gz, ec);
		}
		return true;
	}
}

using namespace cnn;

struct Options {
	Device device = Device::CPU;
	int epochs = 3;
	int batch = 64;
	size_t trainLimit = 0;
	CONV_ALGO algo = CONV_ALGO::Auto;
	std::string image;
	std::string dataDir = "data";
	std::string folder;
	int size = 32;
	int channels = 3;
	bool epochsSet = false;
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
			opt.epochsSet = true;
			++i;
		}
		else if (arg == "--folder" && !next.empty()) {
			opt.folder = next;
			++i;
		}
		else if (arg == "--size" && !next.empty()) {
			opt.size = std::atoi(next.c_str());
			++i;
		}
		else if (arg == "--channels" && !next.empty()) {
			opt.channels = std::atoi(next.c_str());
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

// MNIST 와 --folder 가 같이 쓰는 블럭 구성. 입력 크기는 데이터에서 자동으로 정해진다.
static void addLayers(CNN& model, int classes) {
	model.add(new Conv(3, 16, 1, 1, ACTIVATION::ReLU));
	model.add(new Pooling(POOLING::Max));
	model.add(new Conv(3, 32, 1, 1, ACTIVATION::ReLU));
	model.add(new Pooling(POOLING::Max));
	model.add(new FullyConnected(128, ACTIVATION::ReLU));
	model.add(new FullyConnected(classes, ACTIVATION::Softmax));
}

// --image 로 준 파일 한 장 예측
static void predictImage(CNN& model, const std::string& path, const std::vector<std::string>& classNames, bool mnist) {
	Image image = Image::load(toPath(path));
	if (image.empty()) {
		std::printf("이미지를 읽을 수 없습니다: %s\n", path.c_str());
		return;
	}
	if (mnist)
		image = toMnistStyle(image);
	std::vector<float> prob = model.predictProba(image);
	int best = (int)(std::max_element(prob.begin(), prob.end()) - prob.begin());
	std::string name = std::to_string(best);
	if (best < (int)classNames.size())
		name = classNames[best];
	std::printf("\n%s → %s (확률 %.1f%%)\n", path.c_str(), name.c_str(), prob[best] * 100.0f);
}

static int runMnist(const Options& opt, CNN& model) {
	std::filesystem::path dataDir = toPath(opt.dataDir);
	std::filesystem::path mnistDir = dataDir / "mnist";
	if (!download::prepareMnist(mnistDir)) {
		std::printf("MNIST 를 내려받지 못했습니다. 인터넷 연결을 확인하거나 %s 폴더에 파일을 직접 넣어주세요.\n", pathString(mnistDir).c_str());
		return 1;
	}
	Dataset train = Dataset::loadMNIST(pathString(mnistDir / "train-images-idx3-ubyte"), pathString(mnistDir / "train-labels-idx1-ubyte"), opt.trainLimit);
	Dataset test = Dataset::loadMNIST(pathString(mnistDir / "t10k-images-idx3-ubyte"), pathString(mnistDir / "t10k-labels-idx1-ubyte"));
	std::printf("MNIST: 학습 %zu 장, 테스트 %zu 장 (데이터 메모리 %s)\n", train.size(), test.size(), formatBytes((double)(train.memoryBytes() + test.memoryBytes())).c_str());

	// 1. 블럭 쌓기
	addLayers(model, 10);
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
	if (!opt.image.empty())
		predictImage(loaded, opt.image, test.classNames, true);

	if (accuracy < 0.9f) {
		std::printf("\n정확도가 기대보다 낮습니다.\n");
		return 1;
	}
	std::printf("\n테스트 통과: 정확도 %.2f%%\n", accuracy * 100.0f);
	return 0;
}

// 클래스별 폴더의 이미지 (PNG / BMP / RAW ...) 로 학습 → 평가 → 저장 → 예측
static int runFolder(const Options& opt, CNN& model) {
	if (opt.size < 4 || opt.channels < 1 || opt.channels > 4) {
		std::printf("--size 는 4 이상, --channels 는 1~4 로 주세요.\n");
		return 1;
	}
	Dataset all = Dataset::loadFolder(opt.folder, opt.size, opt.size, opt.channels);
	int classes = (int)all.classNames.size();
	std::printf("%s: %zu 장, 클래스 %d 개 (", opt.folder.c_str(), all.size(), classes);
	for (int i = 0; i < classes; ++i) {
		if (i > 0)
			std::printf(", ");
		std::printf("%s", all.classNames[i].c_str());
	}
	std::printf("), 입력 %dx%dx%d\n", opt.size, opt.size, opt.channels);
	if (classes < 2) {
		std::printf("클래스 폴더가 2 개 이상 있어야 합니다.\n");
		return 1;
	}

	// 80% 학습, 20% 평가. 평가용이 한 장도 없을 만큼 적으면 학습 데이터로 평가한다.
	std::pair<Dataset, Dataset> parts = all.split(0.2f);
	const Dataset& train = parts.first;
	const Dataset* test = &parts.second;
	if (test->empty())
		test = &train;
	addLayers(model, classes);
	model.summary(train.shape(), opt.batch);

	int epochs = opt.epochs;
	if (!opt.epochsSet)
		epochs = 10;
	Timer timer;
	model.fit(train, epochs, opt.batch, test);
	std::printf("학습 시간: %.1f 초\n", timer.seconds());
	float accuracy = model.evaluate(*test);

	std::filesystem::path dataDir = toPath(opt.dataDir);
	std::error_code ec;
	std::filesystem::create_directories(dataDir, ec);
	model.save(pathString(dataDir / "folder_model.scnn"));

	if (!opt.image.empty())
		predictImage(model, opt.image, all.classNames, false);
	std::printf("\n학습 완료: 테스트 정확도 %.2f%%\n", accuracy * 100.0f);
	return 0;
}

int main(int argc, char** argv) {
	Options opt = parse(argc, argv);
	try {
		// 장치부터 정한다. CUDA 면 GPU 정보가 출력 첫 줄에 나온다. 레이어는 데이터를 읽은 뒤 쌓는다.
		CNN model(opt.device);
		model.setConvAlgorithm(opt.algo);
		model.compile(OPTIMIZER::Adam, LOSS::CategoricalCrossentropy);
		if (!opt.folder.empty())
			return runFolder(opt, model);
		return runMnist(opt, model);
	}
	catch (const std::exception& e) {
		std::printf("\n오류: %s\n", e.what());
		return 1;
	}
}
