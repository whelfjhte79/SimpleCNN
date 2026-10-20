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

// ------------------------------------------------------------------------ 내려받기 도우미
// curl 로 받고 (Windows 10 이상 기본 포함), gzip 은 zlib 없이 풀도록 inflate(RFC 1951)를 직접 구현했다.
namespace download {

	class Inflater {
	private:
		struct Huffman {
			std::vector<short> count;
			std::vector<short> symbol;
		};
		const uint8_t* in;
		size_t inSize;
		size_t pos;
		uint32_t bitBuffer = 0;
		int bitCount = 0;
		std::vector<uint8_t>& out;

		int bits(int need) {
			uint32_t value = this->bitBuffer;
			while (this->bitCount < need) {
				if (this->pos >= this->inSize)
					throw std::runtime_error("gzip 데이터가 중간에 끊겼습니다.");
				value |= (uint32_t)this->in[this->pos++] << this->bitCount;
				this->bitCount += 8;
			}
			this->bitBuffer = value >> need;
			this->bitCount -= need;
			return (int)(value & ((1u << need) - 1));
		}
		static void build(Huffman& h, const short* lengths, int n) {
			h.count.assign(16, 0);
			h.symbol.assign(n, 0);
			for (int s = 0; s < n; ++s)
				h.count[lengths[s]]++;
			h.count[0] = 0;
			short offsets[16] = { 0 };
			for (int len = 1; len < 15; ++len)
				offsets[len + 1] = offsets[len] + h.count[len];
			for (int s = 0; s < n; ++s) {
				if (lengths[s] != 0)
					h.symbol[offsets[lengths[s]]++] = (short)s;
			}
		}
		int decode(const Huffman& h) {
			int code = 0;
			int first = 0;
			int index = 0;
			for (int len = 1; len <= 15; ++len) {
				code |= this->bits(1);
				int count = h.count[len];
				if (code - count < first)
					return h.symbol[index + (code - first)];
				index += count;
				first += count;
				first <<= 1;
				code <<= 1;
			}
			throw std::runtime_error("잘못된 허프만 코드");
		}
		void stored() {
			this->bitBuffer = 0;
			this->bitCount = 0;
			if (this->pos + 4 > this->inSize)
				throw std::runtime_error("gzip 데이터가 중간에 끊겼습니다.");
			int length = this->in[this->pos] | (this->in[this->pos + 1] << 8);
			this->pos += 4;
			if (this->pos + length > this->inSize)
				throw std::runtime_error("gzip 데이터가 중간에 끊겼습니다.");
			this->out.insert(this->out.end(), this->in + this->pos, this->in + this->pos + length);
			this->pos += length;
		}
		void codes(const Huffman& lengthCode, const Huffman& distCode) {
			static const short lengthBase[29] = { 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258 };
			static const short lengthExtra[29] = { 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0 };
			static const short distBase[30] = { 1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577 };
			static const short distExtra[30] = { 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13 };
			while (true) {
				int symbol = this->decode(lengthCode);
				if (symbol < 256) {
					this->out.push_back((uint8_t)symbol);
					continue;
				}
				if (symbol == 256)
					return;
				symbol -= 257;
				if (symbol >= 29)
					throw std::runtime_error("잘못된 길이 코드");
				int length = lengthBase[symbol] + this->bits(lengthExtra[symbol]);
				int distSymbol = this->decode(distCode);
				if (distSymbol >= 30)
					throw std::runtime_error("잘못된 거리 코드");
				size_t dist = (size_t)distBase[distSymbol] + this->bits(distExtra[distSymbol]);
				if (dist > this->out.size())
					throw std::runtime_error("잘못된 거리");
				size_t from = this->out.size() - dist;
				for (int i = 0; i < length; ++i)
					this->out.push_back(this->out[from + i]);
			}
		}
		void fixed() {
			short lengths[320];
			int s = 0;
			for (; s < 144; ++s)
				lengths[s] = 8;
			for (; s < 256; ++s)
				lengths[s] = 9;
			for (; s < 280; ++s)
				lengths[s] = 7;
			for (; s < 288; ++s)
				lengths[s] = 8;
			Huffman lengthCode;
			build(lengthCode, lengths, 288);
			for (s = 0; s < 30; ++s)
				lengths[s] = 5;
			Huffman distCode;
			build(distCode, lengths, 30);
			this->codes(lengthCode, distCode);
		}
		void dynamic() {
			static const short order[19] = { 16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15 };
			int nlen = this->bits(5) + 257;
			int ndist = this->bits(5) + 1;
			int ncode = this->bits(4) + 4;
			short lengths[320] = { 0 };
			for (int i = 0; i < ncode; ++i)
				lengths[order[i]] = (short)this->bits(3);
			Huffman lencode;
			build(lencode, lengths, 19);
			int index = 0;
			while (index < nlen + ndist) {
				int symbol = this->decode(lencode);
				if (symbol < 16) {
					lengths[index++] = (short)symbol;
					continue;
				}
				short value = 0;
				int repeat = 0;
				if (symbol == 16) {
					if (index == 0)
						throw std::runtime_error("잘못된 반복 코드");
					value = lengths[index - 1];
					repeat = 3 + this->bits(2);
				}
				else if (symbol == 17) {
					repeat = 3 + this->bits(3);
				}
				else {
					repeat = 11 + this->bits(7);
				}
				if (index + repeat > nlen + ndist)
					throw std::runtime_error("잘못된 길이 표");
				while (repeat-- > 0)
					lengths[index++] = value;
			}
			Huffman lengthCode;
			build(lengthCode, lengths, nlen);
			Huffman distCode;
			build(distCode, lengths + nlen, ndist);
			this->codes(lengthCode, distCode);
		}
	public:
		Inflater(const uint8_t* data, size_t size, size_t start, std::vector<uint8_t>& output) : in(data), inSize(size), pos(start), out(output) {}

		void run() {
			int last = 0;
			while (last == 0) {
				last = this->bits(1);
				int type = this->bits(2);
				if (type == 0)
					this->stored();
				else if (type == 1)
					this->fixed();
				else if (type == 2)
					this->dynamic();
				else
					throw std::runtime_error("잘못된 블록 형식");
			}
		}
	};

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
		std::vector<uint8_t> output;
		output.reserve(data.size() * 4);
		Inflater inflater(data.data(), data.size(), pos, output);
		inflater.run();
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
