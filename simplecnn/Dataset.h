// SimpleCNN - 데이터셋
//
// 메모리 절약 설계
//  - 이미지는 0~255 uint8 로 보관하고, 배치를 만들 때만 0~1 float 로 바꾼다. (float 대비 1/4)
//  - streaming 모드는 파일 경로만 들고 있다가 배치마다 디스크에서 읽는다. (RAM 거의 0)
#ifndef SIMPLE_CNN_DATASET_H
#define SIMPLE_CNN_DATASET_H

#include "Core.h"
#include "Image.h"

namespace cnn {

	class Dataset {
	private:
		int c = 0;
		int h = 0;
		int w = 0;
		std::vector<uint8_t> bytes;
		std::vector<float> floats;
		float floatScale = 1.0f;
		std::vector<std::filesystem::path> files;
		std::vector<float> labelList;

		static uint32_t readBE(std::istream& in) {
			unsigned char b[4] = { 0, 0, 0, 0 };
			in.read(reinterpret_cast<char*>(b), 4);
			return ((uint32_t)b[0] << 24) | ((uint32_t)b[1] << 16) | ((uint32_t)b[2] << 8) | (uint32_t)b[3];
		}
		static std::vector<std::filesystem::path> listImages(const std::filesystem::path& dir) {
			std::vector<std::filesystem::path> found;
			std::error_code ec;
			for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
				if (entry.is_regular_file(ec) && isImageFile(entry.path()))
					found.push_back(entry.path());
			}
			std::sort(found.begin(), found.end());
			return found;
		}
		void setShape(int channels, int height, int width) {
			this->c = channels;
			this->h = height;
			this->w = width;
		}
	public:
		std::vector<std::string> classNames;

		size_t size() const { return this->labelList.size(); }
		bool empty() const { return this->labelList.empty(); }
		Shape shape() const { return Shape(1, this->c, this->h, this->w); }
		bool streaming() const { return !this->files.empty(); }
		float label(size_t i) const { return this->labelList[i]; }
		const std::vector<float>& labels() const { return this->labelList; }
		int classCount() const {
			float maxLabel = 0.0f;
			for (float v : this->labelList)
				maxLabel = (std::max)(maxLabel, v);
			return (int)maxLabel + 1;
		}
		size_t memoryBytes() const {
			return this->bytes.size() + this->floats.size() * sizeof(float) + this->labelList.size() * sizeof(float);
		}

		// 0~255 이미지 한 장 추가. 첫 이미지(또는 지정한 크기)에 맞춰 크기를 통일한다.
		void add(const Image& image, float label) {
			if (this->c == 0)
				this->setShape(image.channels, image.height, image.width);
			Image fitted = image.withChannels(this->c).resized(this->w, this->h);
			size_t offset = this->bytes.size();
			this->bytes.resize(offset + (size_t)this->c * this->h * this->w);
			for (int ch = 0; ch < this->c; ++ch)
				for (int y = 0; y < this->h; ++y)
					for (int x = 0; x < this->w; ++x)
						this->bytes[offset + ((size_t)ch * this->h + y) * this->w + x] = fitted.at(x, y, ch);
			this->labelList.push_back(label);
		}

		// i 번째 샘플을 CHW 0~1 float 로 out 에 채운다.
		void sample(size_t i, float* out) const {
			size_t n = (size_t)this->c * this->h * this->w;
			if (!this->files.empty()) {
				Image image = Image::load(this->files[i]);
				if (image.empty()) {
					std::fill(out, out + n, 0.0f);
					return;
				}
				image.toCHW(out, this->c, this->h, this->w);
				return;
			}
			if (!this->floats.empty()) {
				const float* src = this->floats.data() + i * n;
				for (size_t k = 0; k < n; ++k)
					out[k] = src[k] * this->floatScale;
				return;
			}
			const uint8_t* src = this->bytes.data() + i * n;
			for (size_t k = 0; k < n; ++k)
				out[k] = src[k] * (1.0f / 255.0f);
		}
		// order[begin .. begin+count) 순서로 배치를 만든다.
		void batch(const std::vector<int>& order, size_t begin, int count, float* out, float* outLabels) const {
			size_t n = (size_t)this->c * this->h * this->w;
			parallelFor(count, [&](int k) {
				int index = order[begin + k];
				this->sample(index, out + n * k);
				outLabels[k] = this->labelList[index];
			});
		}
		Dataset subset(size_t begin, size_t count) const {
			Dataset out;
			out.setShape(this->c, this->h, this->w);
			out.classNames = this->classNames;
			out.floatScale = this->floatScale;
			count = (std::min)(count, this->size() - (std::min)(begin, this->size()));
			size_t n = (size_t)this->c * this->h * this->w;
			out.labelList.assign(this->labelList.begin() + begin, this->labelList.begin() + begin + count);
			if (!this->files.empty())
				out.files.assign(this->files.begin() + begin, this->files.begin() + begin + count);
			else if (!this->floats.empty())
				out.floats.assign(this->floats.begin() + begin * n, this->floats.begin() + (begin + count) * n);
			else
				out.bytes.assign(this->bytes.begin() + begin * n, this->bytes.begin() + (begin + count) * n);
			return out;
		}
		// 섞은 뒤 앞쪽을 학습용, 뒤쪽을 검증용으로 나눈다.
		std::pair<Dataset, Dataset> split(float testRatio, unsigned seed = 42) const {
			std::vector<int> order(this->size());
			for (size_t i = 0; i < order.size(); ++i)
				order[i] = (int)i;
			std::mt19937 rng(seed);
			std::shuffle(order.begin(), order.end(), rng);
			Dataset shuffled;
			shuffled.setShape(this->c, this->h, this->w);
			shuffled.classNames = this->classNames;
			shuffled.floatScale = this->floatScale;
			size_t n = (size_t)this->c * this->h * this->w;
			for (int index : order) {
				shuffled.labelList.push_back(this->labelList[index]);
				if (!this->files.empty())
					shuffled.files.push_back(this->files[index]);
				else if (!this->floats.empty())
					shuffled.floats.insert(shuffled.floats.end(), this->floats.begin() + index * n, this->floats.begin() + (index + 1) * n);
				else
					shuffled.bytes.insert(shuffled.bytes.end(), this->bytes.begin() + index * n, this->bytes.begin() + (index + 1) * n);
			}
			size_t testCount = (size_t)(this->size() * testRatio);
			return std::make_pair(shuffled.subset(0, this->size() - testCount), shuffled.subset(this->size() - testCount, testCount));
		}

		// ------------------------------------------------------------ 만들기
		// 예전 방식: V4D [개수][채널][세로][가로] + 라벨. 값이 1 보다 크면 0~255 로 보고 자동으로 나눈다.
		static Dataset fromVectors(const V4D& images, const V1D& labels) {
			Dataset data;
			if (images.empty())
				return data;
			data.setShape((int)images[0].size(), (int)images[0][0].size(), (int)images[0][0][0].size());
			size_t n = (size_t)data.c * data.h * data.w;
			data.floats.resize(n * images.size(), 0.0f);
			float maxValue = 0.0f;
			for (size_t i = 0; i < images.size(); ++i) {
				for (int ch = 0; ch < data.c && ch < (int)images[i].size(); ++ch) {
					for (int y = 0; y < data.h && y < (int)images[i][ch].size(); ++y) {
						for (int x = 0; x < data.w && x < (int)images[i][ch][y].size(); ++x) {
							float v = images[i][ch][y][x];
							data.floats[i * n + ((size_t)ch * data.h + y) * data.w + x] = v;
							maxValue = (std::max)(maxValue, v);
						}
					}
				}
			}
			if (maxValue > 1.5f)
				data.floatScale = 1.0f / 255.0f;
			data.labelList.assign(labels.begin(), labels.begin() + (std::min)(labels.size(), images.size()));
			data.labelList.resize(images.size(), 0.0f);
			return data;
		}

		// MNIST / Fashion-MNIST (IDX 형식, 압축 해제된 파일)
		static Dataset loadMNIST(const std::string& imagesFile, const std::string& labelsFile, size_t limit = 0) {
			Dataset data;
			std::ifstream images(toPath(imagesFile), std::ios::binary);
			std::ifstream labels(toPath(labelsFile), std::ios::binary);
			if (!images || !labels)
				throw std::runtime_error("MNIST 파일을 열 수 없습니다: " + imagesFile);
			if (readBE(images) != 2051 || readBE(labels) != 2049)
				throw std::runtime_error("MNIST 파일 형식이 아닙니다: " + imagesFile);
			size_t count = readBE(images);
			size_t labelCount = readBE(labels);
			int rows = (int)readBE(images);
			int cols = (int)readBE(images);
			count = (std::min)(count, labelCount);
			if (limit > 0)
				count = (std::min)(count, limit);
			data.setShape(1, rows, cols);
			data.bytes.resize(count * rows * cols);
			images.read(reinterpret_cast<char*>(data.bytes.data()), (std::streamsize)data.bytes.size());
			std::vector<uint8_t> raw(count);
			labels.read(reinterpret_cast<char*>(raw.data()), (std::streamsize)count);
			data.labelList.assign(raw.begin(), raw.end());
			for (int d = 0; d < 10; ++d)
				data.classNames.push_back(std::to_string(d));
			return data;
		}

		// 폴더 하나 = 클래스 하나.   root/고양이/*.jpg, root/강아지/*.jpg ...
		// streaming = true 면 경로만 기억하고 학습 중에 읽는다. (수십 GB 데이터셋용)
		static Dataset loadFolder(const std::string& root, int width, int height, int channels = 3, bool streaming = false) {
			Dataset data;
			data.setShape(channels, height, width);
			std::vector<std::filesystem::path> dirs;
			std::error_code ec;
			for (const auto& entry : std::filesystem::directory_iterator(toPath(root), ec)) {
				if (entry.is_directory(ec))
					dirs.push_back(entry.path());
			}
			if (ec)
				throw std::runtime_error("폴더를 열 수 없습니다: " + root);
			std::sort(dirs.begin(), dirs.end());
			for (size_t label = 0; label < dirs.size(); ++label) {
				data.classNames.push_back(pathString(dirs[label].filename()));
				for (const auto& file : listImages(dirs[label]))
					data.addFile(file, (float)label, streaming);
			}
			if (data.empty())
				throw std::runtime_error("이미지가 없습니다: " + root + " (클래스별 하위 폴더에 이미지를 넣어주세요)");
			return data;
		}

		// 폴더의 이미지(이름순) + 라벨 CSV.  CSV 는 "라벨" 또는 "파일명,라벨" 한 줄씩
		static Dataset loadFiles(const std::string& imageDir, const std::string& labelCsv, int width, int height, int channels = 3, bool streaming = false) {
			Dataset data;
			data.setShape(channels, height, width);
			std::vector<std::filesystem::path> images = listImages(toPath(imageDir));
			std::ifstream csv(toPath(labelCsv));
			if (!csv)
				throw std::runtime_error("라벨 파일을 열 수 없습니다: " + labelCsv);
			std::string line;
			size_t index = 0;
			while (std::getline(csv, line)) {
				if (line.empty() || line == "\r")
					continue;
				size_t comma = line.find(',');
				if (comma == std::string::npos) {
					if (index < images.size())
						data.addFile(images[index], std::stof(line), streaming);
					++index;
					continue;
				}
				std::filesystem::path file = toPath(imageDir) / toPath(line.substr(0, comma));
				data.addFile(file, std::stof(line.substr(comma + 1)), streaming);
			}
			return data;
		}
		void addFile(const std::filesystem::path& file, float label, bool streaming) {
			if (streaming) {
				this->files.push_back(file);
				this->labelList.push_back(label);
				return;
			}
			Image image = Image::load(file);
			if (image.empty()) {
				std::cerr << "[SimpleCNN] 이미지를 읽지 못해 건너뜀: " << pathString(file) << "\n";
				return;
			}
			this->add(image, label);
		}
	};
}

#endif
