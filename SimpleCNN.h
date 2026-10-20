/*
 * MIT License
 *
 * Copyright (c) 2023 Sangmin Lee
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software
 * and associated documentation files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom
 * the Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or
 * substantial portions of the Software.
 *
 * Convolutional Neural Network (CNN) Library in C++
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
 * PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
 * FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 * Contact Information:
 *
 * Email: whelfjhte79@naver.com
 */

// 사용법 (이 파일 하나만 include 하면 된다)
//
//   #include "SimpleCNN.h"
//   using namespace cnn;
//
//   Dataset train = Dataset::loadFolder("images", 64, 64);    // 폴더 = 클래스
//   CNN model(Device::CUDA);                                  // GPU 가 없으면 자동으로 CPU
//   model.add(new Conv(3, 16, 1, 1, ACTIVATION::ReLU));
//   model.add(new Pooling(POOLING::Max));
//   model.add(new FullyConnected(10, ACTIVATION::Softmax));
//   model.compile(OPTIMIZER::Adam, LOSS::CategoricalCrossentropy);
//   model.fit(train, 5, 64);
//   int label = model.predict("cat.jpg");

#ifndef SIMPLE_CNN_H
#define SIMPLE_CNN_H

#include "simplecnn/Core.h"
#include "simplecnn/Cuda.h"
#include "simplecnn/Tensor.h"
#include "simplecnn/Ops.h"
#include "simplecnn/Layers.h"
#include "simplecnn/Optimizer.h"
#include "simplecnn/Image.h"
#include "simplecnn/Dataset.h"

#include <type_traits>

namespace cnn {

	struct History {
		std::vector<float> loss;
		std::vector<float> accuracy;
		std::vector<float> validationAccuracy;
	};

	class CNN {
	private:
		std::vector<std::unique_ptr<Layer>> layers;
		std::vector<Tensor> outputs;   // 레이어별 출력 (역전파에 다시 쓰인다)
		Tensor input;
		Tensor gradA;                  // 기울기는 두 개를 번갈아 쓴다 (레이어 수와 무관하게 2개)
		Tensor gradB;
		Tensor labelTensor;
		Tensor lossTensor;
		Tensor predTensor;
		Context ctx;
		Optimizer optimizer;
		LOSS lossType = LOSS::CategoricalCrossentropy;
		Device requested = Device::CPU;
		bool compiled = false;
		bool built = false;
		bool learningRateSet = false;
		Shape inputShape;
		std::mt19937 rng;
		Dataset ownData;
		std::vector<float> hostBatch;
		std::vector<float> hostLabels;
		std::vector<float> hostLoss;
		std::vector<float> hostPred;

		Device resolve(Device device) {
			if (device != Device::CUDA)
				return Device::CPU;
			cuda::Runtime& runtime = cuda::Runtime::instance();
			if (runtime.available()) {
				if (this->ctx.verbose)
					std::printf("[SimpleCNN] GPU: %s\n", runtime.info().c_str());
				return Device::CUDA;
			}
			std::cerr << "[SimpleCNN] CUDA 를 사용할 수 없어 CPU 로 실행합니다.\n  이유: " << runtime.error() << "\n";
			return Device::CPU;
		}
		std::vector<Param*> allParams() {
			std::vector<Param*> all;
			for (auto& layer : this->layers) {
				for (Param* p : layer->params())
					all.push_back(p);
			}
			return all;
		}
		void build(const Shape& sample) {
			if (this->built) {
				if (sample.c != this->inputShape.c || sample.h != this->inputShape.h || sample.w != this->inputShape.w)
					throw std::runtime_error("입력 크기 " + sample.str() + " 가 모델 입력 " + this->inputShape.str() + " 과 다릅니다.");
				return;
			}
			if (this->layers.empty())
				throw std::runtime_error("레이어가 없습니다. add() 로 레이어를 먼저 쌓아주세요.");
			this->inputShape = Shape(1, sample.c, sample.h, sample.w);
			Shape s = this->inputShape;
			for (auto& layer : this->layers)
				s = layer->build(s, this->rng);
			this->outputs.resize(this->layers.size());
			for (auto& layer : this->layers)
				layer->toDevice(this->ctx.device);
			this->built = true;
		}
		void ensureCompiled() {
			if (!this->compiled)
				this->compile(OPTIMIZER::Adam, LOSS::CategoricalCrossentropy);
		}
		int outputUnits() const { return (int)this->layers.back()->outShape.sample(); }
		bool fusedLoss() const {
			ACTIVATION last = this->layers.back()->act;
			bool crossEntropy = this->lossType == LOSS::CategoricalCrossentropy || this->lossType == LOSS::SparseCategoricalCrossentropy;
			if (last == ACTIVATION::Softmax && crossEntropy && this->outputUnits() > 1)
				return true;
			return last == ACTIVATION::Sigmoid && this->lossType == LOSS::BinaryCrossentropy;
		}
		int lossCode() const {
			switch (this->lossType) {
			case LOSS::MSE: return 0;
			case LOSS::RMSE: return 1;
			case LOSS::BinaryCrossentropy: return 2;
			case LOSS::CategoricalCrossentropy: return 3;
			default: return 4;
			}
		}
		void checkLabels(const Dataset& data) {
			int units = this->outputUnits();
			int classes = data.classCount();
			bool crossEntropy = this->lossType == LOSS::CategoricalCrossentropy || this->lossType == LOSS::SparseCategoricalCrossentropy;
			if (crossEntropy && units == 1)
				throw std::runtime_error("출력 유닛이 1개면 LOSS::BinaryCrossentropy 를 쓰거나 마지막 FullyConnected 를 클래스 수만큼 늘리세요.");
			if (units > 1 && classes > units) {
				throw std::runtime_error("라벨 종류(" + std::to_string(classes) + ")가 마지막 레이어 유닛 수(" + std::to_string(units) + ")보다 많습니다. 마지막 FullyConnected 를 " + std::to_string(classes) + " 로 바꾸세요.");
			}
		}
		void setInput(const float* data, int count) {
			this->input.resize(Shape(count, this->inputShape.c, this->inputShape.h, this->inputShape.w), this->ctx.device);
			this->input.upload(data, this->input.size());
		}
		void forward() {
			const Tensor* x = &this->input;
			for (size_t i = 0; i < this->layers.size(); ++i) {
				this->layers[i]->forward(*x, this->outputs[i], this->ctx);
				x = &this->outputs[i];
			}
		}
		// 손실값과 출력 기울기 계산. 맞춘 개수를 correct 에 더한다.
		float lossAndGrad(int rows, const float* labels, int& correct) {
			Tensor& y = this->outputs.back();
			int cols = (int)y.shape.sample();
			int fused = 0;
			if (this->fusedLoss())
				fused = 1;
			this->gradA.resize(y.shape, this->ctx.device);
			this->hostLoss.resize(rows);
			this->hostPred.resize(rows);
			int code = this->lossCode();
			if (this->ctx.device == Device::CUDA) {
				this->labelTensor.resize((size_t)rows, Device::CUDA);
				this->labelTensor.upload(labels, rows);
				this->lossTensor.resize((size_t)rows, Device::CUDA);
				this->predTensor.resize((size_t)rows, Device::CUDA);
				cuda::Runtime& rt = cuda::Runtime::instance();
				rt.launch("loss_kernel", (size_t)rows, (const float*)y.data(), (const float*)this->labelTensor.data(), this->gradA.data(), this->lossTensor.data(), rows, cols, code, fused);
				rt.launch("argmax_rows_kernel", (size_t)rows, (const float*)y.data(), this->predTensor.data(), rows, cols);
				this->lossTensor.download(this->hostLoss.data(), rows);
				this->predTensor.download(this->hostPred.data(), rows);
			}
			else {
				this->lossCpu(y.data(), labels, this->gradA.data(), rows, cols, code, fused);
			}
			double total = 0.0;
			for (int r = 0; r < rows; ++r) {
				total += this->hostLoss[r];
				if ((int)this->hostPred[r] == (int)labels[r])
					++correct;
			}
			float value = (float)(total / rows);
			if (code == 0 || code == 1)
				value /= (float)cols;
			if (code == 1) {
				float rmse = std::sqrt(value);
				float scale = 1.0f / (2.0f * rmse + 1e-7f);
				if (this->ctx.device == Device::CUDA) {
					cuda::Runtime::instance().launch("scale_kernel", this->gradA.size(), this->gradA.data(), scale, (int)this->gradA.size());
				}
				else {
					for (size_t i = 0; i < this->gradA.size(); ++i)
						this->gradA.data()[i] *= scale;
				}
				value = rmse;
			}
			return value;
		}
		// CUDA loss_kernel / argmax_rows_kernel 과 같은 계산
		void lossCpu(const float* y, const float* labels, float* grad, int rows, int cols, int code, int fused) {
			float invN = 1.0f / (float)rows;
			for (int r = 0; r < rows; ++r) {
				const float* yr = y + (size_t)r * cols;
				float* gr = grad + (size_t)r * cols;
				float total = 0.0f;
				for (int j = 0; j < cols; ++j) {
					float t = 0.0f;
					if (cols == 1)
						t = labels[r];
					else if (j == (int)labels[r])
						t = 1.0f;
					float v = yr[j];
					if (code == 0 || code == 1) {
						total += (v - t) * (v - t);
						gr[j] = 2.0f * (v - t) * invN / (float)cols;
					}
					else if (code == 2) {
						total -= (t * std::log(v + 1e-7f) + (1.0f - t) * std::log(1.0f - v + 1e-7f)) / (float)cols;
						if (fused)
							gr[j] = (v - t) * invN / (float)cols;
						else
							gr[j] = (v - t) / ((v + 1e-7f) * (1.0f - v + 1e-7f)) * invN / (float)cols;
					}
					else {
						if (t > 0.0f)
							total -= std::log(v + 1e-7f);
						if (fused)
							gr[j] = (v - t) * invN;
						else
							gr[j] = -t / (v + 1e-7f) * invN;
					}
				}
				this->hostLoss[r] = total;
				this->hostPred[r] = (float)argmax(yr, cols);
			}
		}
		static int argmax(const float* row, int cols) {
			if (cols == 1) {
				if (row[0] > 0.5f)
					return 1;
				return 0;
			}
			return (int)(std::max_element(row, row + cols) - row);
		}
		void backward() {
			Tensor* dy = &this->gradA;
			Tensor* dx = &this->gradB;
			int last = (int)this->layers.size() - 1;
			bool fused = this->fusedLoss();
			for (int i = last; i >= 0; --i) {
				const Tensor* x = &this->input;
				if (i > 0)
					x = &this->outputs[i - 1];
				this->ctx.needInputGrad = i > 0;
				this->ctx.fusedOutput = fused && i == last;
				this->layers[i]->backward(*x, this->outputs[i], *dy, *dx, this->ctx);
				std::swap(dy, dx);
			}
			this->ctx.fusedOutput = false;
			this->ctx.needInputGrad = true;
		}
		static void progress(int epoch, int epochs, size_t done, size_t total, float loss, float acc, double seconds) {
			int width = 30;
			int filled = (int)(width * done / (std::max)(total, (size_t)1));
			std::string bar(filled, '=');
			bar.resize(width, ' ');
			int percent = (int)(100 * done / (std::max)(total, (size_t)1));
			std::printf("\repoch %d/%d [%s] %3d%%  loss %.4f  acc %.2f%%  %.1fs ", epoch, epochs, bar.c_str(), percent, loss, acc * 100.0f, seconds);
			std::fflush(stdout);
		}
	public:
		CNN(Device device = Device::CPU) : rng(1234) {
			this->requested = device;
		}
		// 예전 README 방식: 데이터를 생성자로 넘기고 fit(epochs, batchSize)
		CNN(const V4D& image, const V1D& label, Device device = Device::CPU) : rng(1234) {
			this->requested = device;
			this->ownData = Dataset::fromVectors(image, label);
			if (image.size() != label.size())
				std::cerr << "[SimpleCNN] 이미지 수(" << image.size() << ")와 라벨 수(" << label.size() << ")가 다릅니다.\n";
		}
		CNN(const Dataset& data, Device device = Device::CPU) : rng(1234) {
			this->requested = device;
			this->ownData = data;
		}
		CNN(const CNN&) = delete;
		CNN& operator=(const CNN&) = delete;

		// 레이어 쌓기. new 로 넘긴 레이어는 CNN 이 알아서 지운다.
		CNN& add(Layer* layer) {
			if (this->built)
				throw std::runtime_error("모델이 이미 만들어진 뒤에는 레이어를 추가할 수 없습니다.");
			this->layers.emplace_back(layer);
			return *this;
		}
		// model.add(Conv(3, 16)) 처럼 값으로 넘겨도 된다.
		template <typename T, typename = typename std::enable_if<std::is_base_of<Layer, typename std::decay<T>::type>::value>::type>
		CNN& add(T&& layer) {
			return this->add(new typename std::decay<T>::type(std::move(layer)));
		}

		void compile(OPTIMIZER optimizerType = OPTIMIZER::Adam, LOSS loss = LOSS::CategoricalCrossentropy) {
			this->compile(optimizerType, loss, this->requested);
		}
		// device 에 Device::CUDA 를 주면 GPU 로 학습한다.
		void compile(OPTIMIZER optimizerType, LOSS loss, Device device) {
			float lr = this->optimizer.learningRate;
			this->optimizer.setType(optimizerType);
			if (this->learningRateSet)
				this->optimizer.learningRate = lr;
			this->lossType = loss;
			this->compiled = true;
			this->setDevice(device);
		}
		void setDevice(Device device) {
			this->requested = device;
			Device actual = this->resolve(device);
			if (actual == this->ctx.device)
				return;
			this->ctx.device = actual;
			this->ctx.workspace.free();
			if (this->built) {
				for (auto& layer : this->layers)
					layer->toDevice(actual);
			}
		}
		Device device() const { return this->ctx.device; }
		void setLearningRate(float lr) {
			this->optimizer.learningRate = lr;
			this->learningRateSet = true;
		}
		float learningRate() const { return this->optimizer.learningRate; }
		void setConvAlgorithm(CONV_ALGO algo) { this->ctx.convAlgo = algo; }
		// im2col 등 임시 버퍼 최대 크기. GPU 메모리가 작으면 낮추면 된다. (기본 512MB)
		void setWorkspaceLimit(size_t megabytes) { this->ctx.workspaceLimit = (std::max)((size_t)16, megabytes) << 20; }
		void setVerbose(bool verbose) { this->ctx.verbose = verbose; }
		void setSeed(unsigned seed) { this->rng.seed(seed); }
		size_t layerCount() const { return this->layers.size(); }
		Layer* layer(size_t i) { return this->layers.at(i).get(); }
		Shape inputSize() const { return this->inputShape; }

		History fit(int epochs, int batchSize) {
			return this->fit(this->ownData, epochs, batchSize, nullptr);
		}
		History fit(const Dataset& train, int epochs, int batchSize, const Dataset* validation = nullptr) {
			History history;
			if (train.empty())
				throw std::runtime_error("학습 데이터가 비어 있습니다.");
			this->ensureCompiled();
			this->build(train.shape());
			this->checkLabels(train);
			batchSize = (std::max)(1, batchSize);
			size_t sampleSize = train.shape().sample();
			this->hostBatch.resize(sampleSize * batchSize);
			this->hostLabels.resize(batchSize);
			std::vector<int> order(train.size());
			for (size_t i = 0; i < order.size(); ++i)
				order[i] = (int)i;
			std::vector<Param*> params = this->allParams();
			if (this->ctx.verbose)
				std::printf("[SimpleCNN] 학습 시작: %zu 장, 배치 %d, %s, 학습률 %g\n", train.size(), batchSize, toString(this->ctx.device), this->optimizer.learningRate);
			for (int epoch = 1; epoch <= epochs; ++epoch) {
				std::shuffle(order.begin(), order.end(), this->rng);
				Timer timer;
				double lossSum = 0.0;
				int correct = 0;
				size_t seen = 0;
				size_t step = 0;
				size_t printEvery = (std::max)((size_t)1, train.size() / batchSize / 50);
				for (size_t begin = 0; begin < train.size(); begin += batchSize) {
					int count = (int)(std::min)((size_t)batchSize, train.size() - begin);
					train.batch(order, begin, count, this->hostBatch.data(), this->hostLabels.data());
					this->setInput(this->hostBatch.data(), count);
					this->forward();
					float loss = this->lossAndGrad(count, this->hostLabels.data(), correct);
					this->backward();
					this->optimizer.update(params, this->ctx.device);
					lossSum += (double)loss * count;
					seen += count;
					++step;
					if (this->ctx.verbose && (step % printEvery == 0 || seen == train.size()))
						progress(epoch, epochs, seen, train.size(), (float)(lossSum / seen), (float)correct / seen, timer.seconds());
				}
				history.loss.push_back((float)(lossSum / seen));
				history.accuracy.push_back((float)correct / seen);
				if (this->ctx.verbose)
					std::printf(" (%.0f 장/초)", seen / (std::max)(timer.seconds(), 1e-9));
				if (validation != nullptr && !validation->empty()) {
					bool verbose = this->ctx.verbose;
					this->ctx.verbose = false;
					float valAcc = this->evaluate(*validation, batchSize);
					this->ctx.verbose = verbose;
					history.validationAccuracy.push_back(valAcc);
					if (this->ctx.verbose)
						std::printf("  val acc %.2f%%", valAcc * 100.0f);
				}
				if (this->ctx.verbose)
					std::printf("\n");
			}
			if (this->ctx.verbose)
				this->printMemory();
			return history;
		}

		// 정확도 (0~1)
		float evaluate(const Dataset& data, int batchSize = 256) {
			if (data.empty())
				return 0.0f;
			std::vector<int> predictions = this->predict(data, batchSize);
			int correct = 0;
			for (size_t i = 0; i < predictions.size(); ++i) {
				if (predictions[i] == (int)data.label(i))
					++correct;
			}
			float acc = (float)correct / (float)data.size();
			if (this->ctx.verbose)
				std::printf("[SimpleCNN] 정확도: %.2f%% (%d / %zu)\n", acc * 100.0f, correct, data.size());
			return acc;
		}
		float accuracy() { return this->evaluate(this->ownData); }

		std::vector<int> predict(const Dataset& data, int batchSize = 256) {
			this->ensureCompiled();
			this->build(data.shape());
			batchSize = (std::max)(1, batchSize);
			std::vector<int> result;
			std::vector<int> order(data.size());
			for (size_t i = 0; i < order.size(); ++i)
				order[i] = (int)i;
			std::vector<float> batch(data.shape().sample() * batchSize);
			std::vector<float> labels(batchSize);
			std::vector<float> out;
			for (size_t begin = 0; begin < data.size(); begin += batchSize) {
				int count = (int)(std::min)((size_t)batchSize, data.size() - begin);
				data.batch(order, begin, count, batch.data(), labels.data());
				this->setInput(batch.data(), count);
				this->forward();
				const Tensor& y = this->outputs.back();
				int cols = (int)y.shape.sample();
				out.resize(y.size());
				y.download(out.data(), out.size());
				for (int r = 0; r < count; ++r)
					result.push_back(argmax(out.data() + (size_t)r * cols, cols));
			}
			return result;
		}
		// 이미지 한 장의 클래스별 확률(마지막 레이어 출력)
		std::vector<float> predictProba(const float* chw) {
			if (!this->built)
				throw std::runtime_error("모델이 아직 만들어지지 않았습니다. fit() 이나 load() 를 먼저 호출하세요.");
			this->setInput(chw, 1);
			this->forward();
			return this->outputs.back().toVector();
		}
		std::vector<float> predictProba(const Image& image) {
			if (!this->built)
				throw std::runtime_error("모델이 아직 만들어지지 않았습니다. fit() 이나 load() 를 먼저 호출하세요.");
			std::vector<float> chw(this->inputShape.sample());
			image.toCHW(chw.data(), this->inputShape.c, this->inputShape.h, this->inputShape.w);
			return this->predictProba(chw.data());
		}
		int predict(const Image& image) {
			std::vector<float> p = this->predictProba(image);
			return argmax(p.data(), (int)p.size());
		}
		int predict(const std::string& imagePath) {
			Image image = Image::load(imagePath);
			if (image.empty())
				throw std::runtime_error("이미지를 읽을 수 없습니다: " + imagePath);
			return this->predict(image);
		}
		int predict(const char* imagePath) { return this->predict(std::string(imagePath)); }
		int predict(const V3D& image) {
			Dataset one = Dataset::fromVectors(V4D(1, image), V1D(1, 0.0f));
			std::vector<float> chw(one.shape().sample());
			one.sample(0, chw.data());
			if (!this->built) {
				this->ensureCompiled();
				this->build(one.shape());
			}
			std::vector<float> p = this->predictProba(chw.data());
			return argmax(p.data(), (int)p.size());
		}

		// ------------------------------------------------------------ 저장 / 불러오기
		// 텍스트 헤더(레이어 구성) + 바이너리 가중치. load 는 레이어를 쌓지 않아도 그대로 복원된다.
		void save(const std::string& path) {
			if (!this->built)
				throw std::runtime_error("학습(fit)한 뒤에 저장할 수 있습니다.");
			std::ofstream file(toPath(path), std::ios::binary);
			if (!file)
				throw std::runtime_error("파일을 만들 수 없습니다: " + path);
			file << "SimpleCNN 2\n";
			file << "input " << this->inputShape.c << " " << this->inputShape.h << " " << this->inputShape.w << "\n";
			file << "layers " << this->layers.size() << "\n";
			for (auto& layer : this->layers)
				file << layer->config() << "\n";
			for (Param* p : this->allParams()) {
				std::vector<float> host = p->value.toVector();
				uint64_t count = host.size();
				file.write(reinterpret_cast<const char*>(&count), sizeof(count));
				file.write(reinterpret_cast<const char*>(host.data()), (std::streamsize)(count * sizeof(float)));
			}
			if (this->ctx.verbose)
				std::printf("[SimpleCNN] 저장: %s\n", path.c_str());
		}
		void load(const std::string& path) {
			std::ifstream file(toPath(path), std::ios::binary);
			if (!file)
				throw std::runtime_error("파일을 열 수 없습니다: " + path);
			std::string line;
			std::getline(file, line);
			if (line != "SimpleCNN 2")
				throw std::runtime_error("SimpleCNN 모델 파일이 아닙니다: " + path);
			std::string word;
			Shape shape;
			size_t count = 0;
			file >> word >> shape.c >> shape.h >> shape.w;
			file >> word >> count;
			std::getline(file, line);
			this->layers.clear();
			this->outputs.clear();
			this->built = false;
			for (size_t i = 0; i < count; ++i) {
				std::getline(file, line);
				this->layers.emplace_back(createLayer(line));
			}
			Shape s = Shape(1, shape.c, shape.h, shape.w);
			for (auto& layer : this->layers)
				s = layer->build(s, this->rng);
			for (Param* p : this->allParams()) {
				uint64_t n = 0;
				file.read(reinterpret_cast<char*>(&n), sizeof(n));
				if (!file || n != p->size())
					throw std::runtime_error("모델 파일이 손상되었습니다: " + path);
				std::vector<float> host((size_t)n);
				file.read(reinterpret_cast<char*>(host.data()), (std::streamsize)(n * sizeof(float)));
				p->value.resize((size_t)n, Device::CPU);
				p->value.upload(host.data(), host.size());
				p->m.free();
				p->v.free();
			}
			this->ensureCompiled();
			this->build(shape);
			if (this->ctx.verbose)
				std::printf("[SimpleCNN] 불러옴: %s (%zu 레이어)\n", path.c_str(), this->layers.size());
		}

		// ------------------------------------------------------------ 정보 출력
		void summary(int batchSize = 32) {
			if (!this->built) {
				if (this->ownData.empty())
					throw std::runtime_error("입력 크기를 알 수 없습니다. summary(Shape(1, 채널, 세로, 가로)) 를 사용하세요.");
				this->ensureCompiled();
				this->build(this->ownData.shape());
			}
			std::printf("\n %-16s %-18s %10s   %s\n", "Layer", "Output", "Params", "Detail");
			std::printf(" ------------------------------------------------------------------\n");
			std::printf(" %-16s %-18s %10s\n", "Input", this->inputShape.str().c_str(), "");
			size_t total = 0;
			size_t activations = 0;
			size_t largest = 0;
			for (auto& layer : this->layers) {
				size_t params = layer->paramCount();
				total += params;
				activations += layer->outShape.sample();
				largest = (std::max)(largest, layer->outShape.sample());
				std::printf(" %-16s %-18s %10zu   %s\n", layer->name().c_str(), layer->outShape.str().c_str(), params, layer->detail().c_str());
			}
			std::printf(" ------------------------------------------------------------------\n");
			std::printf(" 파라미터: %zu (%s)\n", total, formatBytes((double)total * 4).c_str());
			double memory = ((double)activations + 2.0 * largest + this->inputShape.sample()) * batchSize * 4.0;
			std::printf(" 배치 %d 기준 예상 메모리: 활성값 %s + 파라미터/기울기 %s + 작업버퍼 최대 %s\n", batchSize, formatBytes(memory).c_str(),
				formatBytes((double)total * 4 * 4).c_str(), formatBytes((double)this->ctx.workspaceLimit).c_str());
			std::printf(" 장치: %s", toString(this->ctx.device));
			if (this->ctx.device == Device::CUDA)
				std::printf(" - %s", cuda::Runtime::instance().info().c_str());
			std::printf("\n\n");
		}
		void summary(const Shape& sample, int batchSize = 32) {
			this->ensureCompiled();
			this->build(sample);
			this->summary(batchSize);
		}
		void printMemory() const {
			std::printf("[SimpleCNN] 메모리 최대 사용량: CPU %s", formatBytes((double)MemoryStats::get().hostPeak).c_str());
			if (this->ctx.device == Device::CUDA)
				std::printf(", GPU %s", formatBytes((double)cuda::Runtime::instance().memoryPeak()).c_str());
			std::printf("\n");
		}
	};
}

#endif
