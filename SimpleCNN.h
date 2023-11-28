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
 * This is a C++ library that implements a Convolutional Neural Network (CNN).
 * It provides tools and utilities for building, training, and using CNNs for various computer vision tasks.
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


#ifndef SIMPLE_CNN_H
#define SIMPLE_CNN_H

#include<vector>
#include<opencv2/opencv.hpp>
#include<opencv2/core.hpp>
#include<iostream>
#include<string>
#include<vector>
#include<dirent.h>
#include<locale.h>
#include<cstring>
#include<algorithm>
#include"ImageRead.h"
#include <random>
#include"kernel.cuh"
#include <chrono>
using namespace cv;
using namespace std;


namespace cnn {
	using V1D = std::vector<float>;
	using V2D = std::vector<V1D>;
	using V3D = std::vector<V2D>;
	using V4D = std::vector<V3D>;
	using V5D = std::vector<V4D>;
	
	enum class LAYER {
		Conv, Pool, Act, Padding, FC, Norm, Flatten, None
	};
	enum class POOLING {
		Max, Average, Min
	};
	enum class ACTIVATION {
		None,Error, Sigmoid, ReLU, Leaky_ReLU, TanH, Maxout, ELU, Softmax
	};
	enum class OPTIMIZER {
		GD, Mini_SGD, Momentum, Adagrad, RMSProp, Adam, Nadam
	};
	enum class LOSS {
		MSE, RMSE, BinaryCrossentropy, CategoricalCrossentropy, SparseCategoricalCrossentropy
	};

	class PrintTest {
	private:
	public:
		void printVec2DSize(V4D data, std::string error_name) {
			cout << error_name << "\n";
			int cnt = 0;
			for (int i = 0; i < data.size(); i++) {
				cnt += data[i].size();
			}
			cnt += data.size();
			cout << "전체크기: " << cnt << "\n";
		}
		void printVec3DSize(V3D data, std::string error_name) {
			cout << error_name << "\n";
			int cnt = 0;
			for (int i = 0; i < data.size(); i++) {
				for (int j = 0; j < data[i].size(); j++) {
					cnt += data[i][j].size();
				}
				cnt += data[i].size();
			}
			cnt += data.size();
			cout << "전체크기: " << cnt << "\n";
		}
		void printVec4DSize(V4D data, std::string error_name) {
			cout << error_name << "\n";
			int cnt = 0;
			for (int i = 0; i < data.size(); i++) {
				for (int j = 0; j < data[i].size(); j++) {
					for (int k = 0; k < data[i][j].size(); k++) {
						cnt+=data[i][j][k].size();
					}
					cnt += data[i][j].size();
				}
				cnt += data[i].size();
			}
			cnt += data.size();
			cout << "전체크기: " << cnt<<"\n";
		}
		void printTestV4D(V4D data, std::string error_name) {
			cout << error_name << "\n";
			for (int i = 0; i < data.size(); i++) {
				for (int j = 0; j < data[i].size(); j++) {
					for (int k = 0; k < data[i][j].size(); k++) {
						for (int l = 0; l < data[i][j][k].size(); l++) {
							cout << fixed;
							cout.precision(4);
							cout << data[i][j][k][l] << " ";
						}
						cout << "\n";
					}
					cout << "\n";
				}
				cout << "\n";
			}
			cout << "출력테스트V4D 끝\n";
		}
		void printTestV3D(V3D data, std::string error_name) {
			cout << error_name << "\n";
			for (int i = 0; i < data.size(); i++) {
				for (int j = 0; j < data[i].size(); j++) {
					for (int k = 0; k < data[i][j].size(); k++) {
						cout << fixed;
						cout.precision(4);
						cout << data[i][j][k] << " ";
					}
					cout << "\n";
				}
				cout << "\n";
			}
			cout << "출력테스트V3D 끝\n";
		}
		void printTestV2D(V2D data, std::string error_name) {
			cout << error_name << "\n";
			for (int i = 0; i < data.size(); i++) {
				for (int j = 0; j < data[i].size(); j++) {
					cout << fixed;
					cout.precision(4);
					cout << data[i][j] << " ";
				}
				cout << "\n";
			}
			cout << "출력테스트V2D 끝\n";
		}
	};

	class RandomGen {
	private:
		std::random_device rd;
		std::mt19937 mt;
		std::uniform_real_distribution<float> dist;
		std::uniform_int_distribution<int> distInt;

	public:
		RandomGen(float min, float max) : mt(rd()), dist(min, max) {}
		RandomGen(int min, int max) : mt(rd()), distInt(min, max) {}

		float getGenRandNumber() {
			return this->dist(mt);
		}
		float getGenRandInt() {
			return (float)this->distInt(mt);
		}
		void randGen(V1D& data) {
			for (int i = 0; i < data.size(); i++)
				data[i] = getGenRandNumber();
		}
		void randGenInt(V1D& data) {
			for (int i = 0; i < data.size(); i++)
				data[i] = static_cast<float>(getGenRandInt());
		}
		void randGen(V2D& data) {
			for (int i = 0; i < data.size(); i++)
				randGen(data[i]);
		}
		void randGenInt(V2D& data) {
			for (int i = 0; i < data.size(); i++)
				randGenInt(data[i]);
		}
		void randGen(V3D& data, bool initialized) {
			if (!initialized) {
				for (int i = 0; i < data.size(); i++)
					randGen(data[i]);
			}
		}
		void randGenInt(V3D& data, bool initialized) {
			if (!initialized) {
				for (int i = 0; i < data.size(); i++)
					randGenInt(data[i]);
			}
		}
	};

	class Layer 
#ifdef _CUDA_GPU_
		: public LayerGPU
#endif
	{
	public: 
		LAYER layerType;
		int filterRowCol;
		int filterSize;
		int stride;
		int paddingSize;
		ACTIVATION act;
		POOLING poolingName;
		int denseSize;
	
		V4D data4D;
		V3D data3D;
		V2D data2D;
		V1D data1D;
	
		LAYER& getLayerType() { return this->layerType; }
		virtual ACTIVATION getActivationType() { return this->act; }
		virtual V4D getData4D() { return this->data4D; }
		virtual V3D getData3D() { return this->data3D; }
		virtual V2D getData2D() { return this->data2D; }
		virtual V1D getData1D() { return this->data1D; }
		V4D* dummy4D;
		V3D* dummy3D;
		V2D* dummy2D;
		V1D* dummy1D;
		virtual void setData(V4D data) {}
		virtual void setData(V3D data) {}
		virtual void setData(V2D data) {}
		virtual void setData(V1D data) {}
	
		virtual void setFilter3D(V3D* filter) {}
		virtual V3D* getFilter3D() { return dummy3D; }
		virtual void setWeight(V1D weight) {}
		virtual V1D* getWeight1D() { return dummy1D; }
		virtual V2D* getWeight2D() { return dummy2D; }
		virtual V3D* getWeight3D() { return dummy3D; }
		virtual void setWeight3D(V3D* weight) {}

		virtual V2D* getDelta2D() { return this->dummy2D; }
		virtual V4D* getDelta4D() { return this->dummy4D; }
		virtual void setDelta2D(V2D* delta){}
		virtual void setDelta4D(V4D* delta) {}
		virtual V4D calculate(V4D data) { return data4D; }
		virtual V2D calculate(V4D data, LAYER layer) { return data2D; }
		virtual V2D calculate(V2D data) { return data2D; }

		virtual V2D* backward(V2D* delta) { return dummy2D; }
		virtual V4D* backward(V4D* delta) { return dummy4D; }

#ifdef _CUDA_GPU_
		virtual V4D calculateGPU(V4D data) override { return LayerGPU::calculateGPU(data); }
		virtual V2D calculateGPU(V2D data) override { return LayerGPU::calculateGPU(data); }
		virtual V1D calculateGPU(V1D data) override { return LayerGPU::calculateGPU(data); }
		virtual V2D* backwardGPU(V2D* delta) override { return LayerGPU::backwardGPU(delta); }
#endif //_CUDA_GPU_



		void deleteV4D(V4D* data) {
			for (auto& d3d : *data) {
				for (auto& d2d : d3d) {
					for (auto& d1d : d2d) {
						d1d.clear();
					}
					d2d.clear();
				}
				d3d.clear();
			}
			data->clear();
			delete data;
		}
		void deleteV3D(V3D* data) {
			for (auto& d2d : *data) {
				for (auto& d1d : d2d) {
					d1d.clear();
				}
				d2d.clear();
			}

			data->clear();
			delete data;
		}
		void deleteV2D(V2D* data) {
			for (auto& d1d : *data) {
				d1d.clear();
			}
			data->clear();
			delete data;
		}

		PrintTest printTest;
	};

	
	class Activation : public Layer
#ifdef _CUDA_GPU_
		, ActivationGPU
#endif //_CUDA_GPU_
	{
	private:
	public:
		Activation() {
			this->layerType = LAYER::Act;
			this->act = ACTIVATION::ReLU;
		}
		Activation(ACTIVATION act) {
			this->layerType = LAYER::Act;
			this->act = act;
		}
		virtual V4D getData4D() override {
			return this->data4D;
		}
		virtual void setData(V4D data4D) override {
			this->data4D = data4D;
		}
		V1D calculate1D(V1D data) {

			if (this->act == ACTIVATION::ReLU) {
				for (int i = 0; i < data.size(); ++i) {
					data[i] = max(data[i], 0.0f);
				}
			}
			else if (this->act == ACTIVATION::Sigmoid) {
				for (int i = 0; i < data.size(); ++i) { data[i] = 1.0f / (1.0f + exp(-data[i])); }
			}
			else if (this->act == ACTIVATION::Softmax) {
				float maxVal = -FLT_MAX;
				for (int i = 0; i < data.size(); ++i) maxVal = max(maxVal, data[i]);
				float expSum = 0.0f;
				for (int i = 0; i < data.size(); ++i) {
					data[i] = exp(data[i] - maxVal);
					expSum += data[i];
				}
				for (int i = 0; i < data.size(); ++i) data[i] /= expSum;
			}
			else if (this->act == ACTIVATION::TanH) {
				for (int i = 0; i < data.size(); ++i) data[i] = tanh(data[i]);
			}
			else if (this->act == ACTIVATION::Leaky_ReLU) {
				float leak = 0.1f;
				for (int i = 0; i < data.size(); ++i) data[i] = (data[i] > 0) ? data[i] : leak * data[i];
			}
			else if (this->act == ACTIVATION::ELU) {
				float alpha = 0.1f;
				for (int i = 0; i < data.size(); ++i) {
					if (data[i] < 0) data[i] = alpha * (exp(data[i]) - 1.0f);
				}
			}

			return data;
		}
		V2D calculate2D(V2D data) {
			V2D result2D = V2D(0);
			for (int i = 0; i < data.size(); i++) {
				result2D.push_back(this->calculate1D(data[i]));

			}
			return result2D;

		}
		V3D calculate3D(V3D data) {
			V3D result3D = V3D(0);
			for (int i = 0; i < data.size(); i++) {
				result3D.push_back(this->calculate2D(data[i]));
			}
			return result3D;
		}
		V4D calculate(V4D data) override {
			V4D result4D = V4D(0);
			for (int i = 0; i < data.size(); i++) {
				result4D.push_back(this->calculate3D(data[i]));
			}
			return result4D;
		}
#ifdef _CUDA_GPU_
		//int testConv() { return 1; }
#endif //_CUDA_GPU_
	};
	class Padding : public Layer 
#ifdef _CUDA_GPU_
		, PaddingGPU
#endif //_CUDA_GPU_
	{
	private:
		V4D* delta = nullptr;
	public:
		Padding(int paddingSize = 2) {
			this->layerType = LAYER::Padding;
			this->paddingSize = paddingSize;
		}
		virtual V4D getData4D() override {
			return this->data4D;
		}
		virtual void setData(V4D data4D) override {
			this->data4D = data4D;
		}
		virtual V4D* getDelta4D() override {
			return this->delta;
		}
		virtual void setDelta4D(V4D* delta) override {
			this->delta = delta;
		}
		V2D padding2D(V2D data, int paddingSize) {
			int numRows = (int)data.size();
			int paddedRows = 0;
			int paddedCols = 0;

			for (const auto& row : data) {
				int cols = (int)row.size();
				if (cols > paddedCols) {
					paddedCols = cols;
				}
			}
			paddedRows = numRows + 2 * paddingSize;
			paddedCols += 2 * paddingSize;

			V2D paddedData = V2D(paddedRows, V1D(paddedCols, 0.0));
			for (int i = 0; i < numRows; i++) {
				int currRowSize = (int)data[i].size();
				for (int j = 0; j < currRowSize; j++) {
					paddedData[i + paddingSize][j + paddingSize] = data[i][j];
				}
			}
			return paddedData;
		}
		virtual V4D calculate(V4D data) override {

			V4D actMap4D = V4D(data.size());
			int P = this->paddingSize * 2;
			V2D data2D;
			for (int i = 0; i < data.size(); i++) {
				for (int j = 0; j < data[i].size(); j++) {
					data2D = padding2D(data[i][j], this->paddingSize);
					actMap4D[i].push_back(data2D);
				}
			}

			return actMap4D;
		}
		virtual V4D* backward(V4D* delta) override {
			// 패딩만큼 제거해야됨

			V4D* calDelta = new V4D((*delta));
			for (int i = 0; i < (*delta).size(); i++) {
				for (int j = 0; j < (*delta)[i].size(); j++) {
					(*calDelta)[i][j].resize((*delta)[i][j].size() - (this->paddingSize * 2));
					for (int k = 0; k < (*delta)[i][j].size() - (this->paddingSize * 2); k++) {
						(*calDelta)[i][j][k].resize((*delta)[i][j][k].size() - (this->paddingSize * 2));
						for (int l = 0; l < (*delta)[i][j][k].size() - (this->paddingSize * 2); l++) {
							(*calDelta)[i][j][k][l] = (*delta)[i][j][k + this->paddingSize][l + this->paddingSize];
						}
					}
				}
			}

			return calDelta;
		}
#ifdef _CUDA_GPU_
		//int testConv() { return 1; }
#endif //_CUDA_GPU_
	};
	class Conv : public Layer 
#ifdef _CUDA_GPU_
		, ConvGPU
#endif //_CUDA_GPU_
	{
	private:
#ifdef _CUDA_GPU_
		//V3D* filter3D = nullptr;
		ActivationGPU* activationGPU = nullptr;
		unsigned int strideGPU;
		unsigned int filterRowColGPU;
#endif //_CUDA_GPU_

		V3D* filter3D = nullptr;

		int outputZ;
		int outputY;
		int outputX;
		V4D actMap4D;
		V4D saveData4DSize;
		V3D initFilter;
		V4D* delta = nullptr;
		
		Activation* activation = nullptr;
		RandomGen* randVal = new RandomGen(-0.1f, 0.1f);
		bool initialized = false;
	public:
		Conv(const int filterRowCol, const int filterSize, const int stride, const int paddingSize, ACTIVATION act, V3D filter) {
			this->layerType = LAYER::Conv;
			this->stride = stride;
			this->paddingSize = paddingSize;
			this->act = act;
			this->filterRowCol = filterRowCol;
			this->filterSize = filterSize;

			initFilter = filter;

			this->filter3D = &initFilter;
			initialized = true;
			this->activation = new Activation(act);
		}
		Conv(const int filterRowCol = 3, const int filterSize = 1, const int stride = 1, const int paddingSize = 1, ACTIVATION act = ACTIVATION::ReLU) {
			this->layerType = LAYER::Conv;
			this->act = act;

			this->filter3D = new V3D(filterSize, V2D(filterRowCol, V1D(filterRowCol, 0.0f)));
			this->randVal->randGen(*this->filter3D, initialized);
			initialized = true;
#ifdef _CUDA_GPU_



			this->setFilter3DGPU(this->filter3D);
			this->setStrideGPU(stride);
			this->setFilterRowColGPU(filterRowCol);

			//this->activationGPU = new ActivationGPU(act);
#else
			
			this->stride = stride;
			this->filterRowCol = filterRowCol;
			this->filterSize = filterSize;
			this->paddingSize = paddingSize;

			this->activation = new Activation(act);
#endif// _CUDA_GPU_
		}
		~Conv() {
			for (auto& d2d : *filter3D) {
				for (auto& d1d : d2d) {
					d1d.clear();
				}
				d2d.clear();
			}
			filter3D->clear();
			delete filter3D;
			delete this->activation;
		}
#ifdef _CUDA_GPU_
		virtual void setFilter3DGPU(V3D* filter) override {
			ConvGPU::setFilter3DGPU(filter);
		}

#endif//
		virtual void setFilter3D(V3D* filter) override {
			this->filter3D = filter;
		}
		virtual V3D* getFilter3D() override {
			return this->filter3D;
		}
		virtual V4D getData4D() override {
			return this->data4D;
		}
		virtual void setData(V4D data4D) override {
			this->data4D = data4D;
		}
		virtual void setData(V3D data3D) override {
			this->data3D = data3D;
		}
		virtual void setData(V2D data2D) override {
			this->data2D = data2D;
		}
		virtual void setData(V1D data1D) override {
			this->data1D = data1D;
		}
		virtual V4D* getDelta4D() override {
			return this->delta;
		}
		virtual void setDelta4D(V4D* delta) override {
			this->delta = delta;
		}
		virtual V4D calculate(V4D data) override {
			V2D image2D = im2col(data, this->filterRowCol, (*this->filter3D).size(), this->stride);
			V2D filter2D = filter2col((*this->filter3D), this->stride);
			V2D imgXfilter = imageXfilter(image2D, filter2D);
			V4D result = col2im(imgXfilter, data.size(), data[0].size(), data[0][0].size(), data[0][0][0].size(), this->filterRowCol, (*this->filter3D).size(), this->stride);
			return result;
		}
		/* // 원래 코드
		virtual V4D calculate(V4D data) override {
			V4D actMap4D = V4D(data.size());

			
			for (int i = 0; i < data.size(); i++) {
				actMap4D[i].resize(data[i].size() * (*filter3D).size());
				for (int j = 0; j < data[i].size(); j++)
					for (int k = 0; k < (*filter3D).size(); k++)
						for (int l = 0; l < (data[i][j].size() - (*filter3D)[0][0].size()) / stride + 1; l++)
							actMap4D[i][j * (*filter3D).size() + k].push_back(vector<float>((data[i][j][l].size() - (*filter3D)[0][0].size()) / stride + 1));
			}
			for (int i = 0; i < data.size(); i++) {
				for (int m = 0; m < data[i].size(); m++) {
					for (int j = 0; j < (*filter3D).size(); j++) {
						for (int k = 0; k < (data[i][m].size() - (*filter3D)[0].size()) / stride + 1; k++) {
							for (int l = 0; l < (data[i][m][k].size() - (*filter3D)[0][0].size()) / stride + 1; l++) {
								float sum = 0.0;
								for (int u = 0; u < (*filter3D)[0].size(); u++) {
									for (int v = 0; v < (*filter3D)[0][0].size(); v++) {
										sum += data[i][m][k * stride + u][l * stride + v] * (*filter3D)[j][u][v];
									}
								}
								actMap4D[i][m * filter3D->size() + j][k][l] = sum;
							}
						}
					}
				}
			}

			

			actMap4D = (*this->activation).calculate(actMap4D);
			return actMap4D;
		}*/
		virtual V4D* backward(V4D* delta) override {
			std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
			V4D saveDelta = saveData4DSize;

			//필터 역전
			reverseFilter(this->filter3D);

			std::chrono::duration<double>sec = std::chrono::system_clock::now() - start;
			std::cout << "필터 역전 : " << sec.count() << "seconds" << "\n";

			/////////////////////////////////////////////
			std::chrono::system_clock::time_point start2 = std::chrono::system_clock::now();
			Padding* padding = new Padding(this->paddingSize*2);
			V4D temp = padding->calculate(*delta);

			std::chrono::duration<double>sec2 = std::chrono::system_clock::now() - start2;
			std::cout << "연산 : " << sec2.count() << "seconds" << "\n"; //여기가 가장 느림.
			delete padding;
			V4D calDelta = this->calculate(temp);

			//std::chrono::duration<double>sec2 = std::chrono::system_clock::now() - start2;
			//std::cout << "연산 : " << sec2.count() << "seconds" << "\n"; //여기가 가장 느림.
			/////////////////////////////////////////////

			std::chrono::system_clock::time_point start3 = std::chrono::system_clock::now();

			V4D* result = new V4D(saveDelta);
			for (int i = 0; i < saveDelta.size(); i++) 
				for (int j = 0; j < saveDelta[i].size(); j++) 
					for (int m = 0; m < calDelta[i].size(); m++) 
						for (int k = 0; k < saveDelta[i][j].size(); k++) 
							for (int l = 0; l < saveDelta[i][j][k].size(); l++) 
								(*result)[i][j][k][l] += calDelta[i][m][k][l];

			std::chrono::duration<double>sec3 = std::chrono::system_clock::now() - start3;
			std::cout << "result : " << sec3.count() << "seconds" << "\n";

			std::chrono::system_clock::time_point start4 = std::chrono::system_clock::now();
			//다시 필터 원래대로
			reverseFilter(this->filter3D);
			std::chrono::duration<double>sec4 = std::chrono::system_clock::now() - start4;
			std::cout << "필터 재역전 : " << sec4.count() << "seconds" << "\n";
			return result;
		}
		void reverseFilter(V3D* filter) {
			V3D reverse = *filter;

			for (int i = 0; i < (*filter).size(); i++) 
				for (int j = 0; j < (*filter)[i].size(); j++) 
					for (int k = 0; k < (*filter)[i][j].size(); k++) 
						reverse[i][(*filter)[i].size() - j - 1][(*filter)[i][j].size() - k - 1] = (*filter)[i][j][k];
			
			for (int i = 0; i < (*filter).size(); i++) 
				for (int j = 0; j < (*filter)[i].size(); j++) 
					for (int k = 0; k < (*filter)[i][j].size(); k++) 
						(*filter)[i][j][k] = reverse[i][j][k];
			
			
		}
		V2D im2col(const V4D& image, int kernelRowCol, int kernelSize, int stride) {
			int imageSize = image.size();
			int channel = image[0].size();
			int height = image[0][0].size();
			int width = image[0][0][0].size();

			int output_height = (height - kernelRowCol) / stride + 1;
			int output_width = (width - kernelRowCol) / stride + 1;

			V2D col;
			for (int i = 0; i < kernelSize; i++) {
				for (int f = 0; f < imageSize; f++) {
					for (int d = 0; d < channel; ++d) {
						V1D col_entry;
						for (int h = 0; h < output_height; ++h) {
							for (int w = 0; w < output_width; ++w) {
								for (int i = h * stride; i < h * stride + kernelRowCol; ++i) {
									for (int j = w * stride; j < w * stride + kernelRowCol; ++j) {
										col_entry.push_back(image[f][d][i][j]);
									}
								}
							}

						}
						col.push_back(col_entry);
					}

				}
			}
			return col;
		}
		V2D filter2col(const V3D& filter, int stride) {
			V2D col;

			for (int i = 0; i < filter.size(); i++) {
				V1D col1D;
				for (int j = 0; j < filter[i].size(); j++) {
					for (int k = 0; k < filter[i][j].size(); k++) {
						col1D.push_back(filter[i][j][k]);
					}
				}
				col.push_back(col1D);
			}
			return col;
		}

		V4D col2im(const V2D& col, int imageSize, int channel, int height, int width, int kernelRowCol, int kernelSize, int stride) {
			int outputH = (height - kernelRowCol) / stride + 1;
			int outputW = (width - kernelRowCol) / stride + 1;

			// 48 9604
			// (2 X [3 X 8]) (98 X 98)
			V4D image = V4D(imageSize, V3D(channel * kernelSize, V2D(outputH, V1D(outputW, 0.0f))));

			for (int i = 0; i < imageSize; i++) { // 이미지 크기는 고정
				for (int j = 0; j < kernelSize * channel; j++) { // 커널 크기는 kernelSize * channel 곱셈
					for (int k = 0; k < outputH; k++) {
						for (int l = 0; l < outputW; l++) {
							image[i][j][k][l] = col[i * (kernelSize * channel) + j][k * outputW + l];
						}
					}
				}
			}

			return image;
		}
		V2D imageXfilter(V2D image, V2D filter) {
			V2D result = V2D(image.size(), V1D(image[0].size() / filter[0].size(), 0.0f));
			for (int i = 0; i < image.size(); i++) {
				for (int j = 0; j < image[i].size(); j++) {
					result[i][(j / filter[0].size()) % filter[0].size()] += image[i][j] * filter[i % filter.size()][j % filter[i % filter.size()].size()];
				}
			}
			return result;
		}


#ifdef _CUDA_GPU_
		virtual V4D calculateGPU(V4D data) override {
			return ConvGPU::calculateGPU(data);
		}

#endif //_CUDA_GPU_
	};

	class Pooling : public Layer 
#ifdef _CUDA_GPU_
		, PoolingGPU
#endif //_CUDA_GPU_
	{
	private:
		
		V4D pool;
		V4D maxPoolCoord;
		V4D minPoolCoord;
		V4D avgPoolCoord;
		V4D* delta = nullptr;
	public:
		Pooling(POOLING poolingName = POOLING::Max) {
			this->layerType = LAYER::Pool;
			this->poolingName = poolingName;
		}
		~Pooling() {
			deleteV4D(delta);
		}
		virtual V4D getData4D() override {
			return this->data4D;
		}
		virtual void setData(V4D data4D) override {
			this->data4D = data4D;
		}
		virtual V4D* getDelta4D() override {
			return this->delta;
		}
		virtual void setDelta4D(V4D* delta) override {
			this->delta = delta;
		}
		virtual V4D calculate(V4D data) override {
			maxPoolCoord = data;
			for (int i = 0; i < data.size(); i++) 
				for (int j = 0; j < data[i].size(); j++) 
					for (int k = 0; k < data[i][j].size(); k++) 
						for (int l = 0; l < data[i][j][k].size(); l++) 
							maxPoolCoord[i][j][k][l] = 0.0f;

			minPoolCoord = maxPoolCoord;
			avgPoolCoord = maxPoolCoord;

			V4D actMap4D = V4D(data);
			V4D maxActMap4D = actMap4D;
			V4D minActMap4D = actMap4D;
			V4D avgActMap4D = actMap4D;
			
			for (int i = 0; i < data.size(); i++) {
				for (int j = 0; j < data[i].size(); j++) {
					actMap4D[i][j].resize(data[i][j].size() / 2);
					maxActMap4D[i][j].resize(data[i][j].size() / 2);
					minActMap4D[i][j].resize(data[i][j].size() / 2);
					avgActMap4D[i][j].resize(data[i][j].size() / 2);
					for (int k = 0; k < data[i][j].size() / 2; k++) {
						actMap4D[i][j][k].resize(data[i][j][k].size() / 2);
						maxActMap4D[i][j][k].resize(data[i][j][k].size() / 2);
						minActMap4D[i][j][k].resize(data[i][j][k].size() / 2);
						avgActMap4D[i][j][k].resize(data[i][j][k].size() / 2);
						for (int l = 0; l < data[i][j][k].size() / 2; l++) {
							float maxValue = max(max(max(data[i][j][2 * k][2 * l], data[i][j][2 * k][2 * l + 1]), data[i][j][2 * k + 1][2 * l]), data[i][j][2 * k + 1][2 * l + 1]);
							float minValue = min(min(min(data[i][j][2 * k][2 * l], data[i][j][2 * k][2 * l + 1]), data[i][j][2 * k + 1][2 * l]), data[i][j][2 * k + 1][2 * l + 1]);
							maxActMap4D[i][j][k][l] = maxValue;
							minActMap4D[i][j][k][l] = minValue;
							avgActMap4D[i][j][k][l] = (data[i][j][2 * k][2 * l] + data[i][j][2 * k][2 * l + 1] + data[i][j][2 * k + 1][2 * l] + data[i][j][2 * k + 1][2 * l + 1]) / 4.0f;
							if (maxValue == data[i][j][2 * k][2 * l]) this->maxPoolCoord[i][j][2 * k][2 * l] = 1.0f;
							else if (maxValue == data[i][j][2 * k][2 * l + 1]) this->maxPoolCoord[i][j][2 * k][2 * l + 1] = 1.0f;
							else if (maxValue == data[i][j][2 * k + 1][2 * l]) this->maxPoolCoord[i][j][2 * k + 1][2 * l] = 1.0f;
							else if (maxValue == data[i][j][2 * k + 1][2 * l + 1]) this->maxPoolCoord[i][j][2 * k + 1][2 * l + 1] = 1.0f;
							if (minValue == data[i][j][2 * k][2 * l]) this->minPoolCoord[i][j][2 * k][2 * l] = 1.0f;
							else if (minValue == data[i][j][2 * k][2 * l + 1]) this->minPoolCoord[i][j][2 * k][2 * l + 1] = 1.0f;
							else if (minValue == data[i][j][2 * k + 1][2 * l]) this->minPoolCoord[i][j][2 * k + 1][2 * l] = 1.0f;
							else if (minValue == data[i][j][2 * k + 1][2 * l + 1]) this->minPoolCoord[i][j][2 * k + 1][2 * l + 1] = 1.0f;

						}
					}

				}

			}
			
			if (poolingName == POOLING::Max) 
				actMap4D = maxActMap4D;
			else if (poolingName == POOLING::Min) 
				actMap4D = minActMap4D;
			else if (poolingName == POOLING::Average) 
				actMap4D = avgActMap4D;
			else 
				cout << "Pooling Name 에러";

			return actMap4D;

		}
		virtual V4D* backward(V4D* delta) override {

			V4D poolCoord;
			if (this->poolingName == POOLING::Max) {
				pool = expandDelta(delta, this->maxPoolCoord);
				poolCoord = this->maxPoolCoord;
			}
			else if (this->poolingName == POOLING::Min) {
				pool = expandDelta(delta, this->minPoolCoord);
				poolCoord = this->minPoolCoord;
			}
			else if (this->poolingName == POOLING::Average) {
				pool = expandDelta(delta, this->avgPoolCoord);
				V4D* copyPool = new V4D(pool);
				return copyPool;
			}
			for (int i = 0; i < pool.size(); i++) {
				for (int j = 0; j < pool[i].size(); j++) {
					for (int k = 0; k < pool[i][j].size(); k++) {
						for (int l = 0; l < pool[i][j][k].size(); l++) {
							pool[i][j][k][l] *= poolCoord[i][j][k][l];
						}
					}
				}
			}
			V4D* copyPool = new V4D(pool);
			
			return copyPool;
		}
		V4D expandDelta(V4D* delta, V4D poolCoord) {
			V4D output = poolCoord;
			
			for (int i = 0; i < (*delta).size(); i++) {
				for (int j = 0; j < (*delta)[i].size(); j++) {
					for (int k = 0; k < (*delta)[i][j].size(); k++) {
						for (int l = 0; l < (*delta)[i][j][k].size(); l++) {
							float value = (*delta)[i][j][k][l];
							output[i][j][k * 2][l * 2] = value;
							output[i][j][k * 2][l * 2 + 1] = value;
							output[i][j][k * 2 + 1][l * 2] = value;
							output[i][j][k * 2 + 1][l * 2 + 1] = value;
						}
					}
				}
			}

			return output;
		}
#ifdef _CUDA_GPU_
		
#endif //_CUDA_GPU_
	};

	// 완성
	class Flatten : public Layer 
#ifdef _CUDA_GPU_
		, FlattenGPU
#endif //_CUDA_GPU_
	{
	private:
		V4D savedFlattenSize;
		V4D* delta = nullptr;
	public:
		Flatten() {
			this->layerType = LAYER::Flatten;
		}
		virtual V2D getData2D() override {
			return this->data2D;
		}
		virtual V4D getData4D() override {
			return this->data4D;
		}
		virtual V4D* getDelta4D() override {
			return this->delta;
		}
		virtual void setDelta4D(V4D* delta) override {
			this->delta = delta;
		}
		virtual void setData(V2D data2D) override {
			this->data2D = data2D;
		}
		virtual void setData(V4D data4D) override {
			this->data4D = data4D;
		}
		V2D calculate(V4D data, LAYER layer) override {
			V2D flat = V2D(0);
			V1D flat1D;
			for (int i = 0; i < data.size(); ++i) {
				for (int j = 0; j < data[i].size(); ++j)
					for (int k = 0; k < data[i][j].size(); ++k)
						for (int l = 0; l < data[i][j][k].size(); ++l)
							flat1D.push_back(data[i][j][k][l]);
				flat.push_back(flat1D);
				flat1D.clear();
			}
			return flat;
		}
		

#ifdef _CUDA_GPU_

#endif //_CUDA_GPU_
	};

	
	class FullyConnected : public Layer 
#ifdef _CUDA_GPU_
		,FullyConnectedGPU
#endif
	{
	private:
		vector<V2D> saveWeight1D;
		
		V2D* weight2D = nullptr;
		V3D* weight3D = new V3D();
		V3D initWeight;
		V1D* result = nullptr;
		RandomGen* randomGen = new RandomGen(-1.0f, 1.0f);
		bool initialized = false;
		Activation* activation = nullptr;

		V2D* deltaFC = nullptr;
	public:
		virtual ACTIVATION getActivationType() override {
			return this->act;
		}
		FullyConnected() {
			this->activation = new Activation();
			this->act = ACTIVATION::Sigmoid;
		}
		FullyConnected(int denseSize, ACTIVATION act, V3D weight) {
			this->layerType = LAYER::FC;
			this->act = act;
			this->denseSize = denseSize;
			this->activation = new Activation(act);
			initWeight = weight;
			initialized = true;
			this->weight3D = &initWeight;
		}
		FullyConnected(int denseSize, ACTIVATION act) {
			this->layerType = LAYER::FC;
			this->act = act;
			this->denseSize = denseSize;
			this->activation = new Activation(act);
		}
		~FullyConnected() {
			deleteV3D(weight3D);
			delete this->randomGen;
		}
		virtual V2D getData2D() override {
			return this->data2D;
		}
		virtual void setData(V2D data) override {
			this->data2D = data;
		}
		virtual void setWeight3D(V3D* weight) override {
			this->weight3D = weight;
		}
		virtual V3D* getWeight3D() override {
			return this->weight3D;
		}

		virtual void setDelta2D(V2D* delta2D) override {
			this->deltaFC = delta2D;
		}
		virtual V2D* getDelta2D() override {
			return this->deltaFC;
		}
		V2D calculate(V2D data) override {
			V2D fully = V2D(data.size());

			//data크기: 2 128
			//output크기: 2 4
			//-->weight크기: 2 128 4
			(*this->weight3D).resize(data.size());
			for (int i = 0; i < data.size(); i++) {
				(*this->weight3D)[i].resize(data[i].size());
				for (int j = 0; j < data[i].size(); j++) {
					(*this->weight3D)[i][j].resize(this->denseSize);
				}
			}
			randomGen->randGen(*this->weight3D, initialized);
			initialized = true;

			for (int i = 0; i < data.size(); i++) {
				for (int j = 0; j < (*this->weight3D)[i][0].size(); j++) {
					float sum = 0.0;
					for (int k = 0; k < data[i].size(); k++) {
						sum += data[i][k] * (*this->weight3D)[i][k][j];
					}
					fully[i].push_back(sum);
				}
			}
			fully = (*this->activation).calculate2D(fully);
			
			return fully;
		}
#ifdef _CUDA_GPU_

#endif //_CUDA_GPU_
	}; 
	class Optimizer {
	private:
		OPTIMIZER opt;
	public:
		Optimizer(){
			opt = OPTIMIZER::Mini_SGD;
		}
		Optimizer(OPTIMIZER opt) {
			this->opt = opt;
		}
		float miniSGD(const float learningRate, float weight, float gradient) {
			weight -= learningRate * gradient;
			return weight;
		}
		float momentum(const float learningRate, float weight, float gradient, const float beta = 0.9f) {
			float velocity = 0.0f;
			velocity = beta * velocity + (1.0f - beta) * gradient;
			weight -= learningRate * velocity;
			return weight;
		}
		float adagrad(const float learningRate, float weight, float gradient) {
			float epsilon = 1e-7;
			float squaredGradient = gradient * gradient;
			weight -= (learningRate / (sqrt(squaredGradient) + epsilon)) * gradient;
			return weight;
		}
		float rmsProp(const float learningRate, float weight, float gradient, const float beta = 0.9f){
			float epsilon = 1e-7;
			float squaredGradient = beta * squaredGradient + (1.0f - beta) * (gradient * gradient);
			weight -= (learningRate / (sqrt(squaredGradient) + epsilon)) * gradient;
			return weight;
		}
		float adam(const float learningRate, float weight, float gradient, float& m, float& v, int epoch, const float beta1 = 0.9f, const float beta2 = 0.999f){
			float epsilon = 1e-7;
			
			m = beta1 * m + (1.0f - beta1) * gradient;
			v = beta2 * v + (1.0f - beta2) * (gradient * gradient);

			float m_hat = m / (1.0f - pow(beta1, epoch));
			float v_hat = v / (1.0f - pow(beta2, epoch));
			weight -= (learningRate / (sqrt(v_hat) + epsilon)) * m_hat;
			return weight;
		}
		float nadam(const float learningRate, float weight, float gradient, float& m, float& v, int epoch, const float beta1 = 0.9f, const float beta2 = 0.999f) {
			const float epsilon = 1e-7;

			m = beta1 * m + (1.0f - beta1) * gradient;
			v = beta2 * v + (1.0f - beta2) * (gradient * gradient);
			
			float m_hat = m / (1.0f - pow(beta1, epoch));
			float v_hat = v / (1.0f - pow(beta2, epoch));
			weight -= (learningRate / (sqrt(v_hat) + epsilon)) * (m_hat + beta1 * m_hat);
			return weight;
		}
	};
	class CNN {
	private:
		V5D imageSet;
		V3D imageSet3D;
		V4D image;
		V3D filter;
		V1D label1D;
		V2D label2D;
		V2D labelSet2D;
		V3D labelSet3D;

		V1D target1D;
		V2D target2D;

		V4D trainImage;
		V1D trainLabel;
		V4D testImage;
		V1D testLabel;
		V2D onehotLabel;
		V2D target;
		V2D output;


		vector<Layer*> layers;


		float learningRate;

		unsigned int epochsCount = 10;
		int batchSize;
		Optimizer* optimizer = new Optimizer();
		OPTIMIZER optimizerType;
		LOSS loss;
		void setLayer() {

		}
	public:
		V4D resize(V4D image, float x, float y) {
			// 양선형 보간법
			//int x1 = int(x);
			//int x2 = x1 + 1;
			//int y1 = int(y);
			//int y2 = y1 + 1;
			//
			//float dx = x - x1;
			//float dy = y - y1;
			//
			//int Q11 = image.getPixel(x1, y1);
			//int Q21 = image.getPixel(x2, y1);
			//int Q12 = image.getPixel(x1, y2);
			//int Q22 = image.getPixel(x2, y2);
			//
			//int result = int((1 - dx) * (1 - dy) * Q11 +
			//	dx * (1 - dy) * Q21 +
			//	(1 - dx) * dy * Q12 +
			//	dx * dy * Q22);
			//
			return image;

		}
		V4D zeroFillToFit(V4D image) {
			unsigned int maxXY = 0, maxC = 0;
			for (int i = 0; i < image.size(); i++) {
				for (int j = 0; j < image[i].size(); j++) {
					for (int k = 0; k < image[i][j].size(); k++) {
						if (maxXY < image[i][j][k].size()) maxXY = image[i][j][k].size();
					}
					if (maxXY < image[i][j].size()) maxXY = image[i][j].size();
				}
				if (maxC < image[i].size()) maxC = image[i].size();
			}
			V4D image4D = V4D(image.size(), V3D(maxC, V2D(maxXY, V1D(maxXY, 0.0f))));

			for (int i = 0; i < image.size(); i++) {
				for (int j = 0; j < image[i].size(); j++) {
					for (int k = 0; k < image[i][j].size(); k++) {
						for (int l = 0; l < image[i][j][k].size(); l++) {
							image4D[i][j][k][l] = image[i][j][k][l];
						}
					}
				}
			}
			
			return image4D;
		}
	private:
		V5D& batchImage(V4D& image, unsigned int batchSize) {
			int numBatches = (int)image.size() / batchSize;

			for (int i = 0; i < numBatches; i++) {
				V4D image4D;
				for (int j = i * batchSize; j < min((i + 1) * batchSize, (int)image.size()); j++) {
					image4D.push_back(image[j]);
				}
				this->imageSet.push_back(image4D);
			}

			return this->imageSet;
		}
		V3D& batchImage2D(V4D& image, unsigned int batchSize) {
			int numBatches = (int)image.size() / batchSize;

			for (int i = 0; i < numBatches; i++) {
				V2D image2D;
				for (int j = i * batchSize; j < min((i + 1) * batchSize, (int)image.size()); j++) {
					V1D image1D;
					for (int k = 0; k < image[j].size(); k++) {
						for (int l = 0; l < image[j][k].size(); l++) {
							for (int m = 0; m <image[j][k][l].size(); m++) {
								image1D.push_back(image[j][k][l][m]);
							}
						}
					}
					image2D.push_back(image1D);
					image1D.clear();
				}
				this->imageSet3D.push_back(image2D);
				image2D.clear();
			}

			return this->imageSet3D;
		}
		V2D& batchLabel(V1D& label, int batchSize) {

			int numBatches = (int)label.size() / batchSize;

			for (int i = 0; i < numBatches; i++) {
				V1D label1D;
				for (int j = i * batchSize; j < (i + 1) * batchSize; j++) {
					label1D.push_back(label[j]);
				}
				this->labelSet2D.push_back(label1D);
			}

			return this->labelSet2D;
		}
	public:
		CNN() {

		}
		CNN(V4D& image, V1D label) {
			this->image = image;
			this->label1D = label;


			cout << "\nimage: " << image.size() << " label: " << label.size() << "\n";
		}
		CNN(V4D& image, V1D label, V4D& testImage, V1D testLabel) {
			this->image = image;
			this->label1D = label;
			this->testImage = testImage;
			this->testLabel = testLabel;
			
		}
		~CNN() {}
		// split train test 담을 배열 정하기. V4D
		
		//void splitTrainTest(double rate = 0.3) {
		//	int cnt = 0;
		//	for (int i = 0; i < this->image.size(); i++) {
		//		if (static_cast<int>(static_cast<double>(this->image.size()) * rate) > cnt++) this->testImage.push_back(this->image.at(i));
		//		else this->trainImage.push_back(this->image.at(i));
		//	}
		//	cnt = 0;
		//	for (int i = 0; i < static_cast<int>(static_cast<double>(this->label.size()) * rate); i++) {
		//		if (static_cast<int>(static_cast<double>(this->label.size()) * rate) > cnt++) this->testLabel.push_back(this->label.at(i));
		//		else this->trainLabel.push_back(this->Label.at(i));
		//	}
		//}
			private:
				std::string actTypeToString(ACTIVATION act) {
					std::string actString;
					switch (act) {
					case ACTIVATION::TanH:
						actString = "TanH";
						break;
					case ACTIVATION::Softmax:
						actString = "Softmax";
						break;
					case ACTIVATION::Sigmoid:
						actString = "Sigmoid";
						break;
					case ACTIVATION::ReLU:
						actString = "ReLU";
						break;
					case ACTIVATION::None:
						actString = "None";
						break;
					case ACTIVATION::Maxout:
						actString = "Maxout";
						break;
					case ACTIVATION::Leaky_ReLU:
						actString = "Leaky_ReLU";
						break;
					case ACTIVATION::ELU:
						actString = "ELU";
						break;
					default:
						actString = "Error";
						break;
					}
					return actString;
				}
				ACTIVATION stringToActType(std::string type) {
					ACTIVATION act;
					if (type == "TanH")
						act = ACTIVATION::TanH;
					else if (type == "Softmax")
						act = ACTIVATION::Softmax;
					else if (type == "Sigmoid")
						act = ACTIVATION::Sigmoid;
					else if (type == "ReLU")
						act = ACTIVATION::ReLU;
					else if (type == "None")
						act = ACTIVATION::None;
					else if (type == "Maxout")
						act = ACTIVATION::Maxout;
					else if (type == "Leaky_ReLU")
						act = ACTIVATION::Leaky_ReLU;
					else if (type == "ELU")
						act = ACTIVATION::ELU;
					else
						act = ACTIVATION::Error;

					return act;
				}

	public:
		void save(const std::string& filename) {
			std::ofstream file(filename);

			if (!file.is_open()) {
				std::cerr << "파일 열기 실패: " << filename << std::endl;
				return;
			}

			for (const auto& layer : layers) {

				if (layer->getLayerType() == LAYER::Conv || layer->getLayerType() == LAYER::FC) {
					V3D* filter3D = nullptr;
					if (layer->getLayerType() == LAYER::Conv) {
						file << "Conv\n";
						file << layer->filterRowCol << " " << layer->filterSize << " " << layer->stride << " " << layer->paddingSize << "\n";
						std::string act = actTypeToString(layer->act);
						file << act << "\n";

						//file << 
						filter3D = layer->getFilter3D();

					}
					else {
						file << "FC\n";


						std::string act = actTypeToString(layer->act);
						file << layer->denseSize << " " << act << "\n";

						filter3D = layer->getWeight3D();
						//file << (*filter3D).size() << " " << (*filter3D)[0].size() << "\n";
					}

					for (int i = 0; i < (*filter3D).size(); i++) {
						for (int j = 0; j < (*filter3D)[i].size(); j++) {
							for (int k = 0; k < (*filter3D)[i][j].size(); k++) {
								file << (*filter3D)[i][j][k] << ' ';
							}
							file << "\n";
						}
						file << "\n";
					}
					file << "End\n";
				}
				else if (layer->getLayerType() == LAYER::Padding) {
					file << "Padding\n";
					file << layer->paddingSize << "\n";
				}
				else if (layer->getLayerType() == LAYER::Pool) {
					file << "Pool\n";
					if (layer->poolingName == POOLING::Max) file << "Max\n";
					else if (layer->poolingName == POOLING::Min) file << "Min\n";
					else if (layer->poolingName == POOLING::Average) file << "Average\n";
					else { file << "Error\n"; }
				}
				else if (layer->getLayerType() == LAYER::Flatten) {
					file << "Flatten\n";
				}
				else {
					file << "Error\n";
				}
			}
			file.close();
			cout << "파일 쓰기 성공\n";
		}

		void load(const std::string& filename) {
			std::ifstream file(filename);

			if (!file.is_open()) {
				std::cerr << "Unable to open file: " << filename << std::endl;
				return;
			}

			layers.clear();

			std::string line, prevLine = "";
			int i = 0;
			while (std::getline(file, line)) {

				if (line == "Conv") {
					std::string convSetting;
					std::getline(file, convSetting);
					std::istringstream convStream(convSetting);
					std::vector<int> convSettingRow;
					int value;
					while (convStream >> value) {
						convSettingRow.push_back(value);
					}
					std::string convAct;
					std::getline(file, convAct);

					ACTIVATION act = stringToActType(convAct);

					std::string weightsLine;
					V3D weight3D;
					V2D weight2D;
					while (weightsLine != "End") {
						std::getline(file, weightsLine);

						std::istringstream weightsStream(weightsLine);
						std::vector<float> weightRow;
						float value;
						while (weightsStream >> value) {
							weightRow.push_back(value);
						}

						if (!weightsLine.empty()) {
							weight2D.push_back(weightRow);
							weightRow.clear();
						}
						else {
							weight3D.push_back(weight2D);
							weight2D.clear();
						}
					}

					layers.push_back(new Conv(convSettingRow[0],
						convSettingRow[1],
						convSettingRow[2],
						convSettingRow[3],
						act,
						weight3D)
					);
				}
				else if (line == "FC") {

					std::string fullySetting;
					std::getline(file, fullySetting);
					//fullySetting
					std::istringstream fullyStream(fullySetting);
					int dense;
					std::string value;
					fullyStream >> dense;
					fullyStream >> value;

					std::string weightsLine;
					V3D weight3D;
					V2D weight2D;
					while (weightsLine != "End") {
						std::getline(file, weightsLine);

						std::istringstream weightsStream(weightsLine);
						std::vector<float> weightRow;
						float value;
						while (weightsStream >> value) {
							weightRow.push_back(value);
						}

						if (!weightsLine.empty()) {
							weight2D.push_back(weightRow);
							weightRow.clear();
						}
						else {
							weight3D.push_back(weight2D);
							weight2D.clear();
						}
					}
					layers.push_back(new FullyConnected(dense, stringToActType(value), weight3D));
				}
				else if (line == "Pool") {
					std::string pool;
					std::getline(file, pool);
					if (pool == "Max")
						layers.push_back(new Pooling(POOLING::Max));
					else if (pool == "Min")
						layers.push_back(new Pooling(POOLING::Min));
					else if (pool == "Average")
						layers.push_back(new Pooling(POOLING::Average));
					else {
						std::cout << "error";
						return;
					}
				}
				else if (line == "Padding") {
					std::string padding;
					std::getline(file, padding);
					int value;
					std::istringstream paddingStream(padding);
					paddingStream >> value;

					layers.push_back(new Padding(value));
				}
				else if (line == "Flatten") {
					layers.push_back(new Flatten());
				}
				else {
					std::cout << "error";
					return;
				}
			}

			file.close();
			std::cout << "Model loaded from: " << filename << std::endl;
		}
		void normalization(V4D& image) {
			for (int i = 0; i < image.size(); ++i)
				for (int j = 0; j < image[i].size(); ++j)
					for (int k = 0; k < image[i][j].size(); ++k)
						for (int l = 0; l < image[i][j][k].size(); ++l)
							image[i][j][k][l] /= 255.0f;
		}
		V2D oneHotEncoding(V1D label,int outputSize) {
			// 로그찍어서 확인
			V2D oneHotLabel = V2D(label.size(), V1D(outputSize, 0.0f));
			for (int i = 0; i < label.size(); ++i) {
				oneHotLabel[i][(const int)label[i]] = 1.0f;
			}
			return oneHotLabel;
		}
		void setPrintEpochsCount(int epochs) {
			this->epochsCount = epochs;
		}
		void add(Layer* layer) {
			this->layers.push_back(layer);
		}

		void compile(OPTIMIZER optimizerType, LOSS loss) {
			this->optimizerType = optimizerType;
			this->loss = loss;
			this->learningRate = 0.0001f;
		}
		void fit(int epochs, int batchSize) {
			if (batchSize <= 0) batchSize = 1;

			this->imageSet = this->batchImage(this->image, batchSize);
			this->labelSet2D = this->batchLabel(this->label1D, batchSize);
			
			for (int i = 0; i < epochs; i++) {
				std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
				cout << "\nepochs:" << i+1 << "\n";

				for (int j = 0; j < this->imageSet.size(); ++j) {		

					this->layers[0]->setData(this->imageSet[j]);

					this->target1D = this->labelSet2D[j];
					this->feedforward(batchSize);
					this->backward(batchSize);

				}
				std::chrono::duration<double>sec = std::chrono::system_clock::now() - start;
				std::cout << "time : " << sec.count() << "seconds" << "\n";
			}
		}
		
		V1D predict(V3D image) {
			V4D image4D;
			image4D.push_back(image);
			this->layers[0]->setData(image4D);
			this->feedforward(1);

			std::cout << "predict: ";
			for (int i = 0; i < output.size(); i++) {
				std::cout << output[0][i] <<" ";
			}
			std::cout << "\n";
			return output[0];
		}
		V1D predict(V4D image) {
			this->layers[0]->setData(image);
			this->feedforward(1);
			V1D prediction;
			for (int i = 0; i < output.size(); i++) {
				prediction.push_back(max_element(output[i].begin(), output[i].end()) - output[i].begin());
			}
			return prediction;
		}
		void accuracy(V1D predictions, V1D labels) {
			int correct_predictions = 0;

			for (size_t i = 0; i < predictions.size(); ++i) {
				if (predictions[i] == labels[i]) {
					correct_predictions++;
				}
			}

			float acc = static_cast<float>(correct_predictions) / predictions.size();
			std::cout << "Accuracy: " << acc * 100 << "%\n";

		}
		void feedforward(int batchSize) {
			
			if (this->layers.size() < 2) {
				cout << "레이어가 너무 작음.";
				return;
			}
			for (int i = 0; i < this->layers.size(); ++i) {
				std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
				const int LAST = (int)this->layers.size() - 1;

				if (this->layers[i]->getLayerType() == LAYER::Conv) {
					
#ifdef _CUDA_GPU_
					std::cout << " CONV GPU 여기";
					V4D data = this->layers[i]->getData4D();
	
					V4D result = this->layers[i]->calculateGPU(data);

					this->layers[(i + 1)]->setData(result);
#else
					V4D data = this->layers[i]->getData4D();
					this->layers[(i + 1)]->setData(this->layers[i]->calculate(data));
#endif					
				}
				else if (i == LAST && this->layers[i]->getLayerType() == LAYER::FC) {
					V2D data = this->layers[i]->getData2D();
#ifdef _CUDA_GPU_
					this->output = this->layers[i]->calculateGPU(data);
#else
					this->output = this->layers[i]->calculate(data);
#endif
				}
				else if (this->layers[i]->getLayerType() == LAYER::Act || this->layers[i]->getLayerType() == LAYER::Pool || this->layers[i]->getLayerType() == LAYER::Padding) {
#ifdef _CUDA_GPU_
					V2D data = this->layers[i]->getData2D();
					this->layers[i + 1]->setData(this->layers[i]->calculateGPU(data));
#else
					V4D data = this->layers[i]->getData4D();	
					this->layers[i + 1]->setData(this->layers[i]->calculate(data));
#endif
					

				}
				else if (this->layers[i]->getLayerType() == LAYER::Flatten) {
#ifdef _CUDA_GPU_
					V2D data = this->layers[i]->getData2D();
					V2D flat = this->layers[i]->calculateGPU(data);
#else

					V4D data = this->layers[i]->getData4D();
					V2D flat = this->layers[i]->calculate(data, LAYER::Flatten);
#endif
					this->layers[i + 1]->setData(flat);


				}
				else if (this->layers[i]->getLayerType() == LAYER::FC) {
#ifdef _CUDA_GPU_
					V2D data = this->layers[i]->getData2D();
					V2D cal = this->layers[i]->calculateGPU(data);
#else
					V2D data = this->layers[i]->getData2D();
					V2D cal = this->layers[i]->calculate(data);

#endif
					this->layers[i + 1]->setData(cal);
										
				}
				else {
					cout << "\n레이어:" << i << " 에러\n";
				}

				std::chrono::duration<double>sec = std::chrono::system_clock::now() - start;
				std::cout << "forward time : " << sec.count() << "seconds" << "\n";
			}
		}
		
		void backward(int batchSize) {
			
			for (int j = (int)this->layers.size() - 1; j >= 0; --j) {
				std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
				if (this->layers[j]->getLayerType() == LAYER::Conv) {
					std::cout << "Backward Conv\n";
					V4D* delta = this->layers[j + 1]->getDelta4D();
					this->layers[j]->setDelta4D(this->layers[j]->backward(delta));
				}
				else if (j == this->layers.size() - 1 && this->layers[j]->getLayerType() == LAYER::FC) {
					// 출력층 FC
					if (loss == LOSS::SparseCategoricalCrossentropy) 
						this->target = this->oneHotEncoding(this->target1D, (int)this->output[0].size());						
					else if (loss == LOSS::BinaryCrossentropy) 
						this->target = this->oneHotEncoding(this->target1D, 2);
					else 
						this->target = this->oneHotEncoding(this->target1D, (int)this->output[0].size());		
					
					float lossValue = this->lossFunc(&this->target, &this->output, this->loss);

					V2D* dLossdOutput = this->derivativeLoss(&this->output, &this->target, this->loss);
					V2D* dOutputdAct = this->derivativeAct(dLossdOutput, this->layers[j]->getActivationType());
					V2D* deltaFC = new V2D((*dLossdOutput).size(), V1D((*dLossdOutput)[0].size(), 0.0f));

					
					for (int i = 0; i < (*deltaFC).size(); i++) {
						for (int j = 0; j < (*deltaFC)[i].size(); j++) {
							(*deltaFC)[i][j] = (*dLossdOutput)[i][j] * (*dOutputdAct)[i][j];

						}
					}

					V2D data = this->layers[j]->getData2D();
	
					this->layers[j]->setWeight3D(updateWeightsFC(this->layers[j]->getWeight3D(), deltaFC, &data, this->optimizerType));
					this->layers[j]->setDelta2D(deltaFC);

				}
				else if (this->layers[j]->getLayerType() == LAYER::Padding) {

					V4D* delta = this->layers[j + 1]->getDelta4D();
					this->layers[j]->setDelta4D(this->layers[j]->backward(delta));
				}

				else if (this->layers[j]->getLayerType() == LAYER::Pool) {
					V4D* delta = this->layers[j + 1]->getDelta4D();
					this->layers[j]->setDelta4D(this->layers[j]->backward(delta));
				}

				else if (this->layers[j]->getLayerType() == LAYER::Flatten) {
					V4D data4D = this->layers[j]->getData4D();
					V2D* curDelta = deltaXweights(this->layers[j + 1]->getDelta2D(), this->layers[j + 1]->getWeight3D());
					
					V2D* delta2D = new V2D((*curDelta));
					for (int i = 0; i < (*delta2D).size(); i++) 
						for (int k = 0; k < (*delta2D)[i].size(); k++) 
							(*delta2D)[i][k] = (*curDelta)[i][k] * this->layers[j + 1]->getData2D()[i][k];
						
					
					V4D* delta4D = new V4D(data4D);
					for (int i = 0; i < data4D.size(); i++) {
						int idx = 0;
						for (int j = 0; j < data4D[i].size(); j++) 
							for (int k = 0; k < data4D[i][j].size(); k++) 
								for (int l = 0; l < data4D[i][j][k].size(); l++) 								
									(*delta4D)[i][j][k][l] = (*delta2D)[i][idx++];
					}
					
					this->layers[j]->setDelta4D(delta4D);
					
				}
				else if (this->layers[j]->getLayerType() == LAYER::FC) {

					V2D* delta2D = this->layers[j + 1]->getDelta2D();
					V3D* weight3D = this->layers[j + 1]->getWeight3D();
					V2D* deltaXw = deltaXweights(delta2D, weight3D);
					V2D* dOutputdAct = this->derivativeAct(deltaXw, this->layers[j]->getActivationType());
					V2D dataW = this->layers[j]->getData2D();
					V2D* dActdW = &dataW;

					V2D* deltaFC = new V2D((*dOutputdAct));
					for (int i = 0; i < (*deltaXw).size(); i++) {
						for (int j = 0; j < (*deltaXw)[i].size(); j++) {
							(*deltaFC)[i][j] = (*deltaXw)[i][j] * (*dOutputdAct)[i][j];
						}
					}

					this->layers[j]->setWeight3D(this->updateWeightsFC(this->layers[j]->getWeight3D(), deltaFC, dActdW, this->optimizerType));
					this->layers[j]->setDelta2D(deltaFC);
				}
				else {
					cout << "에러" << "\n";
				}
				std::chrono::duration<double>sec = std::chrono::system_clock::now() - start;
				std::cout << "backward time : " << sec.count() << "seconds" << "\n";
			}
		}
		V3D* updateWeightsFC(V3D* updateWeights, V2D* deltaFC,V2D* dActdW, OPTIMIZER opt) {
			
			clipDelta(deltaFC);

			if (opt == OPTIMIZER::Mini_SGD) {
				for (int i = 0; i < (*updateWeights).size(); i++)
					for (int j = 0; j < (*updateWeights)[i].size(); j++)
						for (int k = 0; k < (*updateWeights)[i][j].size(); k++)
							(*updateWeights)[i][j][k] = optimizer->miniSGD(this->learningRate,(*updateWeights)[i][j][k], (*deltaFC)[i][k] * (*dActdW)[i][j]);
							//(*updateWeights)[i][j][k] -= this->leaningRate * (*deltaFC)[i][k] * (*dActdW)[i][j];
			}
			else if (opt == OPTIMIZER::Momentum) {
				for (int i = 0; i < (*updateWeights).size(); i++)
					for (int j = 0; j < (*updateWeights)[i].size(); j++)
						for (int k = 0; k < (*updateWeights)[i][j].size(); k++)
							(*updateWeights)[i][j][k] = optimizer->momentum(this->learningRate, (*updateWeights)[i][j][k], ((*deltaFC)[i][k] * (*dActdW)[i][j]));
			}
			else if (opt == OPTIMIZER::Adagrad) {
				for (int i = 0; i < (*updateWeights).size(); i++)
					for (int j = 0; j < (*updateWeights)[i].size(); j++)
						for (int k = 0; k < (*updateWeights)[i][j].size(); k++)
							(*updateWeights)[i][j][k] = optimizer->adagrad(this->learningRate, (*updateWeights)[i][j][k], ((*deltaFC)[i][k] * (*dActdW)[i][j]));
			}
			else if (opt == OPTIMIZER::RMSProp) {
				for (int i = 0; i < (*updateWeights).size(); i++)
					for (int j = 0; j < (*updateWeights)[i].size(); j++)
						for (int k = 0; k < (*updateWeights)[i][j].size(); k++)
							(*updateWeights)[i][j][k] = optimizer->rmsProp(this->learningRate, (*updateWeights)[i][j][k], ((*deltaFC)[i][k] * (*dActdW)[i][j]));
			}
			else if (opt == OPTIMIZER::Adam) {
				float v, m;
				for (int i = 0; i < (*updateWeights).size(); i++)
					for (int j = 0; j < (*updateWeights)[i].size(); j++)
						for (int k = 0; k < (*updateWeights)[i][j].size(); k++)
							(*updateWeights)[i][j][k] = optimizer->adam(this->learningRate, (*updateWeights)[i][j][k], ((*deltaFC)[i][k] * (*dActdW)[i][j]),m,v,this->epochsCount, 0.9f, 0.999f);
			}
			else if (opt == OPTIMIZER::Nadam) {
				float v, m;
				for (int i = 0; i < (*updateWeights).size(); i++)
					for (int j = 0; j < (*updateWeights)[i].size(); j++)
						for (int k = 0; k < (*updateWeights)[i][j].size(); k++)
							(*updateWeights)[i][j][k] = optimizer->nadam(this->learningRate, (*updateWeights)[i][j][k], ((*deltaFC)[i][k] * (*dActdW)[i][j]), m, v, this->epochsCount, 0.9f, 0.999f);
			}
			
			
			return updateWeights;
		}
		float lossFunc(V2D* target, V2D* output, LOSS loss) {
			float returnLoss = 0.0f;
			float mse = 0.0f;
			float crossEntropy = 0.0f;
			float error = 0.0f;
			float rmse = 0.0f;
			if ((*output).size() == (*target).size() && (*output)[0].size() == (*target)[0].size()) {

				for (int i = 0; i < (*output).size(); i++) {
					for (int j = 0; j < (*output)[i].size(); j++) {
						error = (*target)[i][j] - (*output)[i][j];
						mse += error * error;
						crossEntropy += (*target)[i][j] * log((*output)[i][j] + 1e-6f );
					}
				}
				crossEntropy = -crossEntropy;
				rmse = sqrt(mse);
				mse /= output[0].size();
			}
			if (loss == LOSS::BinaryCrossentropy || loss == LOSS::CategoricalCrossentropy || loss == LOSS::SparseCategoricalCrossentropy || loss == LOSS::CategoricalCrossentropy) 
				returnLoss = crossEntropy;
			else if (loss == LOSS::MSE) 
				returnLoss = mse;
			else if(loss == LOSS::RMSE) 
				returnLoss = rmse;
			
			return returnLoss;

		}
		V2D* derivativeLoss(V2D* target, V2D* output, LOSS loss) {
			V2D* deLoss = new V2D((*output).size(), V1D((*output)[0].size()));
			V2D deMSE = V2D((*output).size(), V1D((*output)[0].size(), 0.0f));
			V2D deRMSE = V2D((*output).size(), V1D((*output)[0].size(), 0.0f));
			V2D deBinCrossEntropy = V2D((*output).size(), V1D((*output)[0].size(), 0.0f));
			V2D deCategoricalCrossEntropy = V2D((*output).size(), V1D((*output)[0].size(), 0.0f));
			V2D deSparseCategoricalCrossEntropy = V2D((*output).size(), V1D((*output)[0].size(), 0.0f));

			float N = (float)((int)(*output).size() * (int)(*output)[0].size());
			for (int i = 0; i < (*output).size(); i++) {
				for (int j = 0; j < (*output)[0].size(); j++) {
					deMSE[i][j] = -((*target)[i][j] - (*output)[i][j]);
					deRMSE[i][j] = -((*target)[i][j] - (*output)[i][j]) / (N * this->lossFunc(target,output,LOSS::RMSE));
					deBinCrossEntropy[i][j] = -((*target)[i][j] / (*output)[i][j]) + ((1.0f - (*target)[i][j]) / (1.0f - (*output)[i][j]));
					deCategoricalCrossEntropy[i][j] = -((*target)[i][j] / ((*output)[i][j] + 1e-6f));
					deSparseCategoricalCrossEntropy[i][j] = -1.0f / ((*output)[i][j] + 1e-6f);
				}
			}

			if (this->loss == LOSS::MSE)
				*deLoss = deMSE;
			else if (this->loss == LOSS::RMSE)
				*deLoss = deRMSE;
			else if (this->loss == LOSS::BinaryCrossentropy)
				*deLoss = deBinCrossEntropy;
			else if (this->loss == LOSS::CategoricalCrossentropy)
				*deLoss = deCategoricalCrossEntropy;
			else if (this->loss == LOSS::SparseCategoricalCrossentropy)
				*deLoss = deSparseCategoricalCrossEntropy;

			return deLoss;

		}

	
		V2D* derivativeAct(V2D* weights, ACTIVATION act) {
			V2D* dw = new V2D((*weights));
			float alpha = 0.001f;
			
			if (act == ACTIVATION::Sigmoid || act == ACTIVATION::Softmax) {
				V1D preventOverflow = V1D((*weights).size(),0.0f);
				for (int i = 0; i < (*weights).size(); i++) {
					for (int j = 0; j < (*weights)[i].size(); j++) {
						(*dw)[i][j] = (*weights)[i][j] * (1.0f - (*weights)[i][j]);
						preventOverflow[i] += (*dw)[i][j];
					}
					for (int j = 0; j < (*weights)[i].size(); j++) {
						(*dw)[i][j] = (*dw)[i][j] / preventOverflow[i];
					}

				}
			}
			else if (act == ACTIVATION::ReLU) {
				for (int i = 0; i < (*weights).size(); i++) {
					for (int j = 0; j < (*weights)[i].size(); j++)
						(*dw)[i][j] = ((*weights)[i][j] > 0.0f) ? 1.0f : 0.0f;
				}
			}

			else if (act == ACTIVATION::Maxout) {
				float maxValue = -9999.0f;
				for (int i = 0; i < (*weights).size(); i++) 
					for (int j = 0; j < (*weights)[i].size(); j++) 
						if (maxValue < (*weights)[i][j]) maxValue = (*weights)[i][j];
				for (int i = 0; i < (*weights).size(); i++)
					for (int j = 0; j < (*weights)[i].size(); j++)
						(*dw)[i][j] = (*weights)[i][j] == maxValue ? 1.0f : 0.0f;		
			}
			else if (act == ACTIVATION::ELU) {
				for (int i = 0; i < (*weights).size(); i++) {
					for (int j = 0; j < (*weights)[i].size(); j++) 
						(*dw)[i][j] = ((*weights)[i][j] > 0.0f) ? 1.0f : (alpha * std::exp((*weights)[i][j]));
				}
			}
			else if (act == ACTIVATION::Leaky_ReLU) {
				for (int i = 0; i < (*weights).size(); i++) {
					for (int j = 0; j < (*weights)[i].size(); j++) 
						(*dw)[i][j] = ((*weights)[i][j] < 0) ? alpha : 1.0f;
				}
			}
			else {
				cout << "ACTIVATION 미선택 에러";
			}

			return dw;
		}
		V4D* derivativeAct(V4D* weights, ACTIVATION act) {
			for (int i = 0; i < (*weights).size(); i++) {
				for (int j = 0; j < (*weights)[i].size(); j++) {
					(*weights)[i][j] = *derivativeAct(&(*weights)[i][j], act);
				}
			}

			return weights;
		}

		V2D* deltaXweights(V2D* delta, V3D* weights) {
			clipDelta(delta);


			V2D* dXw = new V2D((*delta).size());
			for (int i = 0; i < (*delta).size(); i++) {
				(*dXw)[i].resize((*weights)[i].size());
				for (int j = 0; j < (*weights)[i].size(); j++) {
					(*dXw)[i][j] = 0.0f;
					for (int l = 0; l < (*weights)[i][j].size(); l++) {
						for (int k = 0; k < (*delta)[i].size(); k++) {
							(*dXw)[i][j] += (*delta)[i][k] * (*weights)[i][j][k];
						}
					}
				}
			}

			return dXw;
		}
		void clipDelta(V2D*	delta){
			double maxNorm = 5.0f;
			V1D deltaNormVal = V1D((*delta).size());

			for (int i = 0; i < (*delta).size(); i++) {
				deltaNormVal[i] = deltaNorm(&(*delta)[i]);
				if (deltaNormVal[i] > maxNorm) {
					for (int j = 0; j < (*delta)[i].size(); j++) {
						(*delta)[i][j] *= maxNorm / deltaNormVal[i];
					}
				}
			}

		}
		float deltaNorm(V1D* delta1D) {
			float norm = 0.0;
			for (int i = 0; i<(*delta1D).size(); i++) {
				norm += std::abs((*delta1D)[i]);
			}
			return norm;
		}
	};


}



#endif //