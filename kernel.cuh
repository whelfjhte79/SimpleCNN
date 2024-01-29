#ifndef KERNEL_CUH
#define KERNEL_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "SimpleCNN.h"
#include<iostream>

using V1D = std::vector<float>;
using V2D = std::vector<V1D>;
using V3D = std::vector<V2D>;
using V4D = std::vector<V3D>;
using V5D = std::vector<V4D>;

#ifdef __cplusplus 
extern "C" {
#endif
	class RandomGPU {
	private:
	public:

	};
	class LayerGPU{
	private:
		V1D dummy1D;
		V2D dummy2D; 
		V3D dummy3D;
		V4D dummy4D;
	public:
		virtual V4D calculateGPU(V4D& data) { return dummy4D; }
		virtual V2D calculateGPU(V2D& data) { return dummy2D; }
		virtual V1D calculateGPU(V1D& data) { return dummy1D; }
		virtual V2D* backwardGPU(V2D* delta) { return &dummy2D; }
	};
	class ActivationGPU : public LayerGPU {
	public:
		int test() { return 0; }
		virtual V2D calculateGPU(V2D& data) override {
			V2D dummy;
			return dummy;
		}
		virtual V4D calculateGPU(V4D& data) override {
			V4D dummy;
			return dummy;
		}
		V2D* backwardGPU(V2D* delta) override {
			return delta;
		}
	};

	class PaddingGPU : public LayerGPU {
	public:
		V2D calculateGPU(V2D& data) override {
			V2D dummy;
			return dummy;
		}
		V2D* backwardGPU(V2D* delta) override {
			return delta;
		}
	};
	class ConvGPU : public LayerGPU {
	private:
		V3D* filter3D = nullptr;
		//V1D* filter1D = nullptr;
		ActivationGPU* activationGPU = nullptr;
		unsigned int strideGPU;
		unsigned int filterRowColGPU;
	public:
		ConvGPU(){
		}
		~ConvGPU() {
			//delete filter3D;
		}
		virtual void setFilter3DGPU(V3D* filter) {
			this->filter3D = filter;
		}
		void setStrideGPU(int stride) {
			this->strideGPU = stride;
		}
		void setFilterRowColGPU(int filterRowCol) {
			this->filterRowColGPU = filterRowCol;
		}
		virtual V4D calculateGPU(V4D& data) override {
			std::cout << "Conv 시작\n";
			/*
			V2D image2D = im2col(X, 3, 8, 1);
			V2D filter2DR = filter2col(&X, filter, 1);
			V2D filter2D = transpose(filter2DR);

			V2D result = V2D(image2D.size(), V1D(filter2D[0].size(), 0.0f));


			imageXfilter(result, image2D, filter2D);


			V4D end = col2im(result, X.size(), X[0].size(), X[0][0].size(), X[0][0][0].size(), 3, 8, 1);
			*/

			float** data2D = nullptr;
			im2colGPU(data, filterRowColGPU,(*filter3D).size(), strideGPU, data2D);
			//V2D data2D;
			float** filter2D = nullptr;
			filter2colGPU((*this->filter3D), strideGPU, filter2D);
			
			
			V2D cal = imageXfilterGPU(data2D, filter2D);

			

			V4D result = col2im(cal, data.size(), data[0].size(), data[0][0].size(), data[0][0][0].size(), filterRowColGPU, (*filter3D).size(), strideGPU);

			//연산끝나면
			//delete data2D
			//delete filter2D

			return result;
		}
		/*
		V4D calculateGPU(V4D data) override {
			float* dInput = nullptr;
			float* dFilter = nullptr;
			float* dOutput = nullptr;

			int imageX = 224, imageY = 224, imageC = 3, dataSize = data.size();
			int totalSize = imageX * imageY;
			int dataZeroSize = data[0].size();
			int filterX = (*filter3D)[0][0].size(), filterY = (*filter3D)[0].size(), filterCnt = (*filter3D).size();
			for (int i = 0; i < dataSize; i++) {
				for (int m = 0; m < imageC; m++) {
					for (int j = 0; j < filterCnt; j++) {
						for (int k = 0; k < (imageY - filterCnt) / stride + 1; k++) {
							for (int l = 0; l < (imageX - filterX) / stride + 1; l++) {

							}
						}
					}
				}
			}
		}*/

		/*
		V4D calculateGPU(V4D data) override {
			V4D actMap4D = V4D(data.size());

			for (int i = 0; i < data.size(); i++) {
				actMap4D[i].resize(data[i].size() * (*filter3D).size());
				for (int j = 0; j < data[i].size(); j++)
					for (int k = 0; k < (*filter3D).size(); k++)
						for (int l = 0; l < (data[i][j].size() - (*filter3D)[0][0].size()) / stride + 1; l++)
							actMap4D[i][j * (*filter3D).size() + k].push_back(vector<float>((data[i][j][l].size() - (*filter3D)[0][0].size()) / stride + 1));
			}

			float* dInput = nullptr;
			float* dFilter = nullptr;
			float* dOutput = nullptr;

			int filter3DSize = (*this->filter3D).size() * (*this->filter3D)[0].size() * (*this->filter3D)[0][0].size() * sizeof(float);
			cudaMalloc(&dInput, filter3DSize);
			cudaMalloc(&dFilter, filter3DSize);
			
			cudaMalloc(&dOutput, filter3DSize);



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
								
								cudaMemcpy(dInput, data[i][m][k*stride][l*stride].data(), sizeof(float), cudaMemcpyHostToDevice);
								cudaMemcpy(dFilter, (*this->filter1D).data(), sizeof(float), cudaMemcpyHostToDevice);




								//actMap4D[i][m * filter3D->size() + j][k][l] = sum;
							}
						}
					}
				}
			}
		}*/
		/*
		V2D calculateGPU(V2D data) override {
			// data는 항상 정사각형, 고정벡터여야함.
			// 1차원이지만 3차원 x y c도 알고있어야함.
			int imageX = 224;
			int imageY = 224;
			int imageC = 3;
			int totalSize = imageX*imageY;
			int dataZeroSize = data[0].size();
			int filterX;
			int filterY;
			int filterCnt;
			if (totalSize != dataZeroSize) { cout << "에러"; }
			
			V2D actMap2D = V2D(data.size(),V1D());
			int index = 0;
			for (int i = 0; i < data.size(); i++) {
				for (int m = 0; m < imageC; m++) {
					for (int j = 0; j < filterCnt; j++) {
						for (int k = 0; k < (imageY - filterCnt) / stride + 1; k++) {
							for (int l = 0; l < (imageX - filterX) / stride + 1; l++) {
								float sum = 0.0;
								for (int u = 0; u < filterY; u++) {
									for (int v = 0; v < filterX; v++) {
								//		sum += data[i][m][k * stride + u][l * stride + v] * (*filter3D)[j][u][v];
									}
								}
								// t[i * maxY * maxX + j * maxX + k] = 1.0f;
								data[i][index++];
								actMap2D[i][index++] = sum;
								//actMap2D[i * (data[i].size() * (*filter3D).size()) + m * (*filter3D).size() + j].push_back(sum);
								//actMap4D[i][m * filter3D->size() + j][k][l] = sum;
							}
						}
					}
				}
			}
			
		}*/
		/*
		V2D calculateGPU(V2D data) override {
			float* d_input = nullptr;
			float* d_filter = nullptr;
			float* d_output = nullptr; // input, filter, stride 고려해서 크기 계산
			cudaMalloc(&d_input, data[0].size() * sizeof(float));
			cudaMalloc(&d_filter, (*this->filter1D).size() * sizeof(float));
			for (int i = 0; i < data.size(); i++) {
				
				//cudaMalloc(&d_output, data[i].size() /  * sizeof(float));
				cudaMemcpy(d_input, data[i].data(), sizeof(float), cudaMemcpyHostToDevice);
				cudaMemcpy(d_filter, (*this->filter1D).data(), sizeof(float), cudaMemcpyHostToDevice);
				int size = 0;
				int threadsPerBlock = 128;
				int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
				//parallelCalculate << < threadsPerBlock, blocksPerGrid >> > ();

			}

			//std::vector<int> dest(std::begin(src), std::end(src));
			data = (*this->activationGPU).calculateGPU(data);
			return data;
		}*/

		V2D* backwardGPU(V2D* delta) override {
			return delta;
		}
	private:
		
		//float**
		void im2colGPU(const V4D& image, int kernelRowCol, const int kernelSize, int stride, float** col) {
			const int imageSize = image.size();
			const int channel = image[0].size();
			int height = image[0][0].size();
			int width = image[0][0][0].size();
			int output_height = (height - kernelRowCol) / stride + 1;
			int output_width = (width - kernelRowCol) / stride + 1;
			const int _kernelSize = kernelSize;

			int col_entry_size = output_height * output_width * kernelRowCol * kernelRowCol;
			col = new float* [kernelSize * imageSize * channel];
			int cnt = 0;

			for (int i = 0; i < kernelSize; i++) {
				for (int f = 0; f < imageSize; f++) {
					for (int d = 0; d < channel; ++d) {
						col[cnt] = new float[col_entry_size];
						int col_entry = 0;
						for (int h = 0; h < output_height; ++h) {
							for (int w = 0; w < output_width; ++w) {
								for (int i = h * stride; i < h * stride + kernelRowCol; ++i) {
									for (int j = w * stride; j < w * stride + kernelRowCol; ++j) {
										col[cnt][col_entry++] += image[f][d][i][j];
									}
								}
							}

						}
						cnt++;
					}
				}
			}
		}
		void filter2colGPU(const V3D& filter, int stride, float** filter2D) {
			filter2D = new float* [filter.size()];
			for (int i = 0; i < filter.size(); i++) {
				filter2D[i] = new float[(int)(filter[0].size() * filter[0][0].size())];
				int cnt = 0;
				for (int j = 0; j < filter[i].size(); j++) {
					for (int k = 0; k < filter[i][j].size(); k++) {
						filter2D[i][cnt++] = filter[i][j][k];
					}
				}
			}
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
		V2D transpose(const vector<V1D>& data) {
			V2D trans = V2D(data[0].size(), V1D(data.size(), 0.0f));
			for (int i = 0; i < data.size(); i++) {
				for (int j = 0; j < data[i].size(); j++) {
					trans[j][i] = data[i][j];
				}
			}
			return trans;
		}

		/*
		__global__ void multiplyKernel(float* a, float* b, float* c, int M, int K, int N) {
			int tid = blockIdx.x * blockDim.x + threadIdx.x;

			if (tid < M * N) {
				int row = tid / N;
				int col = tid % N;

				float sum = 0.0f;
				for (int i = 0; i < K; ++i) {
					sum += a[row * K + i] * b[i * N + col];
				}

				c[row * N + col] = sum;
			}
		}

		std::vector<std::vector<float>> imageXfilterGPU(float** image, float** filter) {
			int imageRowSize = sizeof(image) / sizeof(image[0]);
			int imageColSize = sizeof(image[0]) / sizeof(image[0][0]);
			int filterRowSize = sizeof(filter) / sizeof(filter[0]);
			int filterColSize = sizeof(filter[0]) / sizeof(filter[0][0]);

			std::vector<std::vector<float>> result(imageRowSize, std::vector<float>(filterColSize, 0.0f));

			float* hostImage = *image;
			float* hostFilter = *filter;
			float* hostResult = new float[imageRowSize * filterColSize];

			float* deviceImage = nullptr;
			float* deviceFilter = nullptr;
			float* deviceResult = nullptr;

			int imageSize = imageRowSize * imageColSize * sizeof(float);
			int filterSize = filterRowSize * filterColSize * sizeof(float);
			int resultSize = imageRowSize * filterColSize * sizeof(float);

			cudaMalloc((void**)&deviceImage, imageSize);
			cudaMalloc((void**)&deviceFilter, filterSize);
			cudaMalloc((void**)&deviceResult, resultSize);

			cudaMemcpy(deviceImage, hostImage, imageSize, cudaMemcpyHostToDevice);
			cudaMemcpy(deviceFilter, hostFilter, filterSize, cudaMemcpyHostToDevice);

			dim3 blockSize(256);
			dim3 gridSize((imageRowSize * filterColSize + blockSize.x - 1) / blockSize.x);

			multiplyKernel<<<gridSize,blockSize>>>(deviceImage, deviceFilter, deviceResult, imageRowSize, imageColSize, filterColSize);
			cudaMemcpy(hostResult, deviceResult, resultSize, cudaMemcpyDeviceToHost);

			for (int i = 0; i < imageRowSize; i++) {
				for (int j = 0; j < filterColSize; j++) {
					result[i][j] = hostResult[i * filterColSize + j];
				}
			}

			// 메모리 해제
			delete[] hostResult;
			cudaFree(deviceImage);
			cudaFree(deviceFilter);
			cudaFree(deviceResult);

			return result;
		}*/

		
		__global__ void multiplyKernel(float* a, float* b, float* c, int M, int K, int N) {
			int tid = blockIdx.x * blockDim.x + threadIdx.x;

			if (tid < M * N) {
				int row = tid / N;
				int col = tid % N;

				float sum = 0.0f;
				for (int i = 0; i < K; ++i) {
					sum += a[row * K + i] * b[i * N + col];
				}

				c[row * N + col] = sum;
			}
		}
		V2D imageXfilterGPU(float** image, float** filter) {
			int imageRowSize = sizeof(image) / sizeof(image[0]);
			int imageColSize = sizeof(image[0]) / sizeof(image[0][0]);
			int filterRowSize = sizeof(filter) / sizeof(filter[0]);
			int filterColSize = sizeof(filter[0]) / sizeof(filter[0][0]);


			V2D result = V2D(imageRowSize, V1D(filterColSize, 0.0f));

			//float* cudaImage = new float[imageRowSize * imageColSize];
			//float* cudaFilter = new float[filterRowSize * filterColSize];
			float* hostImage = *image;
			float* hostFilter = *filter;
			float* hostResult = new float[imageRowSize * filterColSize];

			float* deviceImage = nullptr;
			float* deviceFilter = nullptr;
			float* deviceResult = nullptr;

			int imageSize = imageRowSize * imageColSize * sizeof(float);
			int filterSize = filterRowSize * filterColSize * sizeof(float);
			int resultSize = imageRowSize * filterColSize * sizeof(float);

			cudaMalloc((void**)&deviceImage, imageSize);
			cudaMalloc((void**)&deviceFilter, filterSize);
			cudaMalloc((void**)&deviceResult, resultSize);

			cudaMemcpy(deviceImage, hostImage, imageSize, cudaMemcpyHostToDevice);
			cudaMemcpy(deviceFilter, hostImage, filterSize, cudaMemcpyHostToDevice);

			//dim3 blockSize(16, 16);
			//dim3 gridSize((filterColSize + blockSize.x - 1) / blockSize.x, (imageRowSize + blockSize.y - 1) / blockSize.y);
			dim3 blockSize(256);
			dim3 gridSize((imageRowSize * filterColSize + blockSize.x - 1) / blockSize.x);

			//multiplyKernel <<<gridSize, blockSize>>> (deviceImage, deviceFilter, deviceResult, imageRowSize, imageColSize, filterColSize);
			cudaMemcpy(hostResult, deviceResult, resultSize, cudaMemcpyDeviceToHost);

			
			for (int i = 0; i < imageRowSize; i++) {
				V1D result1D;
				for (int j = 0; j < filterColSize; j++) {
					result1D.push_back(hostResult[i * filterColSize + j]);
				}
				result.push_back(result1D);
			}
			return result;
		}

	};
	class PoolingGPU : public LayerGPU {
	public:
		V2D calculateGPU(V2D& data) override {
			V2D dummy;
			return dummy;
		}
		V2D* backwardGPU(V2D* delta) override {
			return delta;
		}
	};
	class FlattenGPU : public LayerGPU {
	public:
		__global__ V2D parallelCalculate(V4D data) {

		}
		V2D flatten(V4D data) {

		}
		V2D calculateGPU(V2D& data) override {
			V2D dummy;
			return dummy;
		}
		V2D* backwardGPU(V2D* delta) override {
			return delta;
		}
	};
	class FullyConnectedGPU : public LayerGPU {
	public:
		V2D calculateGPU(V2D& data) override {
			V2D dummy;
			return dummy;
		}
		V2D* backwardGPU(V2D* delta) override {
			return delta;
		}
	};




#ifdef __cplusplus 
}
#endif

#endif // KERNEL_CUH