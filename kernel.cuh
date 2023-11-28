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
		virtual V4D calculateGPU(V4D data) { return dummy4D; }
		virtual V2D calculateGPU(V2D data) { return dummy2D; }
		virtual V1D calculateGPU(V1D data) { return dummy1D; }
		virtual V2D* backwardGPU(V2D* delta) { return &dummy2D; }
	};
	class ActivationGPU : public LayerGPU {
	public:
		int test() { return 0; }
		virtual V2D calculateGPU(V2D data) override {
			V2D dummy;
			return dummy;
		}
		virtual V4D calculateGPU(V4D data) override {
			V4D dummy;
			return dummy;
		}
		V2D* backwardGPU(V2D* delta) override {
			return delta;
		}
	};

	class PaddingGPU : public LayerGPU {
	public:
		V2D calculateGPU(V2D data) override {
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
		__global__ float* parallelCalculate(float* input, float* filter,int inputSize, int filterSize) {
			


		}
		virtual V4D calculateGPU(V4D data) override {
			std::cout << "Conv 시작\n";
			/*
			V2D image2D = im2col(X, 3, 8, 1);
			V2D filter2DR = filter2col(&X, filter, 1);
			V2D filter2D = transpose(filter2DR);

			V2D result = V2D(image2D.size(), V1D(filter2D[0].size(), 0.0f));


			imageXfilter(result, image2D, filter2D);


			V4D end = col2im(result, X.size(), X[0].size(), X[0][0].size(), X[0][0][0].size(), 3, 8, 1);
			*/
			V2D data2D = im2col(data, filterRowColGPU,(*filter3D).size(), strideGPU);
			
			V2D filter2D = filter2col((*this->filter3D),strideGPU);
			

			cout << "데이터크기: " << data2D.size() << " " << data2D[0].size() <<"\n";
			cout << "필터크기: " << filter2D.size() << " " << filter2D[0].size() << "\n";

			
			V2D cal = imageXfilter(data2D, filter2D);

			cout << "cal크기:" << cal.size() << " " << cal[0].size()<<"\n";
			
			

			V4D result = col2im(cal, data.size(), data[0].size(), data[0][0].size(), data[0][0][0].size(), filterRowColGPU, (*filter3D).size(), strideGPU);

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
		/*virtual V4D calculateTest(V4D data) override {
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
		V2D* backwardGPU(V2D* delta) override {
			return delta;
		}
	private:
		V2D im2col(const V4D& image, int kernelRowCol, int kernelSize, int stride) {

			cout << "TTTTTTTTTT크기4D: " << image.size() << "\n";
			cout << "TTTTTTTTTT크기3D: " << image[0].size() << "\n";
			cout << "TTTTTTTTTT크기2D: " << image[0][0].size() << "\n";
			cout << "TTTTTTTTTT크기1D: " << image[0][0][0].size() << "\n";

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
		/*V2D im2col(const V4D& image, int kernelSize, int stride) {
			int imageSize = image.size();
			int channel = image[0].size();
			int height = image[0][0].size();
			int width = image[0][0][0].size();
			int output_height = (height - kernelSize) / stride + 1;
			int output_width = (width - kernelSize) / stride + 1;

			V2D col;
			for (int f = 0; f < imageSize; f++) {
				for (int h = 0; h < output_height; ++h) {
					for (int d = 0; d < channel; ++d) {
						V1D col_entry;
						for (int w = 0; w < output_width; ++w) {
							for (int i = h * stride; i < h * stride + kernelSize; ++i) {
								for (int j = w * stride; j < w * stride + kernelSize; ++j) {
									col_entry.push_back(image[f][d][i][j]);
								}
							}
						}
						col.push_back(col_entry);
					}
				}

			}
			return col;
		}*/
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
		/*
		V4D col2im(const V2D& col, int imageSize, int channel, int height, int width, int kernelRowCol, int kernelSize, int stride) {
			int output_height = (height - kernelRowCol) / stride + 1;
			int output_width = (width - kernelRowCol) / stride + 1;

			cout << "col2i imageSize: " << imageSize << "\n";
			cout << "col2i chanXkernelSize: " << channel * kernelSize<<"\n";
			cout << "col2i output_height: " << output_height<<"\n";
			cout << "col2i outputwidth: " << output_width <<"\n";

			V4D image(imageSize, V3D(channel * kernelSize, V2D(output_height, V1D(output_width))));

			int col_index = 0;

			cout << "\n\n로그기록:\n\n";
			cout << "kernelSize: " << kernelSize << "\n";
			cout << "imageSize: " << imageSize << "\n";
			cout << "output_height: " << output_height << "\n";
			cout << "channel: " << channel << "\n";
			cout << "output_width: " << output_width << "\n";
			cout << " h * stride: " << output_height * stride << "\n";
			cout << " h * stride + kernelRowCol: " << output_height * stride + kernelRowCol << "\n";
			// 8 x 3 x 100 x 98 = 235,200 
			for (int i = 0; i < kernelSize; i++) {
				for (int f = 0; f < imageSize; f++) {
					for (int h = 0; h < output_height; ++h) {
						for (int d = 0; d < channel; ++d) {
							for (int w = 0; w < output_width; ++w) {
								for (int x = h * stride; x < h * stride + kernelRowCol; ++x) {
									for (int y = w * stride; y < w * stride + kernelRowCol; ++y) {
										image[f][d + i][x][y] = col[col_index][i];
									}
								}
							}
							col_index++;
						}
					}
				}
			}
			*/
			/*
			for (int i = 0; i < kernelSize; i++) {
				for (int f = 0; f < imageSize; f++) {
					for (int h = 0; h < output_height; ++h) {
						for (int d = 0; d < channel; ++d) {
							for (int w = 0; w < output_width; ++w) {
								for (int x = h * stride; x < h * stride + kernelRowCol; ++x) {
									for (int y = w * stride; y < w * stride + kernelRowCol; ++y) {
										image[f][d + i][x][y] = col[col_index][w];
									}
								}
							}
							col_index++;
						}
					}
				}
			}
			*/
		/*
			return image;
		}
		*/




		/*
		V4D col2im(const V2D& col, int imageSize, int channel, int height, int width, int kernelRowCol, int kernelSize, int stride) {
			int output_height = (height - kernelRowCol) / stride + 1;
			int output_width = (width - kernelRowCol) / stride + 1;

			V4D image = V4D(imageSize * kernelSize, V3D(channel));
			int cnt1D = 0;
			for (int f = 0; f < imageSize * kernelSize; f++) {
				for (int h = 0; h < output_height; ++h) {
					for (int d = 0; d < channel; ++d) {
						int cnt = 0;
						for (int w = 0; w < output_width; ++w) {
							image[f][d].resize(h * stride + kernelRowCol);
							for (int i = h * stride; i < h * stride + kernelRowCol; ++i) {
								image[f][d][i].resize(w * stride + kernelRowCol);
								for (int j = w * stride; j < w * stride + kernelRowCol; ++j) {
									image[f][d][i][j] += col[cnt1D][cnt];
									cnt++;
								}
							}

						}
						cnt1D++;
					}
				}

			}

			return image;
		}
		/*
		V2D filter2col(const V4D* image, const V3D& filter, int stride) {
			V2D col;
			int imageXSize = ((*image)[0][0][0].size() - filter[0].size()) / stride + 1;
			int imageYSize = ((*image)[0][0].size() - filter[0].size()) / stride + 1;
			int imageChannel = (*image)[0].size();
			int imageSize = (*image).size();
			for (int i = 0; i < filter.size(); i++) {
				for (int j = 0; j < imageSize; j++) {
					for (int k = 0; k < imageChannel; k++) {
						for (int l = 0; l < imageYSize; l++) {
							V1D col1D;
							for (int o = 0; o < imageXSize; o++) {
								for (int m = 0; m < filter[i].size(); m++) {
									for (int n = 0; n < filter[0].size(); n++) {
										col1D.push_back(filter[i][m][n]);
									}
								}
							}
							col.push_back(col1D);
						}
					}
				}
			}
			return col;
		}*/
		V2D transpose(const vector<V1D>& data) {
			V2D trans = V2D(data[0].size(), V1D(data.size(), 0.0f));
			for (int i = 0; i < data.size(); i++) {
				for (int j = 0; j < data[i].size(); j++) {
					trans[j][i] = data[i][j];
				}
			}
			return trans;
		}
		V2D imageXfilter(V2D image, V2D filter) {
			/*
			* int image4D_XSize;
			이미지: 48 86436
		필터: 8 9 이면
			result가 x축이 9604가 나와야하는데
			x축이 9임.


			*/
			V2D result = V2D(image.size(), V1D(image[0].size() / filter[0].size(), 0.0f));
			for (int i = 0; i < image.size(); i++) {
				for (int j = 0; j < image[i].size(); j++) {
					result[i][(j / filter[0].size()) % filter[0].size()] += image[i][j] * filter[i % filter.size()][j % filter[i % filter.size()].size()];
				}
			}
			return result;
		}

	};
	class PoolingGPU : public LayerGPU {
	public:
		V2D calculateGPU(V2D data) override {
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
		V2D calculateGPU(V2D data) override {
			V2D dummy;
			return dummy;
		}
		V2D* backwardGPU(V2D* delta) override {
			return delta;
		}
	};
	class FullyConnectedGPU : public LayerGPU {
	public:
		V2D calculateGPU(V2D data) override {
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