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

// SimpleCNN - 레이어만 쌓으면 되는 C++ CNN
//
//   #include "SimpleCNN.h"
//   using namespace cnn;
//
//   Dataset train = Dataset::loadFolder("images", 64, 64);    // 폴더 이름 = 클래스
//   CNN model(Device::CUDA);                                  // GPU 가 없으면 자동으로 CPU
//   model.add(new Conv(3, 16, 1, 1, ACTIVATION::ReLU));
//   model.add(new Pooling(POOLING::Max));
//   model.add(new FullyConnected(10, ACTIVATION::Softmax));
//   model.compile(OPTIMIZER::Adam, LOSS::CategoricalCrossentropy);
//   model.fit(train, 5, 64);
//   int label = model.predict("cat.bmp");
//
// 파일은 두 개뿐이다.
//   SimpleCNN.h    이 파일. 라이브러리 전체
//   SimpleCNN.cuh  CUDA 커널. 이 파일이 알아서 읽어 들이므로 따로 include 하지 않는다.
//
// 목차
//   1. 기본 정의        enum, Shape(텐서 모양), 멀티코어 반복
//   2. GPU 런타임       CUDA 드라이버 / NVRTC / cuBLAS 를 실행 중에 연결, GPU 메모리 풀
//   3. Tensor           연속 메모리 한 덩어리 (CPU 또는 GPU)
//   4. 연산             GEMM, im2col, FFT 합성곱, 풀링, 활성화 (cpu:: / gpu::)
//   5. 레이어 (블럭)    Conv, Pooling, Padding, Activation, Flatten, FullyConnected
//   6. 옵티마이저       SGD, Momentum, Adagrad, RMSProp, Adam, Nadam
//   7. 이미지 / 데이터셋
//   8. CNN 모델         add → compile → fit → predict / save / load

#ifndef SIMPLE_CNN_H
#define SIMPLE_CNN_H

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#define SCNN_STDCALL __stdcall
#else
#include <dlfcn.h>
#define SCNN_STDCALL
#endif

// JPG / PNG 는 stb_image.h 를 이 파일 옆에 두면 자동으로 읽는다. (없으면 BMP / PGM / PPM 만)
#if defined(__has_include) && !defined(SIMPLECNN_NO_STB)
#if __has_include("stb_image.h")
#define SIMPLECNN_STB
#ifndef STBI_INCLUDE_STB_IMAGE_H
#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_STDIO
#endif
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
#include "stb_image.h"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
#endif
#endif

namespace cnn {

	// ================================================================================
	// 1. 기본 정의
	// ================================================================================

	// 예전 코드 호환용 중첩 vector (입력으로만 받는다)
	using V1D = std::vector<float>;
	using V2D = std::vector<V1D>;
	using V3D = std::vector<V2D>;
	using V4D = std::vector<V3D>;

	enum class Device { CPU, CUDA };
	enum class LAYER { Conv, Pool, Act, Padding, FC, Flatten, None };
	enum class POOLING { Max, Average, Min };
	enum class ACTIVATION { None, Error, Sigmoid, ReLU, Leaky_ReLU, TanH, Maxout, ELU, Softmax };
	enum class OPTIMIZER { GD, Mini_SGD, Momentum, Adagrad, RMSProp, Adam, Nadam };
	enum class LOSS { MSE, RMSE, BinaryCrossentropy, CategoricalCrossentropy, SparseCategoricalCrossentropy };
	// 합성곱 계산 방식. Auto 는 첫 배치에서 직접 재보고 빠른 쪽을 고른다.
	enum class CONV_ALGO { Auto, Im2col, FFT, Direct };

	// 활성화 함수 상수 (SimpleCNN.cuh 의 LEAKY_SLOPE, ELU_ALPHA 와 같은 값)
	const float kLeakySlope = 0.01f;
	const float kEluAlpha = 1.0f;

	// 텐서 모양 NCHW (개수, 채널, 세로, 가로). FC 출력은 (n, 유닛, 1, 1)
	struct Shape {
		int n = 0;
		int c = 0;
		int h = 0;
		int w = 0;

		Shape() {}
		Shape(int n, int c, int h, int w) : n(n), c(c), h(h), w(w) {}
		size_t sample() const { return (size_t)c * h * w; }
		size_t size() const { return (size_t)n * c * h * w; }
		std::string str() const {
			return "(" + std::to_string(c) + ", " + std::to_string(h) + ", " + std::to_string(w) + ")";
		}
	};

	inline const char* toString(Device device) {
		if (device == Device::CUDA)
			return "CUDA";
		return "CPU";
	}
	inline const char* toString(ACTIVATION act) {
		const char* names[] = { "None", "Error", "Sigmoid", "ReLU", "Leaky_ReLU", "TanH", "Maxout", "ELU", "Softmax" };
		return names[(int)act];
	}
	inline const char* toString(POOLING pool) {
		const char* names[] = { "Max", "Average", "Min" };
		return names[(int)pool];
	}
	inline const char* toString(CONV_ALGO algo) {
		const char* names[] = { "Auto", "im2col+GEMM", "FFT", "Direct" };
		return names[(int)algo];
	}
	inline ACTIVATION parseActivation(const std::string& text) {
		for (int i = 0; i <= (int)ACTIVATION::Softmax; ++i) {
			if (text == toString((ACTIVATION)i))
				return (ACTIVATION)i;
		}
		throw std::runtime_error("알 수 없는 활성화 함수: " + text);
	}

	inline int threadCount() {
#ifdef _OPENMP
		return omp_get_max_threads();
#else
		return 1;
#endif
	}
	inline int threadIndex() {
#ifdef _OPENMP
		return omp_get_thread_num();
#else
		return 0;
#endif
	}
	// fn(0) ~ fn(count-1) 을 여러 코어에 나눠 실행. (OpenMP 를 끄면 그냥 순서대로 실행)
	template <typename F>
	inline void parallelFor(int count, F fn) {
#ifdef _OPENMP
		if (count > 1 && !omp_in_parallel()) {
#pragma omp parallel for schedule(dynamic)
			for (int i = 0; i < count; ++i)
				fn(i);
			return;
		}
#endif
		for (int i = 0; i < count; ++i)
			fn(i);
	}

	class Timer {
	private:
		std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
	public:
		void reset() { this->start = std::chrono::steady_clock::now(); }
		double seconds() const { return std::chrono::duration<double>(std::chrono::steady_clock::now() - this->start).count(); }
	};

	inline std::string formatBytes(double bytes) {
		const char* units[] = { "B", "KB", "MB", "GB", "TB" };
		int u = 0;
		while (bytes >= 1024.0 && u < 4) {
			bytes /= 1024.0;
			++u;
		}
		char text[32];
		std::snprintf(text, sizeof(text), "%.1f %s", bytes, units[u]);
		return text;
	}

	// 환경 변수 읽기 (MSVC 는 getenv 가 SDL 검사에서 오류라 _dupenv_s 사용)
	inline std::string envVar(const char* name) {
		std::string value;
#ifdef _MSC_VER
		char* buffer = nullptr;
		size_t length = 0;
		if (_dupenv_s(&buffer, &length, name) == 0 && buffer != nullptr) {
			value = buffer;
			std::free(buffer);
		}
#else
		const char* text = std::getenv(name);
		if (text != nullptr)
			value = text;
#endif
		return value;
	}

	// ================================================================================
	// 2. GPU 런타임
	//
	//   Backend : GPU 에 일을 시키는 최소 기능 (메모리, 복사, 커널 실행, 행렬곱)
	//   Driver  : Backend 의 실제 구현. nvcuda / nvrtc / cublas 를 실행 중에 찾아 연결한다.
	//             → 프로젝트에 CUDA 를 링크하지 않아도 되고, GPU 가 없는 PC 에서도 같은 exe 가 돈다.
	//   Runtime : 프로그램 전체에서 하나. GPU 메모리 풀과 커널 실행 도우미.
	// ================================================================================
	namespace cuda {

		inline const char* kernelSource() {
			static const char* source =
#include "SimpleCNN.cuh"
				;
			return source;
		}

		class Backend {
		public:
			virtual ~Backend() {}
			// GPU 를 찾고 커널을 컴파일한다. 실패하면 error 에 이유를 적고 false
			virtual bool start(std::string& info, std::string& error) = 0;
			virtual void* alloc(size_t bytes) = 0;
			virtual void release(void* ptr) = 0;
			virtual void upload(void* dst, const void* src, size_t bytes) = 0;
			virtual void download(void* dst, const void* src, size_t bytes) = 0;
			virtual void copy(void* dst, const void* src, size_t bytes) = 0;
			virtual void launch(const char* kernel, unsigned gridX, unsigned gridY, unsigned block, void** args) = 0;
			// 행 우선 행렬곱. cuBLAS 가 있으면 true, 없으면 false (→ gemm_kernel 사용)
			virtual bool gemm(bool transA, bool transB, int M, int N, int K, float alpha, const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc) { return false; }
		};

		class Driver : public Backend {
		private:
			typedef unsigned long long Ptr; // CUdeviceptr

			// CUDA 드라이버 API (nvcuda.dll / libcuda.so)
			int (SCNN_STDCALL* cuInit)(unsigned) = nullptr;
			int (SCNN_STDCALL* cuDriverGetVersion)(int*) = nullptr;
			int (SCNN_STDCALL* cuDeviceGetCount)(int*) = nullptr;
			int (SCNN_STDCALL* cuDeviceGet)(int*, int) = nullptr;
			int (SCNN_STDCALL* cuDeviceGetName)(char*, int, int) = nullptr;
			int (SCNN_STDCALL* cuDeviceGetAttribute)(int*, int, int) = nullptr;
			int (SCNN_STDCALL* cuDeviceTotalMem)(size_t*, int) = nullptr;
			int (SCNN_STDCALL* cuDevicePrimaryCtxRetain)(void**, int) = nullptr;
			int (SCNN_STDCALL* cuCtxSetCurrent)(void*) = nullptr;
			int (SCNN_STDCALL* cuMemAlloc)(Ptr*, size_t) = nullptr;
			int (SCNN_STDCALL* cuMemFree)(Ptr) = nullptr;
			int (SCNN_STDCALL* cuMemcpyHtoD)(Ptr, const void*, size_t) = nullptr;
			int (SCNN_STDCALL* cuMemcpyDtoH)(void*, Ptr, size_t) = nullptr;
			int (SCNN_STDCALL* cuMemcpyDtoD)(Ptr, Ptr, size_t) = nullptr;
			int (SCNN_STDCALL* cuModuleLoadData)(void**, const void*) = nullptr;
			int (SCNN_STDCALL* cuModuleGetFunction)(void**, void*, const char*) = nullptr;
			int (SCNN_STDCALL* cuLaunchKernel)(void*, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, void*, void**, void**) = nullptr;
			int (SCNN_STDCALL* cuGetErrorString)(int, const char**) = nullptr;

			// NVRTC: CUDA 소스를 실행 중에 컴파일 (CUDA Toolkit 에 포함)
			int (*nvrtcCreateProgram)(void**, const char*, const char*, int, const char* const*, const char* const*) = nullptr;
			int (*nvrtcCompileProgram)(void*, int, const char* const*) = nullptr;
			int (*nvrtcGetPTXSize)(void*, size_t*) = nullptr;
			int (*nvrtcGetPTX)(void*, char*) = nullptr;
			int (*nvrtcGetCUBINSize)(void*, size_t*) = nullptr;
			int (*nvrtcGetCUBIN)(void*, char*) = nullptr;
			int (*nvrtcGetProgramLogSize)(void*, size_t*) = nullptr;
			int (*nvrtcGetProgramLog)(void*, char*) = nullptr;
			int (*nvrtcDestroyProgram)(void**) = nullptr;
			int (*nvrtcGetNumSupportedArchs)(int*) = nullptr;
			int (*nvrtcGetSupportedArchs)(int*) = nullptr;

			// cuBLAS: 가장 빠른 행렬곱 (없으면 gemm_kernel 로 대신한다)
			int (SCNN_STDCALL* cublasCreate)(void**) = nullptr;
			int (SCNN_STDCALL* cublasSgemm)(void*, int, int, int, int, int, const float*, const float*, int, const float*, int, const float*, float*, int) = nullptr;

			void* context = nullptr;
			void* module = nullptr;
			void* cublas = nullptr;
			int arch = 0; // compute capability (예: RTX 4070 = 89)
			std::unordered_map<std::string, void*> functions;

			static void* openLibrary(const std::string& path) {
#ifdef _WIN32
				return (void*)LoadLibraryExA(path.c_str(), NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
#else
				return dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
			}
			template <typename T>
			static bool bind(void* lib, T& fn, const char* name) {
#ifdef _WIN32
				fn = reinterpret_cast<T>((void*)GetProcAddress((HMODULE)lib, name));
#else
				fn = reinterpret_cast<T>(dlsym(lib, name));
#endif
				return fn != nullptr;
			}
			// CUDA Toolkit 폴더들에서 prefix 로 시작하는 라이브러리를 찾는다. (최신 버전 우선)
			static void* findLibrary(const std::string& prefix, const std::string& exclude) {
				std::vector<std::string> roots;
				const char* envs[] = { "CUDA_PATH", "CUDA_HOME" };
				for (const char* env : envs) {
					if (!envVar(env).empty())
						roots.push_back(envVar(env));
				}
#ifdef _WIN32
				std::string installed = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA";
#else
				std::string installed = "/usr/local";
#endif
				std::error_code ec;
				std::vector<std::string> versions;
				for (const auto& entry : std::filesystem::directory_iterator(installed, ec)) {
					std::string name = entry.path().filename().string();
					if (name.rfind("v", 0) == 0 || name.rfind("cuda", 0) == 0)
						versions.push_back(entry.path().string());
				}
				std::sort(versions.rbegin(), versions.rend());
				roots.insert(roots.end(), versions.begin(), versions.end());

				const char* subdirs[] = { "/bin", "/bin/x64", "/lib64", "/lib/x64" };
				for (const std::string& root : roots) {
					for (const char* sub : subdirs) {
						std::vector<std::string> files;
						for (const auto& entry : std::filesystem::directory_iterator(root + sub, ec)) {
							std::string name = entry.path().filename().string();
							if (name.rfind(prefix, 0) == 0 && name.find(exclude) == std::string::npos)
								files.push_back(entry.path().string());
						}
						std::sort(files.rbegin(), files.rend());
						for (const std::string& file : files) {
							void* lib = openLibrary(file);
							if (lib != nullptr)
								return lib;
						}
					}
				}
				return nullptr;
			}
			void check(int result, const char* what) {
				if (result == 0)
					return;
				const char* message = "unknown";
				this->cuGetErrorString(result, &message);
				throw std::runtime_error(std::string("CUDA 오류 (") + what + "): " + message);
			}
			bool loadDriver() {
#ifdef _WIN32
				void* lib = openLibrary("nvcuda.dll");
#else
				void* lib = openLibrary("libcuda.so.1");
#endif
				if (lib == nullptr)
					return false;
				return bind(lib, this->cuInit, "cuInit") && bind(lib, this->cuDriverGetVersion, "cuDriverGetVersion")
					&& bind(lib, this->cuDeviceGetCount, "cuDeviceGetCount") && bind(lib, this->cuDeviceGet, "cuDeviceGet")
					&& bind(lib, this->cuDeviceGetName, "cuDeviceGetName") && bind(lib, this->cuDeviceGetAttribute, "cuDeviceGetAttribute")
					&& bind(lib, this->cuDeviceTotalMem, "cuDeviceTotalMem_v2") && bind(lib, this->cuDevicePrimaryCtxRetain, "cuDevicePrimaryCtxRetain")
					&& bind(lib, this->cuCtxSetCurrent, "cuCtxSetCurrent") && bind(lib, this->cuMemAlloc, "cuMemAlloc_v2")
					&& bind(lib, this->cuMemFree, "cuMemFree_v2") && bind(lib, this->cuMemcpyHtoD, "cuMemcpyHtoD_v2")
					&& bind(lib, this->cuMemcpyDtoH, "cuMemcpyDtoH_v2") && bind(lib, this->cuMemcpyDtoD, "cuMemcpyDtoD_v2")
					&& bind(lib, this->cuModuleLoadData, "cuModuleLoadData") && bind(lib, this->cuModuleGetFunction, "cuModuleGetFunction")
					&& bind(lib, this->cuLaunchKernel, "cuLaunchKernel") && bind(lib, this->cuGetErrorString, "cuGetErrorString");
			}
			bool loadNvrtc() {
#ifdef _WIN32
				void* lib = findLibrary("nvrtc64_", "builtins");
#else
				void* lib = openLibrary("libnvrtc.so");
				if (lib == nullptr)
					lib = findLibrary("libnvrtc.so", "builtins");
#endif
				if (lib == nullptr)
					return false;
				bind(lib, this->nvrtcGetCUBINSize, "nvrtcGetCUBINSize");
				bind(lib, this->nvrtcGetCUBIN, "nvrtcGetCUBIN");
				bind(lib, this->nvrtcGetNumSupportedArchs, "nvrtcGetNumSupportedArchs");
				bind(lib, this->nvrtcGetSupportedArchs, "nvrtcGetSupportedArchs");
				return bind(lib, this->nvrtcCreateProgram, "nvrtcCreateProgram") && bind(lib, this->nvrtcCompileProgram, "nvrtcCompileProgram")
					&& bind(lib, this->nvrtcGetPTXSize, "nvrtcGetPTXSize") && bind(lib, this->nvrtcGetPTX, "nvrtcGetPTX")
					&& bind(lib, this->nvrtcGetProgramLogSize, "nvrtcGetProgramLogSize") && bind(lib, this->nvrtcGetProgramLog, "nvrtcGetProgramLog")
					&& bind(lib, this->nvrtcDestroyProgram, "nvrtcDestroyProgram");
			}
			void loadCublas() {
				if (!envVar("SIMPLECNN_NO_CUBLAS").empty())
					return;
#ifdef _WIN32
				void* lib = findLibrary("cublas64_", "cublasLt");
#else
				void* lib = openLibrary("libcublas.so");
				if (lib == nullptr)
					lib = findLibrary("libcublas.so", "cublasLt");
#endif
				if (lib == nullptr || !bind(lib, this->cublasCreate, "cublasCreate_v2") || !bind(lib, this->cublasSgemm, "cublasSgemm_v2"))
					return;
				if (this->cublasCreate(&this->cublas) != 0)
					this->cublas = nullptr;
			}
			// 이 GPU 용으로 커널 컴파일.
			// NVRTC 가 이 GPU 를 정확히 알면 바로 실행되는 CUBIN, 모르면 가장 가까운 PTX(드라이버가 마저 컴파일)
			bool compileKernels(std::string& error) {
				int target = this->arch;
				bool cubin = this->nvrtcGetCUBIN != nullptr;
				if (this->nvrtcGetNumSupportedArchs != nullptr && this->nvrtcGetSupportedArchs != nullptr) {
					int count = 0;
					this->nvrtcGetNumSupportedArchs(&count);
					std::vector<int> supported(count);
					this->nvrtcGetSupportedArchs(supported.data());
					target = 0;
					for (int a : supported) {
						if (a <= this->arch)
							target = (std::max)(target, a);
					}
					if (target != this->arch)
						cubin = false;
				}
				std::string option = "--gpu-architecture=compute_" + std::to_string(target);
				if (cubin)
					option = "--gpu-architecture=sm_" + std::to_string(target);
				const char* options[] = { option.c_str(), "--use_fast_math" };

				void* program = nullptr;
				this->nvrtcCreateProgram(&program, kernelSource(), "SimpleCNN.cuh", 0, nullptr, nullptr);
				if (this->nvrtcCompileProgram(program, 2, options) != 0) {
					size_t size = 0;
					this->nvrtcGetProgramLogSize(program, &size);
					std::string log(size, '\0');
					this->nvrtcGetProgramLog(program, &log[0]);
					this->nvrtcDestroyProgram(&program);
					error = "CUDA 커널 컴파일 실패 (" + option + ")\n" + log;
					return false;
				}
				size_t size = 0;
				std::string image;
				if (cubin) {
					this->nvrtcGetCUBINSize(program, &size);
					image.resize(size);
					this->nvrtcGetCUBIN(program, &image[0]);
				}
				else {
					this->nvrtcGetPTXSize(program, &size);
					image.resize(size);
					this->nvrtcGetPTX(program, &image[0]);
				}
				this->nvrtcDestroyProgram(&program);
				if (this->cuModuleLoadData(&this->module, image.data()) != 0) {
					error = "CUDA 커널 로드 실패. 그래픽 드라이버가 CUDA Toolkit 보다 오래되었습니다. 드라이버를 업데이트하세요.";
					return false;
				}
				return true;
			}
			// 모든 호출 전에 이 스레드에 GPU 컨텍스트를 연결 (다른 스레드에서 불러도 안전)
			void bindContext() { this->cuCtxSetCurrent(this->context); }
		public:
			bool start(std::string& info, std::string& error) override {
				if (!this->loadDriver()) {
					error = "NVIDIA 그래픽 드라이버(nvcuda)를 찾지 못했습니다.";
					return false;
				}
				int count = 0;
				if (this->cuInit(0) != 0 || this->cuDeviceGetCount(&count) != 0 || count == 0) {
					error = "사용할 수 있는 NVIDIA GPU 가 없습니다.";
					return false;
				}
				int device = 0;
				std::string selected = envVar("SIMPLECNN_DEVICE");
				if (!selected.empty())
					device = (std::min)((std::max)(std::atoi(selected.c_str()), 0), count - 1);
				this->check(this->cuDeviceGet(&device, device), "cuDeviceGet");
				this->check(this->cuDevicePrimaryCtxRetain(&this->context, device), "cuDevicePrimaryCtxRetain");
				this->bindContext();

				char name[256] = { 0 };
				int major = 0;
				int minor = 0;
				int driverVersion = 0;
				size_t memory = 0;
				this->cuDeviceGetName(name, 255, device);
				this->cuDeviceGetAttribute(&major, 75, device); // COMPUTE_CAPABILITY_MAJOR
				this->cuDeviceGetAttribute(&minor, 76, device); // COMPUTE_CAPABILITY_MINOR
				this->cuDriverGetVersion(&driverVersion);
				this->cuDeviceTotalMem(&memory, device);
				this->arch = major * 10 + minor;

				if (!this->loadNvrtc()) {
					error = "NVRTC(nvrtc64_*.dll)를 찾지 못했습니다. CUDA Toolkit 을 설치하세요.";
					return false;
				}
				if (!this->compileKernels(error))
					return false;
				this->loadCublas();

				info = std::string(name) + " (sm_" + std::to_string(this->arch) + ", " + formatBytes((double)memory)
					+ ", driver " + std::to_string(driverVersion / 1000) + "." + std::to_string(driverVersion % 1000 / 10);
				if (this->cublas != nullptr)
					info += ", cuBLAS";
				info += ")";
				return true;
			}
			void* alloc(size_t bytes) override {
				this->bindContext();
				Ptr ptr = 0;
				if (this->cuMemAlloc(&ptr, bytes) != 0)
					return nullptr;
				return (void*)(uintptr_t)ptr;
			}
			void release(void* ptr) override {
				this->bindContext();
				this->cuMemFree((Ptr)(uintptr_t)ptr);
			}
			void upload(void* dst, const void* src, size_t bytes) override {
				this->bindContext();
				this->check(this->cuMemcpyHtoD((Ptr)(uintptr_t)dst, src, bytes), "upload");
			}
			void download(void* dst, const void* src, size_t bytes) override {
				this->bindContext();
				this->check(this->cuMemcpyDtoH(dst, (Ptr)(uintptr_t)src, bytes), "download");
			}
			void copy(void* dst, const void* src, size_t bytes) override {
				this->bindContext();
				this->check(this->cuMemcpyDtoD((Ptr)(uintptr_t)dst, (Ptr)(uintptr_t)src, bytes), "copy");
			}
			void launch(const char* kernel, unsigned gridX, unsigned gridY, unsigned block, void** args) override {
				this->bindContext();
				void*& fn = this->functions[kernel];
				if (fn == nullptr)
					this->check(this->cuModuleGetFunction(&fn, this->module, kernel), kernel);
				this->check(this->cuLaunchKernel(fn, gridX, gridY, 1, block, 1, 1, 0, nullptr, args, nullptr), kernel);
			}
			// cuBLAS 는 열 우선이라 C^T = op(B)^T * op(A)^T 로 바꿔서 부른다.
			bool gemm(bool transA, bool transB, int M, int N, int K, float alpha, const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc) override {
				if (this->cublas == nullptr)
					return false;
				this->bindContext();
				int status = this->cublasSgemm(this->cublas, (int)transB, (int)transA, N, M, K, &alpha, B, ldb, A, lda, &beta, C, ldc);
				if (status != 0)
					throw std::runtime_error("cublasSgemm 실패: " + std::to_string(status));
				return true;
			}
		};

		class Runtime {
		private:
			std::unique_ptr<Backend> backend;
			bool tried = false;
			bool ready = false;
			std::string infoText;
			std::string errorText;
			std::mutex mutex;
			// 메모리 풀: 반납된 블록을 크기별로 보관했다가 다시 준다. (cudaMalloc 반복 방지)
			std::unordered_map<size_t, std::vector<void*>> pool;
			size_t used = 0;
			size_t peak = 0;

			static size_t roundUp(size_t bytes) {
				size_t unit = 4096;
				if (bytes >= ((size_t)1 << 20))
					unit = (size_t)1 << 20;
				return (bytes + unit - 1) / unit * unit;
			}
			void releaseCacheLocked() {
				for (auto& entry : this->pool) {
					for (void* ptr : entry.second)
						this->backend->release(ptr);
					entry.second.clear();
				}
			}
		public:
			// 프로그램 종료 순서와 상관없이 안전하도록 일부러 지우지 않는다.
			static Runtime& instance() {
				static Runtime* runtime = new Runtime();
				return *runtime;
			}
			// 다른 Backend (예: 테스트용 에뮬레이터) 를 끼운다. GPU 를 쓰기 전에 호출해야 한다.
			void setBackend(Backend* newBackend) {
				this->backend.reset(newBackend);
				this->tried = false;
				this->ready = false;
			}
			bool available() {
				std::lock_guard<std::mutex> lock(this->mutex);
				if (this->tried)
					return this->ready;
				this->tried = true;
				if (!this->backend)
					this->backend.reset(new Driver());
				try {
					this->ready = this->backend->start(this->infoText, this->errorText);
				}
				catch (const std::exception& e) {
					this->errorText = e.what();
				}
				return this->ready;
			}
			const std::string& info() const { return this->infoText; }
			const std::string& error() const { return this->errorText; }
			size_t memoryPeak() const { return this->peak; }

			void* alloc(size_t bytes) {
				std::lock_guard<std::mutex> lock(this->mutex);
				size_t size = roundUp(bytes);
				std::vector<void*>& blocks = this->pool[size];
				void* ptr = nullptr;
				if (!blocks.empty()) {
					ptr = blocks.back();
					blocks.pop_back();
				}
				if (ptr == nullptr)
					ptr = this->backend->alloc(size);
				if (ptr == nullptr) {
					this->releaseCacheLocked();
					ptr = this->backend->alloc(size);
				}
				if (ptr == nullptr)
					throw std::runtime_error("GPU 메모리 부족 (" + formatBytes((double)size) + "). 배치 크기를 줄이거나 setWorkspaceLimit 를 낮추세요.");
				this->used += size;
				this->peak = (std::max)(this->peak, this->used);
				return ptr;
			}
			void release(void* ptr, size_t bytes) {
				std::lock_guard<std::mutex> lock(this->mutex);
				size_t size = roundUp(bytes);
				this->pool[size].push_back(ptr);
				this->used -= size;
			}
			// 풀에 쌓인 블록을 실제로 GPU 에 돌려준다.
			void releaseCache() {
				std::lock_guard<std::mutex> lock(this->mutex);
				this->releaseCacheLocked();
			}
			void upload(void* dst, const void* src, size_t bytes) { this->backend->upload(dst, src, bytes); }
			void download(void* dst, const void* src, size_t bytes) { this->backend->download(dst, src, bytes); }
			void copy(void* dst, const void* src, size_t bytes) { this->backend->copy(dst, src, bytes); }

			// total 개를 처리하는 커널 실행. 인자는 커널 선언과 같은 타입(int, float, 포인터)으로 넘긴다.
			template <typename... Args>
			void launch(const char* kernel, size_t total, Args... args) {
				size_t blocks = (std::min)((total + 255) / 256, (size_t)65535);
				void* list[] = { (void*)&args... };
				this->backend->launch(kernel, (unsigned)(std::max)(blocks, (size_t)1), 1, 256, list);
			}
			template <typename... Args>
			void launchGrid(const char* kernel, unsigned gridX, unsigned gridY, Args... args) {
				void* list[] = { (void*)&args... };
				this->backend->launch(kernel, gridX, gridY, 256, list);
			}
			void gemm(bool transA, bool transB, int M, int N, int K, float alpha, const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc) {
				if (this->backend->gemm(transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc))
					return;
				this->launchGrid("gemm_kernel", (unsigned)((N + 63) / 64), (unsigned)((M + 63) / 64), (int)transA, (int)transB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
			}
		};
	}

	// ================================================================================
	// 3. Tensor
	//
	// 예전 V4D (vector<vector<vector<vector<float>>>>) 는 행마다 따로 할당돼 메모리 낭비가 크고 느리다.
	// Tensor 는 NCHW 순서의 연속 메모리 한 덩어리다.
	//  - resize 는 용량이 모자랄 때만 새로 잡는다 → 배치마다 할당이 일어나지 않는다.
	//  - CPU 는 64바이트 정렬(SIMD), GPU 는 메모리 풀에서 받아온다.
	// ================================================================================
	class Tensor {
	private:
		float* ptr = nullptr;
		size_t capacity = 0;
		Device dev = Device::CPU;

	public:
		Shape shape;

		// CPU 메모리 최대 사용량 (summary 출력용)
		static size_t& hostUsed() {
			static size_t value = 0;
			return value;
		}
		static size_t& hostPeak() {
			static size_t value = 0;
			return value;
		}

		Tensor() {}
		Tensor(const Shape& shape, Device device) { this->resize(shape, device); }
		Tensor(const Tensor&) = delete;
		Tensor& operator=(const Tensor&) = delete;
		Tensor(Tensor&& other) noexcept { *this = std::move(other); }
		Tensor& operator=(Tensor&& other) noexcept {
			if (this == &other)
				return *this;
			this->free();
			std::swap(this->ptr, other.ptr);
			std::swap(this->capacity, other.capacity);
			std::swap(this->dev, other.dev);
			std::swap(this->shape, other.shape);
			return *this;
		}
		~Tensor() { this->free(); }

		void free() {
			size_t bytes = this->capacity * sizeof(float);
			if (this->ptr != nullptr && this->dev == Device::CUDA)
				cuda::Runtime::instance().release(this->ptr, bytes);
			if (this->ptr != nullptr && this->dev == Device::CPU) {
				::operator delete[](this->ptr, std::align_val_t(64));
				hostUsed() -= bytes;
			}
			this->ptr = nullptr;
			this->capacity = 0;
			this->shape = Shape();
		}
		// 모양을 바꾼다. 용량이 모자라거나 장치가 다를 때만 새로 할당 (기존 값은 보존하지 않음)
		void resize(const Shape& newShape, Device device) {
			size_t count = newShape.size();
			if (device != this->dev || count > this->capacity) {
				this->free();
				this->dev = device;
				this->capacity = count;
				size_t bytes = count * sizeof(float);
				if (count > 0 && device == Device::CUDA)
					this->ptr = static_cast<float*>(cuda::Runtime::instance().alloc(bytes));
				if (count > 0 && device == Device::CPU) {
					this->ptr = static_cast<float*>(::operator new[](bytes, std::align_val_t(64)));
					hostUsed() += bytes;
					hostPeak() = (std::max)(hostPeak(), hostUsed());
				}
			}
			this->shape = newShape;
		}
		void resize(size_t count, Device device) { this->resize(Shape(1, 1, 1, (int)count), device); }

		float* data() { return this->ptr; }
		const float* data() const { return this->ptr; }
		size_t size() const { return this->shape.size(); }
		Device device() const { return this->dev; }

		void fill(float value) {
			if (this->size() == 0)
				return;
			if (this->dev == Device::CUDA)
				cuda::Runtime::instance().launch("fill_kernel", this->size(), this->ptr, value, (int)this->size());
			else
				std::fill(this->ptr, this->ptr + this->size(), value);
		}
		void zero() { this->fill(0.0f); }

		// 호스트 배열 → 텐서
		void upload(const float* host, size_t count) {
			if (count == 0)
				return;
			if (this->dev == Device::CUDA)
				cuda::Runtime::instance().upload(this->ptr, host, count * sizeof(float));
			else
				std::memcpy(this->ptr, host, count * sizeof(float));
		}
		// 텐서 → 호스트 배열
		void download(float* host, size_t count) const {
			if (count == 0)
				return;
			if (this->dev == Device::CUDA)
				cuda::Runtime::instance().download(host, this->ptr, count * sizeof(float));
			else
				std::memcpy(host, this->ptr, count * sizeof(float));
		}
		std::vector<float> toVector() const {
			std::vector<float> out(this->size());
			this->download(out.data(), out.size());
			return out;
		}
		// 같은 장치의 다른 텐서를 그대로 복사 (모양 포함)
		void copyFrom(const Tensor& other) {
			this->resize(other.shape, other.device());
			if (this->dev == Device::CUDA)
				cuda::Runtime::instance().copy(this->ptr, other.data(), other.size() * sizeof(float));
			else
				std::memcpy(this->ptr, other.data(), other.size() * sizeof(float));
		}
		// 값을 유지한 채 장치 이동 (CPU ↔ CUDA)
		void moveTo(Device device) {
			if (device == this->dev)
				return;
			std::vector<float> host = this->toVector();
			Shape s = this->shape;
			this->free();
			this->resize(s, device);
			this->upload(host.data(), host.size());
		}
	};

	// ================================================================================
	// 4. 연산
	//
	//   cpu::gemm      캐시 타일링 행렬곱. 합성곱과 FC 의 핵심
	//   cpu::im2col    합성곱을 행렬곱 한 번으로 바꾸기 위해 이미지 조각을 행렬로 펼친다
	//   cpu::convFFT   주파수 영역 곱셈으로 합성곱 (큰 필터에서 유리)
	//   cpu::convDirect 정의 그대로의 합성곱 (정답 확인용)
	//   gpu::*         같은 연산의 CUDA 버전 (SimpleCNN.cuh 커널 실행)
	// ================================================================================
	namespace cpu {

		inline float dot(const float* a, const float* b, int n) {
			float s[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
			int k = 0;
			for (; k + 8 <= n; k += 8) {
				for (int u = 0; u < 8; ++u)
					s[u] += a[k + u] * b[k + u];
			}
			float sum = (s[0] + s[1]) + (s[2] + s[3]) + (s[4] + s[5]) + (s[6] + s[7]);
			for (; k < n; ++k)
				sum += a[k] * b[k];
			return sum;
		}

		// 행 우선 C = alpha * op(A) * op(B) + beta * C
		// C 를 32x256 타일로 나눠 코어에 나누고, 안쪽 반복은 연속 메모리만 읽게 순서를 잡았다.
		inline void gemm(bool transA, bool transB, int M, int N, int K, float alpha, const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc) {
			const int TM = 32;
			const int TN = 256;
			int tilesN = (N + TN - 1) / TN;
			parallelFor(((M + TM - 1) / TM) * tilesN, [&](int t) {
				int i0 = (t / tilesN) * TM;
				int j0 = (t % tilesN) * TN;
				int i1 = (std::min)(i0 + TM, M);
				int j1 = (std::min)(j0 + TN, N);
				for (int i = i0; i < i1; ++i) {
					float* c = C + (size_t)i * ldc;
					for (int j = j0; j < j1; ++j)
						c[j] *= beta;
					if (beta == 0.0f)
						std::fill(c + j0, c + j1, 0.0f);
				}
				if (transB && !transA) {
					for (int i = i0; i < i1; ++i)
						for (int j = j0; j < j1; ++j)
							C[(size_t)i * ldc + j] += alpha * dot(A + (size_t)i * lda, B + (size_t)j * ldb, K);
					return;
				}
				// K 를 128 씩 끊어서, 그 사이에 읽은 B 조각이 캐시에 남아 있는 동안 모든 행이 재사용하게 한다.
				for (int k0 = 0; k0 < K; k0 += 128) {
					for (int i = i0; i < i1; ++i) {
						float* c = C + (size_t)i * ldc;
						for (int k = k0; k < (std::min)(k0 + 128, K); ++k) {
							float a = 0.0f;
							if (transA)
								a = A[(size_t)k * lda + i];
							else
								a = A[(size_t)i * lda + k];
							// ReLU 뒤에는 0 이 많아 건너뛰기만 해도 빨라진다.
							if (a == 0.0f)
								continue;
							a *= alpha;
							if (transB) {
								for (int j = j0; j < j1; ++j)
									c[j] += a * B[(size_t)j * ldb + k];
								continue;
							}
							const float* b = B + (size_t)k * ldb;
							for (int j = j0; j < j1; ++j)
								c[j] += a * b[j];
						}
					}
				}
			});
		}

		inline float activate(float x, ACTIVATION act) {
			if (act == ACTIVATION::ReLU)
				return (std::max)(x, 0.0f);
			if (act == ACTIVATION::Sigmoid)
				return 1.0f / (1.0f + std::exp(-x));
			if (act == ACTIVATION::TanH)
				return std::tanh(x);
			if (act == ACTIVATION::Leaky_ReLU && x < 0.0f)
				return kLeakySlope * x;
			if (act == ACTIVATION::ELU && x < 0.0f)
				return kEluAlpha * (std::exp(x) - 1.0f);
			return x;
		}
		// 출력 y 만으로 미분을 구한다. (활성화 전 값을 따로 저장하지 않아 메모리 절약)
		inline float activateGrad(float y, ACTIVATION act) {
			if (act == ACTIVATION::ReLU)
				return (float)(y > 0.0f);
			if (act == ACTIVATION::Sigmoid)
				return y * (1.0f - y);
			if (act == ACTIVATION::TanH)
				return 1.0f - y * y;
			if (act == ACTIVATION::Leaky_ReLU && y <= 0.0f)
				return kLeakySlope;
			if (act == ACTIVATION::ELU && y <= 0.0f)
				return y + kEluAlpha;
			return 1.0f;
		}
		inline void activate(const float* x, float* y, size_t n, ACTIVATION act) {
			for (size_t i = 0; i < n; ++i)
				y[i] = activate(x[i], act);
		}
		inline void activateBackward(const float* y, float* dy, size_t n, ACTIVATION act) {
			for (size_t i = 0; i < n; ++i)
				dy[i] *= activateGrad(y[i], act);
		}
		inline void softmax(float* y, int rows, int cols) {
			for (int r = 0; r < rows; ++r) {
				float* row = y + (size_t)r * cols;
				float maxValue = *std::max_element(row, row + cols);
				float sum = 0.0f;
				for (int j = 0; j < cols; ++j) {
					row[j] = std::exp(row[j] - maxValue);
					sum += row[j];
				}
				for (int j = 0; j < cols; ++j)
					row[j] /= sum;
			}
		}
		inline void softmaxBackward(const float* y, float* dy, int rows, int cols) {
			for (int r = 0; r < rows; ++r) {
				const float* yr = y + (size_t)r * cols;
				float* dr = dy + (size_t)r * cols;
				float d = dot(yr, dr, cols);
				for (int j = 0; j < cols; ++j)
					dr[j] = yr[j] * (dr[j] - d);
			}
		}

		// x: [C, H, W] 한 장 → col: [C*K*K, OH*OW]
		// col 의 한 행 = 필터의 한 칸이 이미지 위를 지나가며 보는 값들
		inline void im2col(const float* x, int C, int H, int W, int K, int S, int P, int OH, int OW, float* col) {
			for (int row = 0; row < C * K * K; ++row) {
				int c = row / (K * K);
				int ky = (row / K) % K;
				int kx = row % K;
				for (int oy = 0; oy < OH; ++oy) {
					float* dst = col + ((size_t)row * OH + oy) * OW;
					int iy = oy * S - P + ky;
					for (int ox = 0; ox < OW; ++ox) {
						int ix = ox * S - P + kx;
						dst[ox] = 0.0f;
						if (iy >= 0 && iy < H && ix >= 0 && ix < W)
							dst[ox] = x[((size_t)c * H + iy) * W + ix];
					}
				}
			}
		}
		// im2col 의 반대. 겹치는 위치는 더한다. dx 는 0 으로 시작해야 한다.
		inline void col2im(const float* col, int C, int H, int W, int K, int S, int P, int OH, int OW, float* dx) {
			for (int row = 0; row < C * K * K; ++row) {
				int c = row / (K * K);
				int ky = (row / K) % K;
				int kx = row % K;
				for (int oy = 0; oy < OH; ++oy) {
					const float* src = col + ((size_t)row * OH + oy) * OW;
					int iy = oy * S - P + ky;
					for (int ox = 0; ox < OW; ++ox) {
						int ix = ox * S - P + kx;
						if (iy >= 0 && iy < H && ix >= 0 && ix < W)
							dx[((size_t)c * H + iy) * W + ix] += src[ox];
					}
				}
			}
		}

		// 반복형 radix-2 FFT (제자리 계산, n 은 2의 거듭제곱). inverse 는 1/n 을 곱하지 않는다.
		inline void fft(float* re, float* im, int n, bool inverse) {
			for (int i = 1, j = 0; i < n; ++i) {
				int bit = n >> 1;
				for (; j & bit; bit >>= 1)
					j ^= bit;
				j ^= bit;
				if (i < j) {
					std::swap(re[i], re[j]);
					std::swap(im[i], im[j]);
				}
			}
			double sign = -1.0;
			if (inverse)
				sign = 1.0;
			for (int len = 2; len <= n; len <<= 1) {
				double angle = sign * 2.0 * 3.14159265358979323846 / len;
				for (int k = 0; k < len / 2; ++k) {
					float wr = (float)std::cos(angle * k);
					float wi = (float)std::sin(angle * k);
					for (int a = k; a < n; a += len) {
						int b = a + len / 2;
						float tr = re[b] * wr - im[b] * wi;
						float ti = re[b] * wi + im[b] * wr;
						re[b] = re[a] - tr;
						im[b] = im[a] - ti;
						re[a] += tr;
						im[a] += ti;
					}
				}
			}
		}
		// 2차원 FFT = 행마다 FFT 한 뒤 열마다 FFT. tmp 는 rows*2 크기
		inline void fft2d(float* re, float* im, int rows, int cols, bool inverse, float* tmp) {
			for (int r = 0; r < rows; ++r)
				fft(re + (size_t)r * cols, im + (size_t)r * cols, cols, inverse);
			for (int c = 0; c < cols; ++c) {
				for (int r = 0; r < rows; ++r) {
					tmp[r] = re[(size_t)r * cols + c];
					tmp[rows + r] = im[(size_t)r * cols + c];
				}
				fft(tmp, tmp + rows, rows, inverse);
				for (int r = 0; r < rows; ++r) {
					re[(size_t)r * cols + c] = tmp[r];
					im[(size_t)r * cols + c] = tmp[rows + r];
				}
			}
		}
		inline int nextPow2(int v) {
			int p = 1;
			while (p < v)
				p <<= 1;
			return p;
		}
		// FFT 합성곱에 필요한 메모리 (필터 스펙트럼 + 코어별 버퍼)
		inline size_t fftBytes(int Cin, int Cout, int H, int W, int P) {
			size_t plane = (size_t)nextPow2(H + 2 * P) * nextPow2(W + 2 * P);
			return (((size_t)Cout * Cin + (Cin + 2) * (size_t)threadCount()) * plane * 2) * sizeof(float);
		}

		// FFT 합성곱.
		// 원리: 합성곱은 주파수 영역에서 곱셈이 된다. CNN 의 합성곱은 상관(correlation) 이므로
		// 필터를 뒤집어서 넣고, 결과에서 (K-1) 만큼 밀린 위치부터 읽는다.
		inline void convFFT(const float* x, int N, int Cin, int H, int W, const float* weight, const float* bias, int Cout, int K, int S, int P, int OH, int OW, ACTIVATION act, float* y, std::vector<float>& buffer) {
			int PH = nextPow2(H + 2 * P);
			int PW = nextPow2(W + 2 * P);
			size_t plane = (size_t)PH * PW;
			size_t lane = ((size_t)Cin + 1) * plane * 2 + (size_t)PH * 2;
			buffer.resize((size_t)Cout * Cin * plane * 2 + lane * threadCount());
			float* filterRe = buffer.data();
			float* filterIm = filterRe + (size_t)Cout * Cin * plane;
			float* lanes = filterIm + (size_t)Cout * Cin * plane;

			// 필터 스펙트럼 (배치 전체가 같이 쓴다)
			parallelFor(Cout * Cin, [&](int index) {
				float* re = filterRe + (size_t)index * plane;
				float* im = filterIm + (size_t)index * plane;
				std::fill(re, re + plane, 0.0f);
				std::fill(im, im + plane, 0.0f);
				const float* w = weight + (size_t)index * K * K;
				for (int ky = 0; ky < K; ++ky)
					for (int kx = 0; kx < K; ++kx)
						re[ky * PW + kx] = w[(K - 1 - ky) * K + (K - 1 - kx)];
				std::vector<float> tmp(PH * 2);
				fft2d(re, im, PH, PW, false, tmp.data());
			});

			parallelFor(N, [&](int n) {
				float* inRe = lanes + lane * threadIndex();
				float* inIm = inRe + (size_t)Cin * plane;
				float* accRe = inIm + (size_t)Cin * plane;
				float* accIm = accRe + plane;
				float* tmp = accIm + plane;
				for (int c = 0; c < Cin; ++c) {
					float* re = inRe + (size_t)c * plane;
					float* im = inIm + (size_t)c * plane;
					std::fill(re, re + plane, 0.0f);
					std::fill(im, im + plane, 0.0f);
					for (int iy = 0; iy < H; ++iy)
						std::memcpy(re + (size_t)(iy + P) * PW + P, x + (((size_t)n * Cin + c) * H + iy) * W, W * sizeof(float));
					fft2d(re, im, PH, PW, false, tmp);
				}
				for (int co = 0; co < Cout; ++co) {
					std::fill(accRe, accRe + plane, 0.0f);
					std::fill(accIm, accIm + plane, 0.0f);
					for (int c = 0; c < Cin; ++c) {
						const float* ar = inRe + (size_t)c * plane;
						const float* ai = inIm + (size_t)c * plane;
						const float* br = filterRe + ((size_t)co * Cin + c) * plane;
						const float* bi = filterIm + ((size_t)co * Cin + c) * plane;
						for (size_t i = 0; i < plane; ++i) {
							accRe[i] += ar[i] * br[i] - ai[i] * bi[i];
							accIm[i] += ar[i] * bi[i] + ai[i] * br[i];
						}
					}
					fft2d(accRe, accIm, PH, PW, true, tmp);
					float* dst = y + ((size_t)n * Cout + co) * OH * OW;
					for (int oy = 0; oy < OH; ++oy)
						for (int ox = 0; ox < OW; ++ox)
							dst[oy * OW + ox] = activate(accRe[(size_t)(oy * S + K - 1) * PW + ox * S + K - 1] / plane + bias[co], act);
				}
			});
		}

		inline void convDirect(const float* x, int N, int Cin, int H, int W, const float* weight, const float* bias, int Cout, int K, int S, int P, int OH, int OW, ACTIVATION act, float* y) {
			parallelFor(N * Cout, [&](int index) {
				int n = index / Cout;
				int co = index % Cout;
				for (int oy = 0; oy < OH; ++oy) {
					for (int ox = 0; ox < OW; ++ox) {
						float sum = bias[co];
						for (int c = 0; c < Cin; ++c) {
							for (int ky = 0; ky < K; ++ky) {
								for (int kx = 0; kx < K; ++kx) {
									int iy = oy * S - P + ky;
									int ix = ox * S - P + kx;
									if (iy >= 0 && iy < H && ix >= 0 && ix < W)
										sum += x[(((size_t)n * Cin + c) * H + iy) * W + ix] * weight[(((size_t)co * Cin + c) * K + ky) * K + kx];
								}
							}
						}
						y[((size_t)index * OH + oy) * OW + ox] = activate(sum, act);
					}
				}
			});
		}

		// 풀링. index 에는 Max/Min 이 고른 위치를 저장한다. (역전파 때 그 자리로만 기울기를 보낸다)
		inline void poolForward(const float* x, float* y, float* index, int planes, int H, int W, int OH, int OW, int size, int stride, POOLING mode) {
			parallelFor(planes, [&](int p) {
				const float* src = x + (size_t)p * H * W;
				for (int o = 0; o < OH * OW; ++o) {
					int y0 = (o / OW) * stride;
					int x0 = (o % OW) * stride;
					int best = y0 * W + x0;
					float sum = 0.0f;
					for (int ky = 0; ky < size; ++ky) {
						for (int kx = 0; kx < size; ++kx) {
							int pos = (y0 + ky) * W + x0 + kx;
							sum += src[pos];
							if (mode == POOLING::Max && src[pos] > src[best])
								best = pos;
							if (mode == POOLING::Min && src[pos] < src[best])
								best = pos;
						}
					}
					y[(size_t)p * OH * OW + o] = src[best];
					if (mode == POOLING::Average)
						y[(size_t)p * OH * OW + o] = sum / (size * size);
					index[(size_t)p * OH * OW + o] = (float)best;
				}
			});
		}
		inline void poolBackward(const float* dy, const float* index, float* dx, int planes, int H, int W, int OH, int OW, int size, int stride, POOLING mode) {
			parallelFor(planes, [&](int p) {
				float* dst = dx + (size_t)p * H * W;
				std::fill(dst, dst + (size_t)H * W, 0.0f);
				for (int o = 0; o < OH * OW; ++o) {
					float g = dy[(size_t)p * OH * OW + o];
					if (mode != POOLING::Average) {
						dst[(int)index[(size_t)p * OH * OW + o]] += g;
						continue;
					}
					for (int ky = 0; ky < size; ++ky)
						for (int kx = 0; kx < size; ++kx)
							dst[((o / OW) * stride + ky) * W + (o % OW) * stride + kx] += g / (size * size);
				}
			});
		}
	}

	namespace gpu {
		inline cuda::Runtime& rt() { return cuda::Runtime::instance(); }

		inline void activate(const float* x, float* y, int n, ACTIVATION act) { rt().launch("act_forward_kernel", n, x, y, n, (int)act); }
		inline void activateBackward(const float* y, float* dy, int n, ACTIVATION act) { rt().launch("act_backward_kernel", n, y, dy, n, (int)act); }
		inline void softmax(float* y, int rows, int cols) { rt().launch("softmax_kernel", rows, y, rows, cols); }
		inline void softmaxBackward(const float* y, float* dy, int rows, int cols) { rt().launch("softmax_backward_kernel", rows, y, dy, rows, cols); }
		inline void biasActivate(float* y, const float* bias, int rows, int cols, ACTIVATION act) { rt().launch("bias_act_kernel", (size_t)rows * cols, y, bias, rows, cols, (int)act); }
		inline void colSum(const float* src, float* dst, int rows, int cols) { rt().launch("colsum_kernel", cols, src, dst, rows, cols); }
		inline void rowSum(const float* src, float* dst, int rows, int cols) { rt().launchGrid("rowsum_kernel", (unsigned)rows, 1, src, dst, rows, cols); }
		inline void im2col(const float* x, float* col, int C, int H, int W, int K, int S, int P, int OH, int OW, int nb) {
			rt().launch("im2col_kernel", (size_t)C * K * K * nb * OH * OW, x, col, C, H, W, K, S, P, OH, OW, nb);
		}
		inline void col2im(const float* col, float* dx, int C, int H, int W, int K, int S, int P, int OH, int OW, int nb) {
			rt().launch("col2im_kernel", (size_t)nb * C * H * W, col, dx, C, H, W, K, S, P, OH, OW, nb);
		}
		inline void convOutput(const float* src, const float* bias, float* dst, int Cout, int nb, int P, ACTIVATION act) {
			rt().launch("conv_output_kernel", (size_t)Cout * nb * P, src, bias, dst, Cout, nb, P, (int)act);
		}
		inline void convGrad(const float* src, float* dst, int Cout, int nb, int P) { rt().launch("conv_grad_kernel", (size_t)Cout * nb * P, src, dst, Cout, nb, P); }
		inline void padForward(const float* x, float* y, int planes, int H, int W, int P) {
			rt().launch("pad_forward_kernel", (size_t)planes * (H + 2 * P) * (W + 2 * P), x, y, planes, H, W, P);
		}
		inline void padBackward(const float* dy, float* dx, int planes, int H, int W, int P) { rt().launch("pad_backward_kernel", (size_t)planes * H * W, dy, dx, planes, H, W, P); }
		inline void poolForward(const float* x, float* y, float* index, int planes, int H, int W, int OH, int OW, int size, int stride, POOLING mode) {
			rt().launch("pool_forward_kernel", (size_t)planes * OH * OW, x, y, index, planes, H, W, OH, OW, size, stride, (int)mode);
		}
		inline void poolBackward(const float* dy, const float* index, float* dx, int planes, int H, int W, int OH, int OW, int size, int stride, POOLING mode) {
			rt().launch("pool_backward_kernel", (size_t)planes * OH * OW, dy, index, dx, planes, H, W, OH, OW, size, stride, (int)mode);
		}
		inline void maxoutForward(const float* z, float* y, float* index, int n) { rt().launch("maxout_forward_kernel", n, z, y, index, n); }
		inline void maxoutBackward(const float* dy, const float* index, float* dz, int n) { rt().launch("maxout_backward_kernel", n, dy, index, dz, n); }
	}

	// ================================================================================
	// 5. 레이어 (블럭)
	//
	// 모든 블럭은 같은 모양이다.
	//   build    : 앞 블럭의 출력 모양을 받아 자기 출력 모양을 정하고 가중치를 만든다
	//   forward  : x → y
	//   backward : dy → dx (+ 자기 가중치의 기울기)
	// 새 레이어도 Layer 를 상속해 이 세 개만 만들면 CNN 에 그대로 끼워진다.
	// ================================================================================

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

	// 레이어들이 같이 쓰는 실행 정보
	struct Context {
		Device device = Device::CPU;
		bool needInputGrad = true;  // 첫 레이어는 입력 쪽 기울기가 필요 없다
		bool fusedOutput = false;   // 손실 함수가 마지막 Softmax/Sigmoid 미분까지 합쳐서 넘겨준 경우
		CONV_ALGO convAlgo = CONV_ALGO::Auto;
		size_t workspaceLimit = (size_t)512 << 20;
		bool verbose = true;
		Tensor workspace;           // im2col 임시 버퍼. 모든 Conv 가 하나를 같이 쓴다.
	};

	class Layer {
	public:
		LAYER layerType = LAYER::None;
		ACTIVATION act = ACTIVATION::None;
		Shape inShape;
		Shape outShape;

		virtual ~Layer() {}
		virtual std::string name() const = 0;
		virtual std::string config() const = 0; // 저장 파일의 한 줄 (예: "Conv 3 16 1 1 ReLU")
		virtual std::string detail() const { return ""; }
		virtual Shape build(const Shape& in, std::mt19937& rng) = 0;
		virtual void forward(const Tensor& x, Tensor& y, Context& ctx) = 0;
		virtual void backward(const Tensor& x, const Tensor& y, Tensor& dy, Tensor& dx, Context& ctx) = 0;
		virtual std::vector<Param*> params() { return {}; }

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
		// 가중치 초기화: ReLU 계열은 He, 나머지는 Xavier 균등분포. 이미 불러온 값이 있으면 그대로 둔다.
		static void initParam(Param& p, size_t count, int fanIn, int fanOut, ACTIVATION act, std::mt19937& rng, bool zero) {
			if (p.value.size() == count)
				return;
			float limit = std::sqrt(6.0f / (fanIn + fanOut));
			if (act == ACTIVATION::ReLU || act == ACTIVATION::Leaky_ReLU || act == ACTIVATION::ELU)
				limit = std::sqrt(6.0f / fanIn);
			std::uniform_real_distribution<float> dist(-limit, limit);
			std::vector<float> host(count, 0.0f);
			if (!zero) {
				for (size_t i = 0; i < count; ++i)
					host[i] = dist(rng);
			}
			p.value.resize(count, Device::CPU);
			p.value.upload(host.data(), count);
			p.grad.resize(count, Device::CPU);
		}
		// 활성화 미분 (Softmax / Maxout / None 은 각자 따로 처리)
		static void activationBackward(const Tensor& y, Tensor& dy, ACTIVATION act, Context& ctx) {
			if (ctx.fusedOutput || act == ACTIVATION::None || act == ACTIVATION::Softmax || act == ACTIVATION::Maxout)
				return;
			if (ctx.device == Device::CUDA)
				gpu::activateBackward(y.data(), dy.data(), (int)y.size(), act);
			else
				cpu::activateBackward(y.data(), dy.data(), y.size(), act);
		}
	};

	// -------------------------------------------------------------------------------- Conv
	// 합성곱. 이미지 위로 필터를 밀면서 특징을 뽑는다.
	// 계산은 im2col 로 이미지를 행렬로 펼친 뒤 행렬곱 한 번: y = W[필터, C*K*K] x col[C*K*K, 위치]
	class Conv : public Layer {
	private:
		int kernel;
		int filters;
		int stride;
		int padding;
		CONV_ALGO algo;
		Param weight;               // [filters, Cin*K*K]
		Param bias;                 // [filters]
		std::vector<float> scratch; // CPU: 코어별 기울기 누적 / FFT 스펙트럼

		int rows() const { return this->inShape.c * this->kernel * this->kernel; } // im2col 행 수
		int plane() const { return this->outShape.h * this->outShape.w; }          // 출력 위치 수
		bool pointwise() const { return this->kernel == 1 && this->stride == 1 && this->padding == 0; }
		void im2col(const float* x, float* col) const {
			cpu::im2col(x, this->inShape.c, this->inShape.h, this->inShape.w, this->kernel, this->stride, this->padding, this->outShape.h, this->outShape.w, col);
		}
		// 코어마다 im2col 버퍼를 따로 줘도 메모리 한도 안이면 이미지 단위로 병렬, 아니면 행렬곱 안에서 병렬
		int lanes(size_t extraFloats, const Context& ctx) const {
			size_t bytes = ((size_t)this->rows() * this->plane() + extraFloats) * sizeof(float) * threadCount();
			if (bytes <= ctx.workspaceLimit)
				return threadCount();
			return 1;
		}
		template <typename F>
		static void forEachSample(int N, int laneCount, F fn) {
			if (laneCount > 1)
				parallelFor(N, [&](int n) { fn(n, threadIndex()); });
			else
				for (int n = 0; n < N; ++n)
					fn(n, 0);
		}

		void forwardIm2col(const Tensor& x, Tensor& y, Context& ctx) {
			int K = this->rows();
			int P = this->plane();
			int laneCount = this->lanes(0, ctx);
			ctx.workspace.resize((size_t)K * P * laneCount, Device::CPU);
			forEachSample(x.shape.n, laneCount, [&](int n, int lane) {
				const float* xn = x.data() + n * this->inShape.sample();
				float* yn = y.data() + n * this->outShape.sample();
				const float* col = xn; // 1x1 합성곱은 펼칠 필요가 없다
				if (!this->pointwise()) {
					float* buffer = ctx.workspace.data() + (size_t)K * P * lane;
					this->im2col(xn, buffer);
					col = buffer;
				}
				cpu::gemm(false, false, this->filters, P, K, 1.0f, this->weight.value.data(), K, col, P, 0.0f, yn, P);
				for (int co = 0; co < this->filters; ++co)
					for (int p = 0; p < P; ++p)
						yn[co * P + p] = cpu::activate(yn[co * P + p] + this->bias.value.data()[co], this->act);
			});
		}
		void forwardCpu(const Tensor& x, Tensor& y, Context& ctx, CONV_ALGO method) {
			const Shape& in = this->inShape;
			if (method == CONV_ALGO::FFT)
				cpu::convFFT(x.data(), x.shape.n, in.c, in.h, in.w, this->weight.value.data(), this->bias.value.data(), this->filters, this->kernel, this->stride, this->padding, this->outShape.h, this->outShape.w, this->act, y.data(), this->scratch);
			else if (method == CONV_ALGO::Direct)
				cpu::convDirect(x.data(), x.shape.n, in.c, in.h, in.w, this->weight.value.data(), this->bias.value.data(), this->filters, this->kernel, this->stride, this->padding, this->outShape.h, this->outShape.w, this->act, y.data());
			else
				this->forwardIm2col(x, y, ctx);
		}
		// Auto: 실제 배치로 im2col 과 FFT 를 한 번씩 재보고 빠른 쪽을 계속 쓴다.
		CONV_ALGO measure(const Tensor& x, Tensor& y, Context& ctx) {
			if (this->kernel < 3 || cpu::fftBytes(this->inShape.c, this->filters, this->inShape.h, this->inShape.w, this->padding) > ctx.workspaceLimit)
				return CONV_ALGO::Im2col;
			Timer timer;
			this->forwardCpu(x, y, ctx, CONV_ALGO::FFT);
			double fftTime = timer.seconds();
			timer.reset();
			this->forwardCpu(x, y, ctx, CONV_ALGO::Im2col); // 마지막에 im2col 로 다시 계산해 y 를 채운다
			double gemmTime = timer.seconds();
			std::vector<float>().swap(this->scratch);
			CONV_ALGO best = CONV_ALGO::Im2col;
			if (fftTime < gemmTime)
				best = CONV_ALGO::FFT;
			if (ctx.verbose)
				std::printf("  [Conv %dx%d %d->%d] im2col %.2fms, FFT %.2fms -> %s\n", this->kernel, this->kernel, this->inShape.c, this->filters, gemmTime * 1000, fftTime * 1000, toString(best));
			return best;
		}

		// GPU: 여러 장을 한 번에 펼쳐 큰 행렬곱 하나로 계산. 작업 버퍼가 한도를 넘으면 배치를 나눠서 처리
		int chunkSize(int N, Context& ctx) {
			size_t perSample = ((size_t)this->rows() + this->filters) * this->plane();
			int chunk = (int)(std::min)((size_t)N, (std::max)((size_t)1, ctx.workspaceLimit / (perSample * sizeof(float))));
			ctx.workspace.resize(perSample * chunk, Device::CUDA);
			return chunk;
		}
		void forwardGpu(const Tensor& x, Tensor& y, Context& ctx) {
			int N = x.shape.n;
			int K = this->rows();
			int P = this->plane();
			int chunk = this->chunkSize(N, ctx);
			float* col = ctx.workspace.data();
			float* tmp = col + (size_t)K * P * chunk;
			for (int b = 0; b < N; b += chunk) {
				int count = (std::min)(chunk, N - b);
				gpu::im2col(x.data() + b * this->inShape.sample(), col, this->inShape.c, this->inShape.h, this->inShape.w, this->kernel, this->stride, this->padding, this->outShape.h, this->outShape.w, count);
				gpu::rt().gemm(false, false, this->filters, count * P, K, 1.0f, this->weight.value.data(), K, col, count * P, 0.0f, tmp, count * P);
				gpu::convOutput(tmp, this->bias.value.data(), y.data() + b * this->outShape.sample(), this->filters, count, P, this->act);
			}
		}
		void backwardGpu(const Tensor& x, Tensor& dy, Tensor& dx, Context& ctx) {
			int N = x.shape.n;
			int K = this->rows();
			int P = this->plane();
			int chunk = this->chunkSize(N, ctx);
			float* col = ctx.workspace.data();
			float* tmp = col + (size_t)K * P * chunk;
			this->bias.grad.zero();
			for (int b = 0; b < N; b += chunk) {
				int count = (std::min)(chunk, N - b);
				float beta = (float)(b > 0); // 첫 조각은 덮어쓰고, 다음 조각부터 누적
				gpu::convGrad(dy.data() + b * this->outShape.sample(), tmp, this->filters, count, P);
				gpu::im2col(x.data() + b * this->inShape.sample(), col, this->inShape.c, this->inShape.h, this->inShape.w, this->kernel, this->stride, this->padding, this->outShape.h, this->outShape.w, count);
				gpu::rt().gemm(false, true, this->filters, K, count * P, 1.0f, tmp, count * P, col, count * P, beta, this->weight.grad.data(), K);
				gpu::rowSum(tmp, this->bias.grad.data(), this->filters, count * P);
				if (!ctx.needInputGrad)
					continue;
				gpu::rt().gemm(true, false, K, count * P, this->filters, 1.0f, this->weight.value.data(), K, tmp, count * P, 0.0f, col, count * P);
				gpu::col2im(col, dx.data() + b * this->inShape.sample(), this->inShape.c, this->inShape.h, this->inShape.w, this->kernel, this->stride, this->padding, this->outShape.h, this->outShape.w, count);
			}
		}
		// CPU 역전파: 이미지마다 dW += dy * col^T, dx = col2im(W^T * dy). 코어별로 dW 를 따로 모은 뒤 합친다.
		void backwardCpu(const Tensor& x, Tensor& dy, Tensor& dx, Context& ctx) {
			int K = this->rows();
			int P = this->plane();
			size_t gradSize = (size_t)this->filters * K + this->filters;
			int laneCount = this->lanes(gradSize, ctx);
			ctx.workspace.resize((size_t)K * P * laneCount, Device::CPU);
			this->scratch.assign(gradSize * laneCount, 0.0f);
			const float* W = this->weight.value.data();
			forEachSample(x.shape.n, laneCount, [&](int n, int lane) {
				const float* xn = x.data() + n * this->inShape.sample();
				const float* dyn = dy.data() + n * this->outShape.sample();
				float* col = ctx.workspace.data() + (size_t)K * P * lane;
				float* gW = this->scratch.data() + gradSize * lane;
				float* gB = gW + (size_t)this->filters * K;
				const float* src = xn;
				if (!this->pointwise()) {
					this->im2col(xn, col);
					src = col;
				}
				cpu::gemm(false, true, this->filters, K, P, 1.0f, dyn, P, src, P, 1.0f, gW, K);
				for (int co = 0; co < this->filters; ++co)
					for (int p = 0; p < P; ++p)
						gB[co] += dyn[co * P + p];
				if (!ctx.needInputGrad)
					return;
				float* dxn = dx.data() + n * this->inShape.sample();
				if (this->pointwise()) {
					cpu::gemm(true, false, K, P, this->filters, 1.0f, W, K, dyn, P, 0.0f, dxn, P);
					return;
				}
				cpu::gemm(true, false, K, P, this->filters, 1.0f, W, K, dyn, P, 0.0f, col, P);
				std::fill(dxn, dxn + this->inShape.sample(), 0.0f);
				cpu::col2im(col, this->inShape.c, this->inShape.h, this->inShape.w, this->kernel, this->stride, this->padding, this->outShape.h, this->outShape.w, dxn);
			});
			float* gW = this->weight.grad.data();
			float* gB = this->bias.grad.data();
			std::fill(gW, gW + (size_t)this->filters * K, 0.0f);
			std::fill(gB, gB + this->filters, 0.0f);
			for (int lane = 0; lane < laneCount; ++lane) {
				const float* src = this->scratch.data() + gradSize * lane;
				for (size_t i = 0; i < (size_t)this->filters * K; ++i)
					gW[i] += src[i];
				for (int co = 0; co < this->filters; ++co)
					gB[co] += src[(size_t)this->filters * K + co];
			}
		}
	public:
		// filterRowCol: 필터 한 변 (3 이면 3x3), filterSize: 필터 개수 (= 출력 채널 수)
		Conv(int filterRowCol = 3, int filterSize = 8, int stride = 1, int paddingSize = 1, ACTIVATION act = ACTIVATION::ReLU, CONV_ALGO algo = CONV_ALGO::Auto) {
			this->layerType = LAYER::Conv;
			this->kernel = filterRowCol;
			this->filters = filterSize;
			this->stride = (std::max)(1, stride);
			this->padding = (std::max)(0, paddingSize);
			this->act = act;
			this->algo = algo;
		}
		std::string name() const override { return "Conv"; }
		std::string config() const override {
			return "Conv " + std::to_string(this->kernel) + " " + std::to_string(this->filters) + " " + std::to_string(this->stride) + " " + std::to_string(this->padding) + " " + toString(this->act);
		}
		std::string detail() const override {
			std::string text = std::to_string(this->kernel) + "x" + std::to_string(this->kernel) + " s" + std::to_string(this->stride) + " p" + std::to_string(this->padding) + " " + toString(this->act);
			if (this->algo != CONV_ALGO::Auto)
				text += std::string(" [") + toString(this->algo) + "]";
			return text;
		}
		Shape build(const Shape& in, std::mt19937& rng) override {
			this->inShape = in;
			int oh = (in.h + 2 * this->padding - this->kernel) / this->stride + 1;
			int ow = (in.w + 2 * this->padding - this->kernel) / this->stride + 1;
			if (oh <= 0 || ow <= 0)
				throw std::runtime_error("Conv: 입력 " + in.str() + " 이 필터보다 작습니다. Pooling 을 줄이거나 padding 을 늘리세요.");
			this->outShape = Shape(1, this->filters, oh, ow);
			int fanIn = this->rows();
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
			if (this->algo == CONV_ALGO::Auto)
				this->algo = ctx.convAlgo;
			if (this->algo == CONV_ALGO::Auto) {
				this->algo = this->measure(x, y, ctx);
				return;
			}
			this->forwardCpu(x, y, ctx, this->algo);
		}
		void backward(const Tensor& x, const Tensor& y, Tensor& dy, Tensor& dx, Context& ctx) override {
			activationBackward(y, dy, this->act, ctx);
			if (ctx.needInputGrad)
				dx.resize(x.shape, ctx.device);
			if (ctx.device == Device::CUDA)
				this->backwardGpu(x, dy, dx, ctx);
			else
				this->backwardCpu(x, dy, dx, ctx);
		}
		std::vector<Param*> params() override { return { &this->weight, &this->bias }; }
		void setAlgorithm(CONV_ALGO method) { this->algo = method; }
	};

	// -------------------------------------------------------------------------------- Pooling
	// 크기 줄이기. size x size 창에서 최대/평균/최소 하나를 고른다.
	class Pooling : public Layer {
	private:
		POOLING mode;
		int size;
		int stride;
		Tensor index; // Max/Min 이 고른 위치 (0/1 마스크 전체를 들고 있지 않는다)
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
		std::string config() const override { return std::string("Pool ") + toString(this->mode) + " " + std::to_string(this->size) + " " + std::to_string(this->stride); }
		std::string detail() const override { return std::string(toString(this->mode)) + " " + std::to_string(this->size) + "x" + std::to_string(this->size); }
		Shape build(const Shape& in, std::mt19937&) override {
			if (in.h < this->size || in.w < this->size)
				throw std::runtime_error("Pooling: 입력 " + in.str() + " 이 풀링 크기보다 작습니다. Pooling 레이어 수를 줄이세요.");
			this->inShape = in;
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

	// -------------------------------------------------------------------------------- Padding
	// 테두리에 0 을 채운다.
	class Padding : public Layer {
	private:
		int pad;

		// forward: 작은 그림 → 큰 그림 안쪽, backward: 큰 그림 안쪽 → 작은 그림
		void copy(const float* src, float* dst, int planes, bool forward) {
			int H = this->inShape.h;
			int W = this->inShape.w;
			int OW = W + 2 * this->pad;
			int OH = H + 2 * this->pad;
			parallelFor(planes, [&](int p) {
				for (int iy = 0; iy < H; ++iy) {
					size_t inner = ((size_t)p * H + iy) * W;
					size_t outer = ((size_t)p * OH + iy + this->pad) * OW + this->pad;
					if (forward)
						std::memcpy(dst + outer, src + inner, W * sizeof(float));
					else
						std::memcpy(dst + inner, src + outer, W * sizeof(float));
				}
			});
		}
	public:
		Padding(int paddingSize = 1) {
			this->layerType = LAYER::Padding;
			this->pad = (std::max)(0, paddingSize);
		}
		std::string name() const override { return "Padding"; }
		std::string config() const override { return "Padding " + std::to_string(this->pad); }
		std::string detail() const override { return std::to_string(this->pad); }
		Shape build(const Shape& in, std::mt19937&) override {
			this->inShape = in;
			this->outShape = Shape(1, in.c, in.h + 2 * this->pad, in.w + 2 * this->pad);
			return this->outShape;
		}
		void forward(const Tensor& x, Tensor& y, Context& ctx) override {
			y.resize(Shape(x.shape.n, this->outShape.c, this->outShape.h, this->outShape.w), ctx.device);
			int planes = x.shape.n * x.shape.c;
			if (ctx.device == Device::CUDA) {
				gpu::padForward(x.data(), y.data(), planes, this->inShape.h, this->inShape.w, this->pad);
				return;
			}
			y.zero();
			this->copy(x.data(), y.data(), planes, true);
		}
		void backward(const Tensor& x, const Tensor& y, Tensor& dy, Tensor& dx, Context& ctx) override {
			if (!ctx.needInputGrad)
				return;
			dx.resize(x.shape, ctx.device);
			int planes = x.shape.n * x.shape.c;
			if (ctx.device == Device::CUDA)
				gpu::padBackward(dy.data(), dx.data(), planes, this->inShape.h, this->inShape.w, this->pad);
			else
				this->copy(dy.data(), dx.data(), planes, false);
		}
	};

	// -------------------------------------------------------------------------------- Activation
	// 활성화 함수만 따로 쓰고 싶을 때. Softmax 는 이미지 한 장 전체에 적용된다.
	class Activation : public Layer {
	public:
		Activation(ACTIVATION act = ACTIVATION::ReLU) {
			this->layerType = LAYER::Act;
			this->act = act;
		}
		std::string name() const override { return "Activation"; }
		std::string config() const override { return std::string("Act ") + toString(this->act); }
		std::string detail() const override { return toString(this->act); }
		Shape build(const Shape& in, std::mt19937&) override {
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
					gpu::softmax(y.data(), rows, cols);
				return;
			}
			cpu::activate(x.data(), y.data(), x.size(), this->act);
			if (this->act == ACTIVATION::Softmax)
				cpu::softmax(y.data(), rows, cols);
		}
		void backward(const Tensor& x, const Tensor& y, Tensor& dy, Tensor& dx, Context& ctx) override {
			if (!ctx.needInputGrad)
				return;
			activationBackward(y, dy, this->act, ctx);
			if (this->act == ACTIVATION::Softmax && !ctx.fusedOutput && ctx.device == Device::CUDA)
				gpu::softmaxBackward(y.data(), dy.data(), y.shape.n, (int)y.shape.sample());
			if (this->act == ACTIVATION::Softmax && !ctx.fusedOutput && ctx.device == Device::CPU)
				cpu::softmaxBackward(y.data(), dy.data(), y.shape.n, (int)y.shape.sample());
			dx.copyFrom(dy);
		}
	};

	// -------------------------------------------------------------------------------- Flatten
	// 펼치기. FullyConnected 가 입력을 알아서 펼치므로 생략해도 되지만, 예전 코드처럼 써도 되게 남겨둔다.
	class Flatten : public Layer {
	public:
		Flatten() { this->layerType = LAYER::Flatten; }
		std::string name() const override { return "Flatten"; }
		std::string config() const override { return "Flatten"; }
		Shape build(const Shape& in, std::mt19937&) override {
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

	// -------------------------------------------------------------------------------- FullyConnected
	// 완전연결층: y = act(x * W + b).   x [n, 입력], W [입력, 유닛]
	// Maxout 은 유닛마다 선형 출력 2개를 만들어 큰 쪽을 쓴다.
	class FullyConnected : public Layer {
	private:
		int units;
		Param weight;  // [입력, 유닛 * pieces]
		Param bias;    // [유닛 * pieces]
		Tensor z;      // Maxout 전 선형 출력
		Tensor zGrad;
		Tensor index;  // Maxout 이 고른 쪽 (0 또는 1)

		int inputs() const { return (int)this->inShape.sample(); }
		int outputs() const {
			if (this->act == ACTIVATION::Maxout)
				return this->units * 2;
			return this->units;
		}
		ACTIVATION linearAct() const {
			if (this->act == ACTIVATION::Softmax || this->act == ACTIVATION::Maxout)
				return ACTIVATION::None;
			return this->act;
		}
	public:
		FullyConnected(int denseSize = 10, ACTIVATION act = ACTIVATION::ReLU) {
			this->layerType = LAYER::FC;
			this->units = (std::max)(1, denseSize);
			this->act = act;
		}
		std::string name() const override { return "FullyConnected"; }
		std::string config() const override { return "FC " + std::to_string(this->units) + " " + toString(this->act); }
		std::string detail() const override { return toString(this->act); }
		Shape build(const Shape& in, std::mt19937& rng) override {
			this->inShape = in;
			this->outShape = Shape(1, this->units, 1, 1);
			initParam(this->weight, (size_t)this->inputs() * this->outputs(), this->inputs(), this->outputs(), this->act, rng, false);
			initParam(this->bias, (size_t)this->outputs(), this->inputs(), this->outputs(), this->act, rng, true);
			return this->outShape;
		}
		void forward(const Tensor& x, Tensor& y, Context& ctx) override {
			int rows = x.shape.n;
			int in = this->inputs();
			int out = this->outputs();
			y.resize(Shape(rows, this->units, 1, 1), ctx.device);
			Tensor* linear = &y;
			if (this->act == ACTIVATION::Maxout) {
				this->z.resize(Shape(rows, out, 1, 1), ctx.device);
				this->index.resize(y.shape, ctx.device);
				linear = &this->z;
			}
			if (ctx.device == Device::CUDA) {
				gpu::rt().gemm(false, false, rows, out, in, 1.0f, x.data(), in, this->weight.value.data(), out, 0.0f, linear->data(), out);
				gpu::biasActivate(linear->data(), this->bias.value.data(), rows, out, this->linearAct());
				if (this->act == ACTIVATION::Softmax)
					gpu::softmax(y.data(), rows, out);
				if (this->act == ACTIVATION::Maxout)
					gpu::maxoutForward(this->z.data(), y.data(), this->index.data(), (int)y.size());
				return;
			}
			cpu::gemm(false, false, rows, out, in, 1.0f, x.data(), in, this->weight.value.data(), out, 0.0f, linear->data(), out);
			for (int r = 0; r < rows; ++r)
				for (int j = 0; j < out; ++j)
					linear->data()[(size_t)r * out + j] = cpu::activate(linear->data()[(size_t)r * out + j] + this->bias.value.data()[j], this->linearAct());
			if (this->act == ACTIVATION::Softmax)
				cpu::softmax(y.data(), rows, out);
			if (this->act != ACTIVATION::Maxout)
				return;
			for (size_t i = 0; i < y.size(); ++i) {
				int best = (int)(this->z.data()[2 * i + 1] > this->z.data()[2 * i]);
				y.data()[i] = this->z.data()[2 * i + best];
				this->index.data()[i] = (float)best;
			}
		}
		void backward(const Tensor& x, const Tensor& y, Tensor& dy, Tensor& dx, Context& ctx) override {
			int rows = x.shape.n;
			int in = this->inputs();
			int out = this->outputs();
			bool cuda = ctx.device == Device::CUDA;
			// g = 선형 출력(활성화 전)에 대한 기울기
			Tensor* g = &dy;
			activationBackward(y, dy, this->act, ctx);
			if (this->act == ACTIVATION::Softmax && !ctx.fusedOutput && cuda)
				gpu::softmaxBackward(y.data(), dy.data(), rows, out);
			if (this->act == ACTIVATION::Softmax && !ctx.fusedOutput && !cuda)
				cpu::softmaxBackward(y.data(), dy.data(), rows, out);
			if (this->act == ACTIVATION::Maxout) {
				this->zGrad.resize(this->z.shape, ctx.device);
				g = &this->zGrad;
				if (cuda) {
					gpu::maxoutBackward(dy.data(), this->index.data(), g->data(), (int)dy.size());
				}
				else {
					g->zero();
					for (size_t i = 0; i < dy.size(); ++i)
						g->data()[2 * i + (int)this->index.data()[i]] = dy.data()[i];
				}
			}
			// dW = x^T * g,  db = g 의 열 합,  dx = g * W^T
			if (ctx.needInputGrad)
				dx.resize(x.shape, ctx.device);
			this->bias.grad.zero();
			if (cuda) {
				gpu::rt().gemm(true, false, in, out, rows, 1.0f, x.data(), in, g->data(), out, 0.0f, this->weight.grad.data(), out);
				gpu::colSum(g->data(), this->bias.grad.data(), rows, out);
				if (ctx.needInputGrad)
					gpu::rt().gemm(false, true, rows, in, out, 1.0f, g->data(), out, this->weight.value.data(), out, 0.0f, dx.data(), in);
				return;
			}
			cpu::gemm(true, false, in, out, rows, 1.0f, x.data(), in, g->data(), out, 0.0f, this->weight.grad.data(), out);
			for (int r = 0; r < rows; ++r)
				for (int j = 0; j < out; ++j)
					this->bias.grad.data()[j] += g->data()[(size_t)r * out + j];
			if (ctx.needInputGrad)
				cpu::gemm(false, true, rows, in, out, 1.0f, g->data(), out, this->weight.value.data(), out, 0.0f, dx.data(), in);
		}
		std::vector<Param*> params() override { return { &this->weight, &this->bias }; }
	};

	// 저장 파일의 한 줄로 레이어를 다시 만든다. (load 에서 사용)
	inline Layer* createLayer(const std::string& line) {
		std::istringstream in(line);
		std::string type;
		std::string text;
		int a = 0;
		int b = 0;
		int c = 0;
		int d = 0;
		in >> type;
		if (type == "Conv") {
			in >> a >> b >> c >> d >> text;
			return new Conv(a, b, c, d, parseActivation(text));
		}
		if (type == "Pool") {
			in >> text >> a >> b;
			POOLING mode = POOLING::Max;
			if (text == "Average")
				mode = POOLING::Average;
			if (text == "Min")
				mode = POOLING::Min;
			return new Pooling(mode, a, b);
		}
		if (type == "Padding") {
			in >> a;
			return new Padding(a);
		}
		if (type == "Act") {
			in >> text;
			return new Activation(parseActivation(text));
		}
		if (type == "Flatten")
			return new Flatten();
		if (type == "FC") {
			in >> a >> text;
			return new FullyConnected(a, parseActivation(text));
		}
		throw std::runtime_error("알 수 없는 레이어: " + line);
	}

	// ================================================================================
	// 6. 옵티마이저
	// 상태 버퍼(m, v)는 그 옵티마이저가 실제로 쓸 때만 만든다. (SGD 0벌, Momentum 1벌, Adam 2벌)
	// ================================================================================
	class Optimizer {
	private:
		OPTIMIZER type = OPTIMIZER::Adam;
		long long step = 0;

		// SimpleCNN.cuh optimizer_kernel 의 type 번호
		int code() const {
			const int codes[] = { 0, 0, 1, 2, 3, 4, 5 }; // GD, Mini_SGD, Momentum, Adagrad, RMSProp, Adam, Nadam
			return codes[(int)this->type];
		}
	public:
		float learningRate = 0.001f;
		float beta1 = 0.9f;
		float beta2 = 0.999f;
		float epsilon = 1e-7f;

		void setType(OPTIMIZER newType) {
			this->type = newType;
			this->step = 0;
			this->learningRate = 0.001f;
			if (this->code() <= 2)
				this->learningRate = 0.01f; // SGD, Momentum, Adagrad
		}

		void update(const std::vector<Param*>& params, Device device) {
			++this->step;
			int t = this->code();
			bool useM = t == 1 || t == 4 || t == 5;
			bool useV = t >= 2;
			float b2 = this->beta2;
			if (t == 3)
				b2 = 0.9f; // RMSProp 의 감쇠율
			float bc1 = 1.0f - (float)std::pow(this->beta1, (double)this->step);
			float bc2 = 1.0f - (float)std::pow(b2, (double)this->step);
			float bc1Next = 1.0f - (float)std::pow(this->beta1, (double)this->step + 1);
			for (Param* p : params) {
				if (useM && p->m.size() != p->size()) {
					p->m.resize(p->value.shape, device);
					p->m.zero();
				}
				if (useV && p->v.size() != p->size()) {
					p->v.resize(p->value.shape, device);
					p->v.zero();
				}
				int n = (int)p->size();
				float* w = p->value.data();
				const float* g = p->grad.data();
				float* m = w; // 안 쓰는 상태는 아무 유효한 주소나 넘긴다
				float* v = w;
				if (useM)
					m = p->m.data();
				if (useV)
					v = p->v.data();
				if (device == Device::CUDA) {
					cuda::Runtime::instance().launch("optimizer_kernel", n, w, g, m, v, n, t, this->learningRate, this->beta1, b2, this->epsilon, bc1, bc2, bc1Next);
					continue;
				}
				float lr = this->learningRate;
				for (int i = 0; i < n; ++i) {
					if (t == 0) {
						w[i] -= lr * g[i];
					}
					else if (t == 1) {
						m[i] = this->beta1 * m[i] - lr * g[i];
						w[i] += m[i];
					}
					else if (t == 2) {
						v[i] += g[i] * g[i];
						w[i] -= lr * g[i] / (std::sqrt(v[i]) + this->epsilon);
					}
					else if (t == 3) {
						v[i] = b2 * v[i] + (1.0f - b2) * g[i] * g[i];
						w[i] -= lr * g[i] / (std::sqrt(v[i]) + this->epsilon);
					}
					else {
						m[i] = this->beta1 * m[i] + (1.0f - this->beta1) * g[i];
						v[i] = b2 * v[i] + (1.0f - b2) * g[i] * g[i];
						float mHat = m[i] / bc1;
						if (t == 5)
							mHat = this->beta1 * m[i] / bc1Next + (1.0f - this->beta1) * g[i] / bc1;
						w[i] -= lr * mHat / (std::sqrt(v[i] / bc2) + this->epsilon);
					}
				}
			}
		}
	};

	// ================================================================================
	// 7. 이미지 / 데이터셋
	// ================================================================================

	// 경로 문자열은 OS 기본 인코딩 그대로 (Windows 는 CP949 라 한글 폴더명도 그대로 동작)
	inline std::filesystem::path toPath(const std::string& text) { return std::filesystem::path(text); }
	inline std::string pathString(const std::filesystem::path& path) { return path.string(); }

	inline std::vector<uint8_t> readFile(const std::filesystem::path& path) {
		std::ifstream file(path, std::ios::binary);
		return std::vector<uint8_t>((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
	}

	// 이미지 한 장. 픽셀은 HWC (가로줄 순서, RGB) 0~255
	struct Image {
		int width = 0;
		int height = 0;
		int channels = 0;
		std::vector<uint8_t> pixels;

		Image() {}
		Image(int width, int height, int channels) : width(width), height(height), channels(channels), pixels((size_t)width * height * channels, 0) {}

		bool empty() const { return this->pixels.empty(); }
		uint8_t& at(int x, int y, int c) { return this->pixels[((size_t)y * this->width + x) * this->channels + c]; }
		uint8_t at(int x, int y, int c) const { return this->pixels[((size_t)y * this->width + x) * this->channels + c]; }

		// BMP, PGM, PPM 은 직접 읽고, stb_image.h 가 있으면 JPG / PNG 등도 읽는다. 실패하면 빈 이미지
		static Image load(const std::filesystem::path& path) {
			std::vector<uint8_t> bytes = readFile(path);
			if (bytes.size() > 2 && bytes[0] == 'B' && bytes[1] == 'M')
				return decodeBMP(bytes);
			if (bytes.size() > 2 && bytes[0] == 'P' && (bytes[1] == '5' || bytes[1] == '6'))
				return decodePNM(bytes);
			Image image;
#ifdef SIMPLECNN_STB
			int w = 0;
			int h = 0;
			int c = 0;
			unsigned char* data = nullptr;
			if (!bytes.empty())
				data = stbi_load_from_memory(bytes.data(), (int)bytes.size(), &w, &h, &c, 0);
			if (data != nullptr) {
				image = Image(w, h, c);
				std::memcpy(image.pixels.data(), data, image.pixels.size());
				stbi_image_free(data);
			}
#endif
			return image;
		}

		// 24비트 BMP 로 저장
		bool saveBMP(const std::string& path) const {
			Image rgb = this->withChannels(3);
			int rowBytes = (rgb.width * 3 + 3) / 4 * 4;
			uint32_t dataSize = (uint32_t)rowBytes * rgb.height;
			uint8_t header[54] = { 'B', 'M' };
			writeLE(header + 2, 54 + dataSize);
			writeLE(header + 10, 54);
			writeLE(header + 14, 40);
			writeLE(header + 18, (uint32_t)rgb.width);
			writeLE(header + 22, (uint32_t)rgb.height);
			header[26] = 1;
			header[28] = 24;
			writeLE(header + 34, dataSize);
			std::ofstream file(toPath(path), std::ios::binary);
			file.write(reinterpret_cast<const char*>(header), 54);
			std::vector<uint8_t> row(rowBytes, 0);
			for (int y = rgb.height - 1; y >= 0; --y) {
				for (int x = 0; x < rgb.width; ++x)
					for (int c = 0; c < 3; ++c)
						row[x * 3 + c] = rgb.at(x, y, 2 - c); // BMP 는 BGR 순서
				file.write(reinterpret_cast<const char*>(row.data()), rowBytes);
			}
			return (bool)file;
		}

		// 양선형 보간 크기 변환
		Image resized(int newWidth, int newHeight) const {
			if (newWidth == this->width && newHeight == this->height)
				return *this;
			Image out(newWidth, newHeight, this->channels);
			for (int y = 0; y < newHeight; ++y) {
				float fy = (std::max)(0.0f, (y + 0.5f) * this->height / newHeight - 0.5f);
				int y0 = (std::min)((int)fy, this->height - 1);
				int y1 = (std::min)(y0 + 1, this->height - 1);
				float wy = fy - y0;
				for (int x = 0; x < newWidth; ++x) {
					float fx = (std::max)(0.0f, (x + 0.5f) * this->width / newWidth - 0.5f);
					int x0 = (std::min)((int)fx, this->width - 1);
					int x1 = (std::min)(x0 + 1, this->width - 1);
					float wx = fx - x0;
					for (int c = 0; c < this->channels; ++c) {
						float top = this->at(x0, y0, c) * (1.0f - wx) + this->at(x1, y0, c) * wx;
						float bottom = this->at(x0, y1, c) * (1.0f - wx) + this->at(x1, y1, c) * wx;
						out.at(x, y, c) = (uint8_t)(top * (1.0f - wy) + bottom * wy + 0.5f);
					}
				}
			}
			return out;
		}
		// 채널 수 변환 (흑백 ↔ 컬러, 알파 제거)
		Image withChannels(int target) const {
			if (target == this->channels)
				return *this;
			Image out(this->width, this->height, target);
			for (int y = 0; y < this->height; ++y) {
				for (int x = 0; x < this->width; ++x) {
					uint8_t rgb[3] = { this->at(x, y, 0), this->at(x, y, 0), this->at(x, y, 0) };
					if (this->channels >= 3) {
						rgb[1] = this->at(x, y, 1);
						rgb[2] = this->at(x, y, 2);
					}
					for (int c = 0; c < target; ++c)
						out.at(x, y, c) = 255; // 알파
					for (int c = 0; c < target && c < 3; ++c)
						out.at(x, y, c) = rgb[c];
					if (target == 1)
						out.at(x, y, 0) = (uint8_t)(0.299f * rgb[0] + 0.587f * rgb[1] + 0.114f * rgb[2] + 0.5f);
				}
			}
			return out;
		}
		// 모델 입력 모양 (C, H, W) 에 맞춰 0~1 float, CHW 순서로 채운다.
		void toCHW(float* out, int c, int h, int w) const {
			Image fitted = this->withChannels(c).resized(w, h);
			for (int ch = 0; ch < c; ++ch)
				for (int y = 0; y < h; ++y)
					for (int x = 0; x < w; ++x)
						out[((size_t)ch * h + y) * w + x] = fitted.at(x, y, ch) / 255.0f;
		}
		// CHW 0~1 float → 이미지
		static Image fromCHW(const float* data, int c, int h, int w) {
			Image out(w, h, c);
			for (int ch = 0; ch < c; ++ch)
				for (int y = 0; y < h; ++y)
					for (int x = 0; x < w; ++x)
						out.at(x, y, ch) = (uint8_t)(std::min)(255.0f, (std::max)(0.0f, data[((size_t)ch * h + y) * w + x] * 255.0f + 0.5f));
			return out;
		}
	private:
		static void writeLE(uint8_t* p, uint32_t v) {
			for (int i = 0; i < 4; ++i)
				p[i] = (uint8_t)(v >> (8 * i));
		}
		static uint32_t readLE(const uint8_t* p) {
			return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
		}
		// BMP: 비압축 8비트(팔레트) / 24비트 / 32비트
		static Image decodeBMP(const std::vector<uint8_t>& bytes) {
			if (bytes.size() < 54)
				return Image();
			uint32_t offset = readLE(&bytes[10]);
			uint32_t headerSize = readLE(&bytes[14]);
			int w = (int)readLE(&bytes[18]);
			int h = (int)readLE(&bytes[22]);
			int bits = bytes[28];
			int step = bits / 8;
			int rowBytes = (w * bits / 8 + 3) / 4 * 4;
			bool topDown = h < 0;
			h = std::abs(h);
			if ((bits != 8 && bits != 24 && bits != 32) || bytes.size() < offset + (size_t)rowBytes * h)
				return Image();
			Image image(w, h, 3);
			const uint8_t* palette = &bytes[14 + headerSize];
			for (int y = 0; y < h; ++y) {
				int row = h - 1 - y;
				if (topDown)
					row = y;
				const uint8_t* src = &bytes[offset + (size_t)row * rowBytes];
				for (int x = 0; x < w; ++x) {
					const uint8_t* bgr = src + x * step;
					if (bits == 8)
						bgr = palette + src[x] * 4;
					for (int c = 0; c < 3; ++c)
						image.at(x, y, c) = bgr[2 - c];
				}
			}
			return image;
		}
		// PGM(P5, 흑백) / PPM(P6, 컬러)
		static Image decodePNM(const std::vector<uint8_t>& bytes) {
			std::string text(bytes.begin(), bytes.begin() + (std::min)(bytes.size(), (size_t)256));
			std::istringstream in(text);
			std::string magic;
			int values[3] = { 0, 0, 0 };
			in >> magic;
			for (int i = 0; i < 3; ++i) {
				in >> std::ws;
				while (in.peek() == '#')
					in.ignore(256, '\n') >> std::ws;
				in >> values[i];
			}
			Image image(values[0], values[1], 1);
			if (magic == "P6")
				image = Image(values[0], values[1], 3);
			size_t start = (size_t)in.tellg() + 1;
			if (values[2] > 255 || bytes.size() < start + image.pixels.size())
				return Image();
			std::memcpy(image.pixels.data(), &bytes[start], image.pixels.size());
			return image;
		}
	};

	inline bool isImageFile(const std::filesystem::path& path) {
		std::string ext = path.extension().string();
		std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char ch) { return (char)std::tolower(ch); });
		const char* known[] = { ".bmp", ".pgm", ".ppm", ".jpg", ".jpeg", ".png", ".tga", ".gif" };
		return std::find(std::begin(known), std::end(known), ext) != std::end(known);
	}

	// 데이터셋
	//  - 이미지는 0~255 uint8 로 보관하고 배치를 만들 때만 0~1 float 로 바꾼다. (float 대비 메모리 1/4)
	//  - streaming 모드는 파일 경로만 들고 있다가 배치마다 디스크에서 읽는다. (큰 데이터셋용)
	class Dataset {
	private:
		Shape sampleShape;
		std::vector<uint8_t> bytes;
		std::vector<float> floats;     // fromVectors 로 float 를 직접 넣은 경우
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
		// 샘플 i 를 다른 데이터셋 끝에 붙인다. (subset / split 에서 사용)
		void appendTo(Dataset& out, size_t i) const {
			size_t n = this->sampleShape.sample();
			out.labelList.push_back(this->labelList[i]);
			if (!this->files.empty())
				out.files.push_back(this->files[i]);
			else if (!this->floats.empty())
				out.floats.insert(out.floats.end(), this->floats.begin() + i * n, this->floats.begin() + (i + 1) * n);
			else
				out.bytes.insert(out.bytes.end(), this->bytes.begin() + i * n, this->bytes.begin() + (i + 1) * n);
		}
		Dataset emptyCopy() const {
			Dataset out;
			out.sampleShape = this->sampleShape;
			out.floatScale = this->floatScale;
			out.classNames = this->classNames;
			return out;
		}
	public:
		std::vector<std::string> classNames;

		size_t size() const { return this->labelList.size(); }
		bool empty() const { return this->labelList.empty(); }
		Shape shape() const { return this->sampleShape; }
		float label(size_t i) const { return this->labelList[i]; }
		int classCount() const { return (int)*std::max_element(this->labelList.begin(), this->labelList.end()) + 1; }
		size_t memoryBytes() const { return this->bytes.size() + (this->floats.size() + this->labelList.size()) * sizeof(float); }

		// 이미지 한 장 추가. 첫 이미지 (또는 정해둔 크기) 에 맞춰 크기와 채널을 통일한다.
		void add(const Image& image, float label) {
			if (this->sampleShape.c == 0)
				this->sampleShape = Shape(1, image.channels, image.height, image.width);
			Image fitted = image.withChannels(this->sampleShape.c).resized(this->sampleShape.w, this->sampleShape.h);
			for (int c = 0; c < fitted.channels; ++c)
				for (int y = 0; y < fitted.height; ++y)
					for (int x = 0; x < fitted.width; ++x)
						this->bytes.push_back(fitted.at(x, y, c));
			this->labelList.push_back(label);
		}
		void addFile(const std::filesystem::path& file, float label, bool streaming) {
			if (streaming) {
				this->files.push_back(file);
				this->labelList.push_back(label);
				return;
			}
			Image image = Image::load(file);
			if (image.empty())
				std::cerr << "[SimpleCNN] 이미지를 읽지 못해 건너뜀: " << pathString(file) << "\n";
			else
				this->add(image, label);
		}

		// 샘플 i 를 CHW 0~1 float 로 out 에 채운다.
		void sample(size_t i, float* out) const {
			size_t n = this->sampleShape.sample();
			if (!this->files.empty()) {
				Image image = Image::load(this->files[i]);
				std::fill(out, out + n, 0.0f);
				if (!image.empty())
					image.toCHW(out, this->sampleShape.c, this->sampleShape.h, this->sampleShape.w);
				return;
			}
			if (!this->floats.empty()) {
				for (size_t k = 0; k < n; ++k)
					out[k] = this->floats[i * n + k] * this->floatScale;
				return;
			}
			for (size_t k = 0; k < n; ++k)
				out[k] = this->bytes[i * n + k] / 255.0f;
		}
		// order[begin] ~ order[begin+count-1] 번 샘플로 배치를 만든다.
		void batch(const std::vector<int>& order, size_t begin, int count, float* out, float* outLabels) const {
			parallelFor(count, [&](int k) {
				this->sample(order[begin + k], out + this->sampleShape.sample() * k);
				outLabels[k] = this->labelList[order[begin + k]];
			});
		}
		Dataset subset(size_t begin, size_t count) const {
			Dataset out = this->emptyCopy();
			for (size_t i = begin; i < begin + count && i < this->size(); ++i)
				this->appendTo(out, i);
			return out;
		}
		// 섞어서 (1 - testRatio) 는 학습용, testRatio 는 테스트용으로 나눈다.
		std::pair<Dataset, Dataset> split(float testRatio, unsigned seed = 42) const {
			std::vector<size_t> order(this->size());
			for (size_t i = 0; i < order.size(); ++i)
				order[i] = i;
			std::mt19937 rng(seed);
			std::shuffle(order.begin(), order.end(), rng);
			size_t testCount = (size_t)(this->size() * testRatio);
			std::pair<Dataset, Dataset> result(this->emptyCopy(), this->emptyCopy());
			for (size_t k = 0; k < order.size(); ++k) {
				if (k < order.size() - testCount)
					this->appendTo(result.first, order[k]);
				else
					this->appendTo(result.second, order[k]);
			}
			return result;
		}

		// 예전 방식: V4D [개수][채널][세로][가로] + 라벨. 값이 1 보다 크면 0~255 로 보고 255 로 나눈다.
		static Dataset fromVectors(const V4D& images, const V1D& labels) {
			Dataset data;
			if (images.empty())
				return data;
			data.sampleShape = Shape(1, (int)images[0].size(), (int)images[0][0].size(), (int)images[0][0][0].size());
			float maxValue = 0.0f;
			for (size_t i = 0; i < images.size(); ++i) {
				for (int c = 0; c < data.sampleShape.c; ++c) {
					for (int y = 0; y < data.sampleShape.h; ++y) {
						for (int x = 0; x < data.sampleShape.w; ++x) {
							data.floats.push_back(images[i][c][y][x]);
							maxValue = (std::max)(maxValue, images[i][c][y][x]);
						}
					}
				}
				float label = 0.0f;
				if (i < labels.size())
					label = labels[i];
				data.labelList.push_back(label);
			}
			if (maxValue > 1.5f)
				data.floatScale = 1.0f / 255.0f;
			return data;
		}

		// MNIST / Fashion-MNIST (압축 푼 IDX 파일). limit > 0 이면 앞에서부터 그만큼만
		static Dataset loadMNIST(const std::string& imagesFile, const std::string& labelsFile, size_t limit = 0) {
			std::ifstream images(toPath(imagesFile), std::ios::binary);
			std::ifstream labels(toPath(labelsFile), std::ios::binary);
			if (!images || !labels || readBE(images) != 2051 || readBE(labels) != 2049)
				throw std::runtime_error("MNIST 파일이 아니거나 열 수 없습니다: " + imagesFile);
			size_t count = (std::min)(readBE(images), readBE(labels));
			if (limit > 0)
				count = (std::min)(count, limit);
			Dataset data;
			int rows = (int)readBE(images);
			int cols = (int)readBE(images);
			data.sampleShape = Shape(1, 1, rows, cols);
			data.bytes.resize(count * rows * cols);
			images.read(reinterpret_cast<char*>(data.bytes.data()), (std::streamsize)data.bytes.size());
			std::vector<uint8_t> raw(count);
			labels.read(reinterpret_cast<char*>(raw.data()), (std::streamsize)count);
			data.labelList.assign(raw.begin(), raw.end());
			for (int d = 0; d < 10; ++d)
				data.classNames.push_back(std::to_string(d));
			return data;
		}

		// 폴더 하나 = 클래스 하나.  root/고양이/*.jpg, root/강아지/*.jpg ...
		static Dataset loadFolder(const std::string& root, int width, int height, int channels = 3, bool streaming = false) {
			Dataset data;
			data.sampleShape = Shape(1, channels, height, width);
			std::vector<std::filesystem::path> dirs;
			std::error_code ec;
			for (const auto& entry : std::filesystem::directory_iterator(toPath(root), ec)) {
				if (entry.is_directory(ec))
					dirs.push_back(entry.path());
			}
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

		// 폴더의 이미지 (이름순) + 라벨 CSV.  CSV 는 줄마다 "라벨" 또는 "파일명,라벨"
		static Dataset loadFiles(const std::string& imageDir, const std::string& labelCsv, int width, int height, int channels = 3, bool streaming = false) {
			Dataset data;
			data.sampleShape = Shape(1, channels, height, width);
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
				if (comma != std::string::npos)
					data.addFile(toPath(imageDir) / toPath(line.substr(0, comma)), std::stof(line.substr(comma + 1)), streaming);
				else if (index < images.size())
					data.addFile(images[index++], std::stof(line), streaming);
			}
			return data;
		}
	};

	// ================================================================================
	// 8. CNN 모델
	//   add      블럭 쌓기
	//   compile  옵티마이저 / 손실 함수 / 장치 (CPU, CUDA) 정하기
	//   fit      학습      evaluate  정확도      predict  예측
	//   save     저장      load      불러오기 (레이어를 다시 쌓지 않아도 된다)
	// ================================================================================
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
		Tensor gradA;                  // 기울기 버퍼는 두 개를 번갈아 쓴다 (레이어 수와 상관없이 2개)
		Tensor gradB;
		Tensor labelTensor;            // GPU 손실 계산용
		Tensor lossTensor;
		Tensor predTensor;
		std::vector<float> hostLoss;
		std::vector<float> hostPred;
		Context ctx;
		Optimizer optimizer;
		LOSS lossType = LOSS::CategoricalCrossentropy;
		Device requested = Device::CPU;
		bool compiled = false;
		bool built = false;
		bool learningRateSet = false;
		Shape inputShape;
		std::mt19937 rng = std::mt19937(1234);
		Dataset ownData;               // 예전 방식 CNN(images, labels) 로 받은 데이터

		// CUDA 를 요청했는데 쓸 수 없으면 이유를 출력하고 CPU 로
		Device resolve(Device device) {
			if (device == Device::CPU)
				return Device::CPU;
			cuda::Runtime& runtime = cuda::Runtime::instance();
			if (!runtime.available()) {
				std::cerr << "[SimpleCNN] CUDA 를 사용할 수 없어 CPU 로 실행합니다.\n  이유: " << runtime.error() << "\n";
				return Device::CPU;
			}
			if (this->ctx.verbose)
				std::printf("[SimpleCNN] GPU: %s\n", runtime.info().c_str());
			return Device::CUDA;
		}
		std::vector<Param*> allParams() {
			std::vector<Param*> all;
			for (auto& layer : this->layers)
				for (Param* p : layer->params())
					all.push_back(p);
			return all;
		}
		// 입력 모양을 처음 알게 됐을 때 블럭마다 출력 모양을 정하고 가중치를 만든다.
		void build(const Shape& sample) {
			if (!this->compiled)
				this->compile(OPTIMIZER::Adam, LOSS::CategoricalCrossentropy);
			if (this->built && (sample.c != this->inputShape.c || sample.h != this->inputShape.h || sample.w != this->inputShape.w))
				throw std::runtime_error("입력 크기 " + sample.str() + " 가 모델 입력 " + this->inputShape.str() + " 과 다릅니다.");
			if (this->built)
				return;
			if (this->layers.empty())
				throw std::runtime_error("레이어가 없습니다. add() 로 레이어를 먼저 쌓아주세요.");
			this->inputShape = Shape(1, sample.c, sample.h, sample.w);
			Shape s = this->inputShape;
			for (auto& layer : this->layers) {
				s = layer->build(s, this->rng);
				layer->toDevice(this->ctx.device);
			}
			this->outputs.resize(this->layers.size());
			this->built = true;
		}
		int outputUnits() const { return (int)this->layers.back()->outShape.sample(); }
		bool crossEntropy() const { return this->lossType == LOSS::CategoricalCrossentropy || this->lossType == LOSS::SparseCategoricalCrossentropy; }
		// Softmax + CrossEntropy, Sigmoid + BinaryCrossentropy 는 합친 기울기 (y - t) 가 정확하고 빠르다.
		bool fusedLoss() const {
			ACTIVATION last = this->layers.back()->act;
			if (last == ACTIVATION::Softmax && this->crossEntropy() && this->outputUnits() > 1)
				return true;
			return last == ACTIVATION::Sigmoid && this->lossType == LOSS::BinaryCrossentropy;
		}
		void checkLabels(const Dataset& data) {
			int units = this->outputUnits();
			if (this->crossEntropy() && units == 1)
				throw std::runtime_error("출력 유닛이 1개면 LOSS::BinaryCrossentropy 를 쓰거나 마지막 FullyConnected 를 클래스 수만큼 늘리세요.");
			if (units > 1 && data.classCount() > units)
				throw std::runtime_error("라벨 종류(" + std::to_string(data.classCount()) + ")가 마지막 레이어 유닛 수(" + std::to_string(units) + ")보다 많습니다.");
		}
		void forward(const float* batch, int count) {
			this->input.resize(Shape(count, this->inputShape.c, this->inputShape.h, this->inputShape.w), this->ctx.device);
			this->input.upload(batch, this->input.size());
			const Tensor* x = &this->input;
			for (size_t i = 0; i < this->layers.size(); ++i) {
				this->layers[i]->forward(*x, this->outputs[i], this->ctx);
				x = &this->outputs[i];
			}
		}
		// CPU 손실 계산 (SimpleCNN.cuh 의 loss_kernel 과 같은 계산)
		void lossCpu(const float* y, const float* labels, int rows, int cols, int type, int fused) {
			for (int r = 0; r < rows; ++r) {
				const float* yr = y + (size_t)r * cols;
				float* gr = this->gradA.data() + (size_t)r * cols;
				float total = 0.0f;
				for (int j = 0; j < cols; ++j) {
					float t = (float)(cols > 1 && j == (int)labels[r]);
					if (cols == 1)
						t = labels[r];
					float v = yr[j];
					float g = 0.0f;
					if (type <= 1) {
						total += (v - t) * (v - t);
						g = 2.0f * (v - t) / cols;
					}
					else if (type == 2) {
						total -= (t * std::log(v + 1e-7f) + (1.0f - t) * std::log(1.0f - v + 1e-7f)) / cols;
						g = (v - t) / ((v + 1e-7f) * (1.0f - v + 1e-7f)) / cols;
						if (fused)
							g = (v - t) / cols;
					}
					else {
						if (t > 0.0f)
							total -= std::log(v + 1e-7f);
						g = -t / (v + 1e-7f);
						if (fused)
							g = v - t;
					}
					gr[j] = g / rows;
				}
				this->hostLoss[r] = total;
				this->hostPred[r] = (float)argmax(yr, cols);
			}
		}
		// 손실값을 돌려주고 출력 기울기를 gradA 에 만든다. 맞춘 개수는 correct 에 더한다.
		float computeLoss(const float* labels, int rows, int& correct) {
			Tensor& y = this->outputs.back();
			int cols = (int)y.shape.sample();
			int type = (int)this->lossType;
			int fused = (int)this->fusedLoss();
			this->gradA.resize(y.shape, this->ctx.device);
			this->hostLoss.resize(rows);
			this->hostPred.resize(rows);
			if (this->ctx.device == Device::CUDA) {
				this->labelTensor.resize((size_t)rows, Device::CUDA);
				this->lossTensor.resize((size_t)rows, Device::CUDA);
				this->predTensor.resize((size_t)rows, Device::CUDA);
				this->labelTensor.upload(labels, rows);
				cuda::Runtime::instance().launch("loss_kernel", rows, (const float*)y.data(), (const float*)this->labelTensor.data(), this->gradA.data(), this->lossTensor.data(), this->predTensor.data(), rows, cols, type, fused);
				this->lossTensor.download(this->hostLoss.data(), rows);
				this->predTensor.download(this->hostPred.data(), rows);
			}
			else {
				this->lossCpu(y.data(), labels, rows, cols, type, fused);
			}
			double sum = 0.0;
			for (int r = 0; r < rows; ++r) {
				sum += this->hostLoss[r];
				correct += (int)((int)this->hostPred[r] == (int)labels[r]);
			}
			float value = (float)(sum / rows);
			if (type <= 1)
				value /= cols; // 평균 제곱 오차
			if (type == 1) {
				// RMSE = sqrt(MSE). 기울기는 MSE 기울기 / (2 * RMSE)
				value = std::sqrt(value);
				float scale = 1.0f / (2.0f * value + 1e-7f);
				if (this->ctx.device == Device::CUDA)
					cuda::Runtime::instance().launch("scale_kernel", this->gradA.size(), this->gradA.data(), scale, (int)this->gradA.size());
				else
					for (size_t i = 0; i < this->gradA.size(); ++i)
						this->gradA.data()[i] *= scale;
			}
			return value;
		}
		void backward() {
			Tensor* dy = &this->gradA;
			Tensor* dx = &this->gradB;
			int last = (int)this->layers.size() - 1;
			for (int i = last; i >= 0; --i) {
				const Tensor* x = &this->input;
				if (i > 0)
					x = &this->outputs[i - 1];
				this->ctx.needInputGrad = i > 0;
				this->ctx.fusedOutput = i == last && this->fusedLoss();
				this->layers[i]->backward(*x, this->outputs[i], *dy, *dx, this->ctx);
				std::swap(dy, dx);
			}
			this->ctx.needInputGrad = true;
			this->ctx.fusedOutput = false;
		}
		static int argmax(const float* row, int cols) {
			if (cols == 1)
				return (int)(row[0] > 0.5f);
			return (int)(std::max_element(row, row + cols) - row);
		}
		static void progress(int epoch, int epochs, size_t done, size_t total, double loss, double acc, double seconds) {
			int filled = (int)(30 * done / total);
			std::string bar = std::string(filled, '=') + std::string(30 - filled, ' ');
			std::printf("\repoch %d/%d [%s] %3d%%  loss %.4f  acc %.2f%%  %.1fs ", epoch, epochs, bar.c_str(), (int)(100 * done / total), loss, acc * 100.0, seconds);
			std::fflush(stdout);
		}
	public:
		CNN(Device device = Device::CPU) { this->requested = device; }
		// 예전 방식: 데이터를 생성자로 넘기고 fit(epochs, batchSize)
		CNN(const V4D& images, const V1D& labels, Device device = Device::CPU) {
			this->requested = device;
			this->ownData = Dataset::fromVectors(images, labels);
			if (images.size() != labels.size())
				std::cerr << "[SimpleCNN] 이미지 수(" << images.size() << ")와 라벨 수(" << labels.size() << ")가 다릅니다.\n";
		}
		CNN(const CNN&) = delete;
		CNN& operator=(const CNN&) = delete;

		// 블럭 쌓기. new 로 넘긴 레이어는 CNN 이 알아서 지운다.
		CNN& add(Layer* layer) {
			if (this->built)
				throw std::runtime_error("학습을 시작한 뒤에는 레이어를 추가할 수 없습니다.");
			this->layers.emplace_back(layer);
			return *this;
		}
		// model.add(Conv(3, 16)) 처럼 new 없이 넘겨도 된다.
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
		// 학습 중간에 장치를 바꿔도 된다. (가중치가 따라서 옮겨진다)
		void setDevice(Device device) {
			this->requested = device;
			Device actual = this->resolve(device);
			if (actual == this->ctx.device)
				return;
			this->ctx.device = actual;
			this->ctx.workspace.free();
			for (auto& layer : this->layers)
				layer->toDevice(actual);
		}
		Device device() const { return this->ctx.device; }
		void setLearningRate(float lr) {
			this->optimizer.learningRate = lr;
			this->learningRateSet = true;
		}
		void setConvAlgorithm(CONV_ALGO algo) { this->ctx.convAlgo = algo; }
		// im2col 임시 버퍼 최대 크기 (MB). GPU 메모리가 작으면 낮춘다. 기본 512MB
		void setWorkspaceLimit(size_t megabytes) { this->ctx.workspaceLimit = (std::max)((size_t)16, megabytes) << 20; }
		void setVerbose(bool verbose) { this->ctx.verbose = verbose; }
		void setSeed(unsigned seed) { this->rng.seed(seed); }
		size_t layerCount() const { return this->layers.size(); }
		Layer* layer(size_t i) { return this->layers.at(i).get(); }

		History fit(int epochs, int batchSize) { return this->fit(this->ownData, epochs, batchSize); }
		History fit(const Dataset& train, int epochs, int batchSize, const Dataset* validation = nullptr) {
			if (train.empty())
				throw std::runtime_error("학습 데이터가 비어 있습니다.");
			this->build(train.shape());
			this->checkLabels(train);
			batchSize = (std::max)(1, batchSize);
			std::vector<float> batch(train.shape().sample() * batchSize);
			std::vector<float> labels(batchSize);
			std::vector<int> order(train.size());
			for (size_t i = 0; i < order.size(); ++i)
				order[i] = (int)i;
			std::vector<Param*> params = this->allParams();
			History history;
			if (this->ctx.verbose)
				std::printf("[SimpleCNN] 학습 시작: %zu 장, 배치 %d, %s, 학습률 %g\n", train.size(), batchSize, toString(this->ctx.device), this->optimizer.learningRate);
			for (int epoch = 1; epoch <= epochs; ++epoch) {
				std::shuffle(order.begin(), order.end(), this->rng);
				Timer timer;
				double lossSum = 0.0;
				int correct = 0;
				size_t done = 0;
				for (size_t begin = 0; begin < train.size(); begin += batchSize) {
					int count = (int)(std::min)((size_t)batchSize, train.size() - begin);
					train.batch(order, begin, count, batch.data(), labels.data());
					this->forward(batch.data(), count);
					lossSum += (double)this->computeLoss(labels.data(), count, correct) * count;
					this->backward();
					this->optimizer.update(params, this->ctx.device);
					done += count;
					bool show = (begin / batchSize) % 20 == 0 || done == train.size();
					if (this->ctx.verbose && show)
						progress(epoch, epochs, done, train.size(), lossSum / done, (double)correct / done, timer.seconds());
				}
				history.loss.push_back((float)(lossSum / done));
				history.accuracy.push_back((float)correct / done);
				if (this->ctx.verbose)
					std::printf(" (%.0f 장/초)", done / (std::max)(timer.seconds(), 1e-9));
				if (validation != nullptr && !validation->empty()) {
					std::vector<int> predicted = this->predict(*validation, batchSize);
					int right = 0;
					for (size_t i = 0; i < predicted.size(); ++i)
						right += (int)(predicted[i] == (int)validation->label(i));
					history.validationAccuracy.push_back((float)right / validation->size());
					if (this->ctx.verbose)
						std::printf("  val acc %.2f%%", 100.0f * right / validation->size());
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
			std::vector<int> predicted = this->predict(data, batchSize);
			int correct = 0;
			for (size_t i = 0; i < predicted.size(); ++i)
				correct += (int)(predicted[i] == (int)data.label(i));
			float acc = (float)correct / (std::max)((size_t)1, data.size());
			if (this->ctx.verbose)
				std::printf("[SimpleCNN] 정확도: %.2f%% (%d / %zu)\n", acc * 100.0f, correct, data.size());
			return acc;
		}
		float accuracy() { return this->evaluate(this->ownData); }

		// 데이터셋 전체 예측 (클래스 번호)
		std::vector<int> predict(const Dataset& data, int batchSize = 256) {
			this->build(data.shape());
			std::vector<int> result;
			std::vector<int> order(data.size());
			for (size_t i = 0; i < order.size(); ++i)
				order[i] = (int)i;
			std::vector<float> batch(data.shape().sample() * batchSize);
			std::vector<float> labels(batchSize);
			for (size_t begin = 0; begin < data.size(); begin += batchSize) {
				int count = (int)(std::min)((size_t)batchSize, data.size() - begin);
				data.batch(order, begin, count, batch.data(), labels.data());
				this->forward(batch.data(), count);
				std::vector<float> out = this->outputs.back().toVector();
				int cols = this->outputUnits();
				for (int r = 0; r < count; ++r)
					result.push_back(argmax(out.data() + (size_t)r * cols, cols));
			}
			return result;
		}
		// 한 장의 클래스별 확률 (마지막 레이어 출력). chw 는 0~1 float, CHW 순서
		std::vector<float> predictProba(const float* chw) {
			if (!this->built)
				throw std::runtime_error("모델이 아직 없습니다. fit() 이나 load() 를 먼저 호출하세요.");
			this->forward(chw, 1);
			return this->outputs.back().toVector();
		}
		std::vector<float> predictProba(const Image& image) {
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
			this->build(one.shape());
			std::vector<float> chw(one.shape().sample());
			one.sample(0, chw.data());
			std::vector<float> p = this->predictProba(chw.data());
			return argmax(p.data(), (int)p.size());
		}

		// 저장: 텍스트 헤더 (입력 크기, 레이어 구성) + 바이너리 가중치
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
				std::vector<float> values = p->value.toVector();
				uint64_t count = values.size();
				file.write(reinterpret_cast<const char*>(&count), sizeof(count));
				file.write(reinterpret_cast<const char*>(values.data()), (std::streamsize)(count * sizeof(float)));
			}
			if (this->ctx.verbose)
				std::printf("[SimpleCNN] 저장: %s\n", path.c_str());
		}
		// 불러오기: 레이어 구성까지 파일에서 복원한다.
		void load(const std::string& path) {
			std::ifstream file(toPath(path), std::ios::binary);
			std::string line;
			std::getline(file, line);
			if (line != "SimpleCNN 2")
				throw std::runtime_error("SimpleCNN 모델 파일이 아니거나 열 수 없습니다: " + path);
			std::string word;
			Shape shape(1, 0, 0, 0);
			size_t count = 0;
			file >> word >> shape.c >> shape.h >> shape.w >> word >> count;
			std::getline(file, line);
			this->layers.clear();
			this->built = false;
			for (size_t i = 0; i < count; ++i) {
				std::getline(file, line);
				this->layers.emplace_back(createLayer(line));
			}
			// 모양을 정해 가중치 크기를 알아낸 뒤 파일의 값으로 채운다.
			Shape s = shape;
			for (auto& layer : this->layers)
				s = layer->build(s, this->rng);
			for (Param* p : this->allParams()) {
				uint64_t n = 0;
				file.read(reinterpret_cast<char*>(&n), sizeof(n));
				if (!file || n != p->size())
					throw std::runtime_error("모델 파일이 손상되었습니다: " + path);
				std::vector<float> values((size_t)n);
				file.read(reinterpret_cast<char*>(values.data()), (std::streamsize)(n * sizeof(float)));
				p->value.resize((size_t)n, Device::CPU);
				p->value.upload(values.data(), values.size());
			}
			this->build(shape);
			if (this->ctx.verbose)
				std::printf("[SimpleCNN] 불러옴: %s (%zu 레이어)\n", path.c_str(), this->layers.size());
		}

		// 레이어별 출력 모양, 파라미터 수, 예상 메모리 출력. 학습 전이면 입력 모양을 넘긴다.
		void summary(int batchSize = 32) {
			if (!this->built && this->ownData.empty())
				throw std::runtime_error("입력 크기를 알 수 없습니다. summary(Shape(1, 채널, 세로, 가로)) 를 사용하세요.");
			if (!this->built)
				this->build(this->ownData.shape());
			std::printf("\n %-16s %-18s %10s   %s\n", "Layer", "Output", "Params", "Detail");
			std::printf(" ------------------------------------------------------------------\n");
			std::printf(" %-16s %-18s\n", "Input", this->inputShape.str().c_str());
			size_t total = 0;
			size_t activations = this->inputShape.sample();
			for (auto& layer : this->layers) {
				total += layer->paramCount();
				activations += layer->outShape.sample();
				std::printf(" %-16s %-18s %10zu   %s\n", layer->name().c_str(), layer->outShape.str().c_str(), layer->paramCount(), layer->detail().c_str());
			}
			std::printf(" ------------------------------------------------------------------\n");
			std::printf(" 파라미터 %zu 개 (%s)\n", total, formatBytes(total * 4.0).c_str());
			std::printf(" 배치 %d 기준 예상 메모리: 레이어 출력 %s + 가중치/기울기/옵티마이저 %s + 작업 버퍼 최대 %s\n", batchSize,
				formatBytes(activations * batchSize * 4.0).c_str(), formatBytes(total * 16.0).c_str(), formatBytes((double)this->ctx.workspaceLimit).c_str());
			std::printf(" 장치: %s\n\n", toString(this->ctx.device));
		}
		void summary(const Shape& sample, int batchSize = 32) {
			this->build(sample);
			this->summary(batchSize);
		}
		void printMemory() const {
			std::printf("[SimpleCNN] 최대 메모리: CPU %s", formatBytes((double)Tensor::hostPeak()).c_str());
			if (this->ctx.device == Device::CUDA)
				std::printf(", GPU %s", formatBytes((double)cuda::Runtime::instance().memoryPeak()).c_str());
			std::printf("\n");
		}
	};
}

#endif
