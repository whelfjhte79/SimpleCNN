// SimpleCNN - 공통 정의 (enum, Shape, 병렬 반복, 유틸)
#ifndef SIMPLE_CNN_CORE_H
#define SIMPLE_CNN_CORE_H

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace cnn {
	// 예전 코드와의 호환용 (중첩 vector 입력을 그대로 받을 수 있게)
	using V1D = std::vector<float>;
	using V2D = std::vector<V1D>;
	using V3D = std::vector<V2D>;
	using V4D = std::vector<V3D>;

	enum class Device {
		CPU, CUDA
	};
	enum class LAYER {
		Conv, Pool, Act, Padding, FC, Flatten, None
	};
	enum class POOLING {
		Max, Average, Min
	};
	enum class ACTIVATION {
		None, Error, Sigmoid, ReLU, Leaky_ReLU, TanH, Maxout, ELU, Softmax
	};
	enum class OPTIMIZER {
		GD, Mini_SGD, Momentum, Adagrad, RMSProp, Adam, Nadam
	};
	enum class LOSS {
		MSE, RMSE, BinaryCrossentropy, CategoricalCrossentropy, SparseCategoricalCrossentropy
	};
	// 합성곱 계산 방식. Auto는 첫 배치에서 직접 재보고 가장 빠른 방식을 고른다.
	enum class CONV_ALGO {
		Auto, Im2col, FFT, Direct
	};

	// 텐서 모양 (NCHW). FC 출력은 c=유닛 수, h=w=1 로 표현한다.
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
			std::ostringstream os;
			os << "(" << c << ", " << h << ", " << w << ")";
			return os.str();
		}
	};

	inline const char* toString(Device device) {
		if (device == Device::CUDA)
			return "CUDA";
		return "CPU";
	}
	inline const char* toString(ACTIVATION act) {
		switch (act) {
		case ACTIVATION::None: return "None";
		case ACTIVATION::Sigmoid: return "Sigmoid";
		case ACTIVATION::ReLU: return "ReLU";
		case ACTIVATION::Leaky_ReLU: return "Leaky_ReLU";
		case ACTIVATION::TanH: return "TanH";
		case ACTIVATION::Maxout: return "Maxout";
		case ACTIVATION::ELU: return "ELU";
		case ACTIVATION::Softmax: return "Softmax";
		default: return "Error";
		}
	}
	inline const char* toString(POOLING pool) {
		if (pool == POOLING::Average)
			return "Average";
		if (pool == POOLING::Min)
			return "Min";
		return "Max";
	}
	inline const char* toString(CONV_ALGO algo) {
		switch (algo) {
		case CONV_ALGO::Im2col: return "im2col+GEMM";
		case CONV_ALGO::FFT: return "FFT";
		case CONV_ALGO::Direct: return "Direct";
		default: return "Auto";
		}
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

	// [0, count) 를 스레드에 나눠 실행한다. OpenMP가 없으면 그냥 순차 실행.
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

	// 환경 변수 읽기. (MSVC 의 SDL 검사에서 getenv 가 오류로 처리되지 않도록 _dupenv_s 사용)
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

	class Timer {
	private:
		std::chrono::steady_clock::time_point start;
	public:
		Timer() { this->reset(); }
		void reset() { this->start = std::chrono::steady_clock::now(); }
		double seconds() const {
			std::chrono::duration<double> d = std::chrono::steady_clock::now() - this->start;
			return d.count();
		}
	};

	inline std::string formatBytes(double bytes) {
		const char* units[] = { "B", "KB", "MB", "GB", "TB" };
		int u = 0;
		while (bytes >= 1024.0 && u < 4) {
			bytes /= 1024.0;
			++u;
		}
		std::ostringstream os;
		os.setf(std::ios::fixed);
		os.precision(1);
		os << bytes << " " << units[u];
		return os.str();
	}

	// 활성화 함수 인자 (커널과 CPU 코드가 같은 값을 쓰도록 한 곳에서 정의)
	const float kLeakySlope = 0.01f;
	const float kEluAlpha = 1.0f;
}

#endif
