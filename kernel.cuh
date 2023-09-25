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
	class LayerGPU {
	private:
		V1D dummy1D;
		V2D dummy2D; // 가능하면 2차원 선에서 다 구현해야함.
		V3D dummy3D; // 최대 3차원
	public:
		virtual V2D calculateGPU(V2D data) { return dummy2D; }
		virtual V1D calculate(V1D data) { return dummy1D; }
		virtual V2D* backwardGPU(V2D* delta) { return &dummy2D; }
	};
	class ActivationGPU : public LayerGPU {
	public:
		int test() { return 0; }
		V2D calculateGPU(V2D data) override {
			V2D dummy;
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
	public:
		V2D calculateGPU(V2D data) override {
			V2D dummy;
			return dummy;
		}
		V2D* backwardGPU(V2D* delta) override {
			return delta;
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