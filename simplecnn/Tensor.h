// SimpleCNN - Tensor
//
// 예전 V4D(vector<vector<vector<vector<float>>>>)는 행마다 따로 할당돼서
// 포인터/헤더 오버헤드가 크고 캐시도 깨진다. Tensor 는 NCHW 순서의 연속 메모리 한 덩어리다.
//  - resize 는 용량이 부족할 때만 새로 잡는다 → 배치마다 할당이 일어나지 않는다.
//  - CPU 는 64바이트 정렬(SIMD), CUDA 는 Runtime 메모리 풀에서 받아온다.
#ifndef SIMPLE_CNN_TENSOR_H
#define SIMPLE_CNN_TENSOR_H

#include "Core.h"
#include "Cuda.h"

#include <new>

namespace cnn {

	// 호스트 메모리 사용량 (summary 출력용)
	struct MemoryStats {
		size_t hostInUse = 0;
		size_t hostPeak = 0;

		static MemoryStats& get() {
			static MemoryStats stats;
			return stats;
		}
	};

	class Tensor {
	private:
		float* ptr = nullptr;
		size_t capacity = 0;
		Device dev = Device::CPU;

		void allocate(size_t count, Device device) {
			this->dev = device;
			this->capacity = count;
			if (count == 0)
				return;
			size_t bytes = count * sizeof(float);
			if (device == Device::CUDA) {
				this->ptr = static_cast<float*>(cuda::Runtime::instance().alloc(bytes));
				return;
			}
			this->ptr = static_cast<float*>(::operator new[](bytes, std::align_val_t(64)));
			MemoryStats& stats = MemoryStats::get();
			stats.hostInUse += bytes;
			stats.hostPeak = (std::max)(stats.hostPeak, stats.hostInUse);
		}
	public:
		Shape shape;

		Tensor() {}
		Tensor(const Shape& shape, Device device) { this->resize(shape, device); }
		Tensor(const Tensor&) = delete;
		Tensor& operator=(const Tensor&) = delete;
		Tensor(Tensor&& other) noexcept { *this = std::move(other); }
		Tensor& operator=(Tensor&& other) noexcept {
			if (this != &other) {
				this->free();
				this->ptr = other.ptr;
				this->capacity = other.capacity;
				this->dev = other.dev;
				this->shape = other.shape;
				other.ptr = nullptr;
				other.capacity = 0;
				other.shape = Shape();
			}
			return *this;
		}
		~Tensor() { this->free(); }

		void free() {
			if (this->ptr != nullptr) {
				size_t bytes = this->capacity * sizeof(float);
				if (this->dev == Device::CUDA) {
					cuda::Runtime::instance().release(this->ptr, bytes);
				}
				else {
					::operator delete[](this->ptr, std::align_val_t(64));
					MemoryStats::get().hostInUse -= bytes;
				}
			}
			this->ptr = nullptr;
			this->capacity = 0;
			this->shape = Shape();
		}
		// 모양만 바꾸고, 용량이 모자랄 때만 다시 할당한다. (기존 값은 보존하지 않음)
		void resize(const Shape& newShape, Device device) {
			size_t need = newShape.size();
			if (device != this->dev || need > this->capacity) {
				this->free();
				this->allocate(need, device);
			}
			this->shape = newShape;
		}
		void resize(size_t count, Device device) {
			this->resize(Shape(1, 1, 1, (int)count), device);
		}

		float* data() { return this->ptr; }
		const float* data() const { return this->ptr; }
		size_t size() const { return this->shape.size(); }
		size_t bytes() const { return this->capacity * sizeof(float); }
		Device device() const { return this->dev; }
		bool empty() const { return this->size() == 0; }

		void fill(float value) {
			if (this->size() == 0)
				return;
			if (this->dev == Device::CUDA) {
				cuda::Runtime::instance().launch("fill_kernel", this->size(), this->ptr, value, (int)this->size());
				return;
			}
			std::fill(this->ptr, this->ptr + this->size(), value);
		}
		void zero() { this->fill(0.0f); }

		// 호스트 배열 → 이 텐서 (CPU 면 복사, CUDA 면 업로드)
		void upload(const float* host, size_t count) {
			if (count == 0)
				return;
			if (this->dev == Device::CUDA) {
				cuda::Runtime::instance().upload(this->ptr, host, count * sizeof(float));
				return;
			}
			std::memcpy(this->ptr, host, count * sizeof(float));
		}
		void download(float* host, size_t count) const {
			if (count == 0)
				return;
			if (this->dev == Device::CUDA) {
				cuda::Runtime::instance().download(host, this->ptr, count * sizeof(float));
				return;
			}
			std::memcpy(host, this->ptr, count * sizeof(float));
		}
		std::vector<float> toVector() const {
			std::vector<float> out(this->size());
			this->download(out.data(), out.size());
			return out;
		}
		// 같은 장치의 다른 텐서 내용을 그대로 복사 (모양 포함)
		void copyFrom(const Tensor& other) {
			this->resize(other.shape, other.device());
			if (this->dev == Device::CUDA) {
				cuda::Runtime::instance().copy(this->ptr, other.data(), other.size() * sizeof(float));
				return;
			}
			std::memcpy(this->ptr, other.data(), other.size() * sizeof(float));
		}
		// 장치를 옮긴다. (CPU ↔ CUDA, 값 유지)
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
}

#endif
