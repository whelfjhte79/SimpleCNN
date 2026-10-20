// SimpleCNN - 옵티마이저
// 상태 버퍼(m, v)는 그 옵티마이저가 실제로 필요할 때만 잡는다.
// (SGD 는 0, Momentum 은 1벌, Adam 은 2벌 → 파라미터 메모리 1~3배)
#ifndef SIMPLE_CNN_OPTIMIZER_H
#define SIMPLE_CNN_OPTIMIZER_H

#include "Layers.h"

namespace cnn {

	class Optimizer {
	private:
		OPTIMIZER type = OPTIMIZER::Adam;
		long long step = 0;

		int kernelType() const {
			switch (this->type) {
			case OPTIMIZER::Momentum: return 1;
			case OPTIMIZER::Adagrad: return 2;
			case OPTIMIZER::RMSProp: return 3;
			case OPTIMIZER::Adam: return 4;
			case OPTIMIZER::Nadam: return 5;
			default: return 0;
			}
		}
		bool needsM() const {
			int t = this->kernelType();
			return t == 1 || t == 4 || t == 5;
		}
		bool needsV() const {
			return this->kernelType() >= 2;
		}
		void prepare(Param& p, Device device) {
			if (this->needsM() && p.m.size() != p.size()) {
				p.m.resize(p.value.shape, device);
				p.m.zero();
			}
			if (this->needsV() && p.v.size() != p.size()) {
				p.v.resize(p.value.shape, device);
				p.v.zero();
			}
		}
	public:
		float learningRate = 0.001f;
		float beta1 = 0.9f;
		float beta2 = 0.999f;
		float epsilon = 1e-7f;

		Optimizer() {}
		Optimizer(OPTIMIZER type) { this->setType(type); }

		static float defaultLearningRate(OPTIMIZER type) {
			switch (type) {
			case OPTIMIZER::GD:
			case OPTIMIZER::Mini_SGD:
			case OPTIMIZER::Momentum:
				return 0.01f;
			case OPTIMIZER::Adagrad:
				return 0.01f;
			default:
				return 0.001f;
			}
		}
		void setType(OPTIMIZER newType) {
			this->type = newType;
			this->learningRate = defaultLearningRate(newType);
			this->step = 0;
		}
		OPTIMIZER getType() const { return this->type; }

		void update(const std::vector<Param*>& params, Device device) {
			++this->step;
			int t = this->kernelType();
			float b2 = this->beta2;
			if (t == 3)
				b2 = 0.9f;
			float bc1 = 1.0f - (float)std::pow((double)this->beta1, (double)this->step);
			float bc2 = 1.0f - (float)std::pow((double)b2, (double)this->step);
			float bc1Next = 1.0f - (float)std::pow((double)this->beta1, (double)(this->step + 1));
			for (Param* p : params) {
				this->prepare(*p, device);
				int n = (int)p->size();
				float* w = p->value.data();
				const float* g = p->grad.data();
				float* m = w;
				float* v = w;
				if (this->needsM())
					m = p->m.data();
				if (this->needsV())
					v = p->v.data();
				if (device == Device::CUDA) {
					cuda::Runtime::instance().launch("optimizer_kernel", (size_t)n, w, g, m, v, n, t, this->learningRate, this->beta1, b2, this->epsilon, bc1, bc2, bc1Next);
					continue;
				}
				float lr = this->learningRate;
				float eps = this->epsilon;
				float b1 = this->beta1;
				for (int i = 0; i < n; ++i) {
					float grad = g[i];
					if (t == 0) {
						w[i] -= lr * grad;
					}
					else if (t == 1) {
						m[i] = b1 * m[i] - lr * grad;
						w[i] += m[i];
					}
					else if (t == 2) {
						v[i] += grad * grad;
						w[i] -= lr * grad / (std::sqrt(v[i]) + eps);
					}
					else if (t == 3) {
						v[i] = b2 * v[i] + (1.0f - b2) * grad * grad;
						w[i] -= lr * grad / (std::sqrt(v[i]) + eps);
					}
					else {
						m[i] = b1 * m[i] + (1.0f - b1) * grad;
						v[i] = b2 * v[i] + (1.0f - b2) * grad * grad;
						float mHat = m[i] / bc1;
						if (t == 5)
							mHat = b1 * m[i] / bc1Next + (1.0f - b1) * grad / bc1;
						float vHat = v[i] / bc2;
						w[i] -= lr * mHat / (std::sqrt(vHat) + eps);
					}
				}
			}
		}
	};
}

#endif
