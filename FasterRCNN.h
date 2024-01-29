#ifndef FASTER_RCNN_H
#define FASTER_RCNN_H

#include"SimpleCNN.h"

namespace faster_rcnn {
	//https://www.youtube.com/watch?v=RjfgHOiBWaE
	// https://welcome-to-dewy-world.tistory.com/110
	// 1번 이미지에서 Conv, pooling 층로만 구성해 통과시켜서 나온 특징맵을 입력층으로 사용함.
	// 나온 특징맵에 3x3 conv를 256(or 512) channel 만큼 수행함. //이때 padding을 1로 해서 H x W를 보존함.
    // 나온 두번째 특징맵에 (1) classification layer, (2) bbox regression 예측값을 계산함.
	// (1)번 에서는  (2(Object인지 아닌지를 나타냄) x 9(앵커(Anchor) 개수)) 채널 수 만큼 수행해서 H x W * 18 크기로 특징맵을 얻음
	//       H*W 상의 하나의 인덱스는 피쳐맵 상의 좌표를 의미하고, 
	//       그 아래 18개의 채널은 각각 해당 좌표를 Anchor로 삼아 k개의 앵커 박스들이 
	//       object인지 아닌지에 대한 예측 값을 담는다. 
	//       즉, 한번의 1x1 컨볼루션으로 H x W 개의 Anchor 좌표들에 대한 예측을 모두 수행할 수 있다. 
	//       이제 이 값들을 적절히 reshape 해준 다음 Softmax를 적용하여 해당 Anchor가 Object일 확률 값을 얻는다.
	//       +) 여기서 앵커(Anchor)란, 각 슬라이딩 윈도우에서 bounding box의 후보로 사용되는 상자를 의미하며 이동불변성의 특징을 갖는다.
	// 
	// 
	// (2)번 에서는 Bounding Box Regression 예측 값을 얻기 위한 1 x 1 컨볼루션을 (4 x 9) 채널 수 만큼 수행하며 
	//             regression이기 때문에 결과로 얻은 값을 그대로 사용한다.
	//
	// 이후 Classification의 값과 Bounding Box Regression의 값들을 통해 RoI를 계산한다.
	//       +) 먼저 Classification을 통해서 얻은 물체일 확률 값들을 정렬한 다음, 높은 순으로 K개의 앵커를 추려낸다. 
	//          그 후, K개의 앵커들에 각각 Bounding box regression을 적용하고 Non-Maximum-Suppression을 적용하여 RoI을 구한다.
	//
	// 
	// 
	//    학습:
	//    Faster R-CNN은 크게 RPN과 Fast R-CNN으로 구성되어있는 것으로 나눌 수 있고, 결론적으로 RPN이 RoI를 제대로 만들어야 학습이 잘 되기 때문에 4단계에 걸쳐 모델을 번갈아 학습시킨다. 4단계는 다음과 같다.
	//
	//        1. ImageNet pretrained 모델을 불러온 후, Region Proposal을 위해 end - to - end로 RPN을 학습시킨다.
	//	      2. 1단계에서 학습시킨 RPN에서 기본 CNN을 제외한 Region Proposal layer만을 가져와 Fast R - CNN을 학습시킨다.
	//	          + ) 첫 feature map을 추출하는 CNN까지 fine tune을 시킨다.
	//	      3. 학습시킨 Fast R - CNN과 RPN을 불러와 다른 가중치들(공통된 conv layer 고정)을 고정하고 RPN에 해당하는 layer들만 fine tune 시킨다.
	//	          + ) 이 때부터 RPN과 Fast R - CNN이 conv weight를 공유하게 된다.
	//	      4. 공유된 conv layer를 고정시키고 Fast R - CNN에 해당하는 레이어만 fine tune 시킨다.
	// 
	// 
	// 
	//
	class RPN {
		V4D featureMap;
		cnn::Conv* conv = new cnn::Conv();
		V3D* intermediateFilter3D = nullptr;
		V3D* classificationFilter3D = nullptr;
		V3D* bboxRegressionFilter3D = nullptr;
	public:
		RPN() {

		}
		RPN(V4D& input) {
			this->featureMap = input;
		}
		virtual ~RPN() {

		}
		V4D intermediate(V4D& featureMap, V3D* filter3D) {
			
			conv->setFilter3D(filter3D);
			
			cnn::Padding pad = cnn::Padding(1);
			V4D resultFeatureMap = pad.calculate(conv->calculate(featureMap));
			return resultFeatureMap;
		}
		V2D classificationForward(V4D& intermediateFeatureMap,V3D* filter3D) {
			conv->setFilter3D(filter3D);
			// 1x1 conv  [2(isObject) x 9(Anchor)] X channel
			V4D featureMap = conv->calculate(intermediateFeatureMap);

			cnn::Flatten flat = cnn::Flatten();
			V2D flatFeatureMap = flat.calculate(featureMap);
			cnn::FullyConnected fc = cnn::FullyConnected(10, cnn::ACTIVATION::Softmax);


			return fc.calculate(flatFeatureMap);
		}
		V4D bboxRegressionForward(V4D& intermediateFeatureMap,V3D* filter3D) {
			conv->setFilter3D(filter3D);
			// 1x1 conv  [4(bbox) x 9(Anchor)] x channel

			cnn::Padding pad = cnn::Padding(1);
			
			V4D resultFeatureMap = conv->calculate(pad.calculate(intermediateFeatureMap));
			return resultFeatureMap;
		}
		void classificationBackward() {

		}
		void bboxRegressionBackward() {

		}
		void forward() {

		}
		void backward() {

		}
	};
	class FasterRCNN {
		float IoU;

	};


}






#endif