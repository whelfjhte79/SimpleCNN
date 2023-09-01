#include<opencv2/opencv.hpp>
#include<opencv2/core.hpp>
#include<iostream>
#include"SimpleCNN.h"


using namespace cv;
using namespace std;
using namespace img_read;
using namespace cnn;

//#include"test.h"

// 메인부분
#if !defined(TEST) && !defined(TEST2)

int main(void) {

	Directory* dirent = new Directory("C:\\Users\\이상민\\source\\repos\\SimpleCNN_image\\SimpleCNN_image",img_read::FILENAME_EXTENSION::JPG);
	FileRead* label = new FileRead("C:\\Users\\이상민\\source\\repos\\SimpleCNN_image\\SimpleCNN_image\\test.csv");
	CNN* cnn = new CNN(dirent->getImageSet(), label->getLabel());
	//cnn->splitTrainTest(0.3);
	cnn->add(new Conv(3, 3, 1, 1, ACTIVATION::ReLU)); 
	cnn->add(new Pooling(POOLING::Max));
	cnn->add(new Padding(1));
	cnn->add(new Conv(3, 3, 1, 1, ACTIVATION::ReLU));
	cnn->add(new Pooling(POOLING::Max));
	cnn->add(new Padding(1));
	cnn->add(new Pooling(POOLING::Max));
	cnn->add(new Pooling(POOLING::Max));
	cnn->add(new Pooling(POOLING::Max));
	cnn->add(new Pooling(POOLING::Max));
	
	cnn->add(new Flatten());
	cnn->add(new FullyConnected(256, ACTIVATION::ReLU));
	cnn->add(new FullyConnected(128, ACTIVATION::ReLU));
	cnn->add(new FullyConnected(64, ACTIVATION::ReLU));

	cnn->add(new FullyConnected(10, ACTIVATION::Softmax));

	cnn->compile(OPTIMIZER::Mini_SGD, LOSS::CategoricalCrossentropy);
	cnn->fit(3,2);



	cnn->predict(dirent->getImageSet().at(2),label->getLabel().at(2));

	cnn->accuracy();
	
	

	return 0;
}
#endif
#ifdef TEST
#include"test.h"
/*
* 할일: 각 레이어별 calculate 함수 검사
* 검사되면 합치기
* 벡터 반환 왜 안돼는지 확인

*/

#include <random>
#include <vector>
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_int_distribution<int> dis(1, 10000);

int main(void) {
	V2D testVec2D = { {0.3f,-1.4f,2.0f,-1.7f}, 
					  {-2.3f,1.1f,1.9f,0.5f} 
					};
	V2D label2D = { {0.0f,0.0f,1.0f,0.0f},
					{1.0f,0.0f,2.0f,4.0f}
	};
	
	CNN* cnn = new CNN();

	cout<< cnn->derivativeLossFunc(&testVec2D, &label2D);



	return 0;
}

#endif
#ifdef TEST2
#include"test.h"


int main(void) {
	test();
	return 0;
}


#endif