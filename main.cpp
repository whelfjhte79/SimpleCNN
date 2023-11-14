#include<opencv2/opencv.hpp>
#include<opencv2/core.hpp>
#include<iostream>
#include"WinVisualization.h" // 시각화 차트
//#define _CUDA_GPU_ // GPU 사용
#include"SimpleCNN.h"

#ifdef _CUDA_GPU_
#include"kernel.cuh"
#endif //__CUDACC__

using namespace cv;
using namespace std;
using namespace img_read;
using namespace cnn;





//#include"test.h"
//#define __CUDACC__
// 메인부분
/*
#if !defined(TEST) && !defined(TEST2)

int sum_int(int a, int b);
int sum_int(int a, int b) {
	int c;
	c = a + b;
	return c;
}



#ifdef __CUDACC__
int main() {
	cout << "테스트";
	return 0;
}

#else
int main(void) {
	int a = 7, b = 5, c, d;
	
	c = sum_int(7, 5);
	GPU_TEST gpu_test;
	gpu_test.sum_cuda(a, b, &d);
	
	printf_s("CPU : %d + %d = %d  \n",a,b,c);
	printf_s("GPU : %d + %d = %d  \n",a,b,d);


	return 0;
}
#endif
*/

#ifdef _CUDA_GPU_
int main(void) {
	std::cout << "테스트";
	Directory* dirent = new Directory("C:\\Users\\이상민\\source\\repos\\SimpleCNN_image\\SimpleCNN_image", img_read::FILENAME_EXTENSION::JPG);
	FileRead* label = new FileRead("C:\\Users\\이상민\\source\\repos\\SimpleCNN_image\\SimpleCNN_image\\test.csv");

	//ActivationGPU a;
	//std::cout<< a.test() <<"\n";
	
	cnn::CNN cnn;
	V4D test = cnn.zeroFillToFit(dirent->getImageSet());
	
	Flatten flat;
	V2D t = flat.calculate(test, cnn::LAYER::Flatten);
	
	std::cout << t.size() << " " << t[0].size();
	
	WinApp* app = new WinApp({ 50 , 100 , 150 , 200 , 250, 300 }, { 100 , 50 , 200 , 150 , 250, 400 }, "X Axis", "Y Axis");
	app->Run();

	return 0;
}


#else

int main(void) {

	Directory* dirent = new Directory("C:\\Users\\이상민\\source\\repos\\SimpleCNN_image\\SimpleCNN_image",img_read::FILENAME_EXTENSION::JPG);
	FileRead* label = new FileRead("C:\\Users\\이상민\\source\\repos\\SimpleCNN_image\\SimpleCNN_image\\test.csv");
	CNN* cnn = new CNN(dirent->getImageSet(), label->getLabel());
	//cnn->splitTrainTest(0.3);
	cnn->add(new Conv(3, 3, 1, 1, ACTIVATION::TanH)); 
	cnn->add(new Pooling(POOLING::Max));
	cnn->add(new Padding(1));
	cnn->add(new Conv(3, 3, 1, 1, ACTIVATION::ReLU));
	cnn->add(new Pooling(POOLING::Max));
	cnn->add(new Padding(1));
	cnn->add(new Pooling(POOLING::Min));
	cnn->add(new Pooling(POOLING::Average));
	cnn->add(new Pooling(POOLING::Max));
	cnn->add(new Pooling(POOLING::Min));
	cnn->add(new Pooling(POOLING::Min));
	cnn->add(new Pooling(POOLING::Min));
	cnn->add(new Flatten());
	cnn->add(new FullyConnected(256, ACTIVATION::Maxout));
	cnn->add(new FullyConnected(128, ACTIVATION::Sigmoid));
	cnn->add(new FullyConnected(64, ACTIVATION::ReLU));

	cnn->add(new FullyConnected(10, ACTIVATION::Softmax));

	cnn->compile(OPTIMIZER::Momentum, LOSS::CategoricalCrossentropy);
	cnn->fit(1,5);

	cnn->save("cnn_test.txt");
	cnn->load("cnn_test.txt");
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