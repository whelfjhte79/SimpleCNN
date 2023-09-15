#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include<iostream>
#include"SimpleCNN.h"

#ifdef __cplusplus 

extern "C" {//<-- extern ½ÃÀÛ

#endif



	class GPU_TEST

	{

	public:

		GPU_TEST(void);

		virtual ~GPU_TEST(void);
		int sum_cuda(int a, int b, int* c);
	};



#ifdef __cplusplus 

}

#endif


