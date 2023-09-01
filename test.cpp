#include"test.h"

#ifdef TEST

int test() {

	return 0;
}

#endif

#ifdef TEST2

int test() {
	V1D* a = new V1D();
	a->push_back(5.0f);
	a->push_back(3.0f);
	a->push_back(7.0f);
	a->push_back(8.0f);
	for (int i = 0; i < a->size(); i++) {
		cout<< (*a)[i] << " ";
	}
	return 0;
}

#endif


