/*


0829 할일:
	1. CNN 필터 업데이트를 구현해야함.
		// 목표
			// delta : 2 27 368 280
			// -> 2 9 370 282 로 변경해야함.
			// 근데 27에서 9로 그냥 바꿀수있는건 아님
			// x y 값이 바뀌기 때문에
			// 전체 크기를 알고있어야함
	1. pooling backward, padding backward 제대로 잘 안넘어감.
	2. deep copy setDelta, getDelta 벡터 일일이 확인해서 뭐가 문제인지 확인
	3. pooling에서 setDelta하고나서 벡터 소멸됨
	
	1. gradient = delta니까 개수 잘 확인하고 변경하기
	
	

	  Last FC에서 deltaXw 랑 derivativeAct(dLossOutput) 부분이 이상함
	  애초에 deltaFC가 안구해져서 그런듯.
	

	
	
	1. delta에서 가져온 값을 4D벡터에 풀링만큼 평균적으로 나눠 대입해야함.
	위의 주석 코드는 틀린것임 수정해야함.
	모든 범위에 대해서 2x2풀링만큼 평균내서 대입해야함 (근데 나중에 할 것)  Conv역전파 끝낸후에.
    
	Vector 할당 해제는 맨 마지막에
	
	
	2. Conv역전파 구현


	3. CONV PADD POOL 섞었을때 또는 CONV 반복할때 실행안됨. 예상으로는 CONV가 여러개면 Weight도 여러번 공유되서인듯. 
*/
