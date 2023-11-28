#ifndef FASTER_RCNN_H
#define FASTER_RCNN_H


namespace faster_rcnn {
	// https://welcome-to-dewy-world.tistory.com/110
	// 1�� �̹������� Conv, pooling ���θ� ������ ������Ѽ� ���� Ư¡���� �Է������� �����.
	// ���� Ư¡�ʿ� 3x3 conv�� 256(or 512) channel ��ŭ ������. //�̶� padding�� 1�� �ؼ� H x W�� ������.
    // ���� �ι�° Ư¡�ʿ� (1) classification layer, (2) bbox regression �������� �����.
	// (1)�� ������  (2(Object���� �ƴ����� ��Ÿ��) x 9(��Ŀ(Anchor) ����)) ä�� �� ��ŭ �����ؼ� H x W * 18 ũ��� Ư¡���� ����
	//       H*W ���� �ϳ��� �ε����� ���ĸ� ���� ��ǥ�� �ǹ��ϰ�, 
	//       �� �Ʒ� 18���� ä���� ���� �ش� ��ǥ�� Anchor�� ��� k���� ��Ŀ �ڽ����� 
	//       object���� �ƴ����� ���� ���� ���� ��´�. 
	//       ��, �ѹ��� 1x1 ����������� H x W ���� Anchor ��ǥ�鿡 ���� ������ ��� ������ �� �ִ�. 
	//       ���� �� ������ ������ reshape ���� ���� Softmax�� �����Ͽ� �ش� Anchor�� Object�� Ȯ�� ���� ��´�.
	//       +) ���⼭ ��Ŀ(Anchor)��, �� �����̵� �����쿡�� bounding box�� �ĺ��� ���Ǵ� ���ڸ� �ǹ��ϸ� �̵��Һ����� Ư¡�� ���´�.
	// 
	// 
	// (2)�� ������ Bounding Box Regression ���� ���� ��� ���� 1 x 1 ��������� (4 x 9) ä�� �� ��ŭ �����ϸ� 
	//             regression�̱� ������ ����� ���� ���� �״�� ����Ѵ�.
	//
	// ���� Classification�� ���� Bounding Box Regression�� ������ ���� RoI�� ����Ѵ�.
	//       +) ���� Classification�� ���ؼ� ���� ��ü�� Ȯ�� ������ ������ ����, ���� ������ K���� ��Ŀ�� �߷�����. 
	//          �� ��, K���� ��Ŀ�鿡 ���� Bounding box regression�� �����ϰ� Non-Maximum-Suppression�� �����Ͽ� RoI�� ���Ѵ�.
	//
	// 
	// 
	//    �н�:
	//    Faster R-CNN�� ũ�� RPN�� Fast R-CNN���� �����Ǿ��ִ� ������ ���� �� �ְ�, ��������� RPN�� RoI�� ����� ������ �н��� �� �Ǳ� ������ 4�ܰ迡 ���� ���� ������ �н���Ų��. 4�ܰ�� ������ ����.
	//
	//        1. ImageNet pretrained ���� �ҷ��� ��, Region Proposal�� ���� end - to - end�� RPN�� �н���Ų��.
	//	      2. 1�ܰ迡�� �н���Ų RPN���� �⺻ CNN�� ������ Region Proposal layer���� ������ Fast R - CNN�� �н���Ų��.
	//	          + ) ù feature map�� �����ϴ� CNN���� fine tune�� ��Ų��.
	//	      3. �н���Ų Fast R - CNN�� RPN�� �ҷ��� �ٸ� ����ġ��(����� conv layer ����)�� �����ϰ� RPN�� �ش��ϴ� layer�鸸 fine tune ��Ų��.
	//	          + ) �� ������ RPN�� Fast R - CNN�� conv weight�� �����ϰ� �ȴ�.
	//	      4. ������ conv layer�� ������Ű�� Fast R - CNN�� �ش��ϴ� ���̾ fine tune ��Ų��.
	// 
	// 
	// 
	//




}






#endif