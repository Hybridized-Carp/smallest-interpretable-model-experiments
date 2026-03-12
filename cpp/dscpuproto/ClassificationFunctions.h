#pragma once

#include<array>
using namespace std;

namespace classificationFunctions {

	template <class T>
	constexpr T pwrtwo(T exponent) {
		return (T(1) << exponent-1);
	}

	bool clsf1(unsigned char i) {
		if (!(i & 128))
			return true;
		else if (i & 64)
			return false;
		else if (i & 32)
			return true;
		else
			return false;
			
	}

	//spread features across uint
	char mapuinttochar(unsigned int x, std::array<char,8> map) {
		char rt = 0;
		for (char pos : map) {
			char temp = (x & pwrtwo(pos)) ? 1 : 0;
			rt |= temp;
			rt <<= 1;
		}
		return rt;
	}

	bool clsf2(unsigned int i) {
		if ((i & 8)==0)
			return true;
		else if (i& 4)
			return false;
		else if (i & 2)
			return true;
		else
			return false;

	}
}