#pragma once
#include "cuda_runtime.h"
#include "dataset.h"
using namespace std;
namespace model {

	struct term {
		vector<uint32_t> zliterals;
		vector<uint32_t> oliterals;
		vector<dataset::datapoint> annotation;
	};

	struct decisionSetModel {
		bool defaultRule;
		vector<term>terms;
		dataset::datapoint annotation;
	};

	

}