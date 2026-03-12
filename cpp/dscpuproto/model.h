#pragma once

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

	struct nterm {
		vector<uint32_t> fm;
		vector<uint32_t> vm;
		dataset::datapoint annotation;
	};

	struct ndsm {
		bool valid;
		bool defaultRule;
		vector<nterm> terms;
		dataset::datapoint annotation;
		int size;
	};

	bool applyNDSM(dataset::datapoint example, const ndsm& model) {
		for (nterm t : model.terms) {
			uint32_t rt = 0;
			for (int i = 0; i < example.featuregroups.size(); i++) {
				rt |= (t.fm[i]&example.featuregroups[i])^t.vm[i];
			}
			if (!rt)//mb rt==0
				return !model.defaultRule;
		}
		return model.defaultRule;
	}

	void evaluateNDSM(vector<dataset::datapoint> Ci, ndsm& model, dataset::datapoint &miscex) {
		for (dataset::datapoint example : Ci) {
			if (example.result != applyNDSM(example, model)) {
				miscex = example;
				model.valid = false;
				return;
			}
		}
		model.valid=true;
	}

	int FindNDSMSize(const ndsm& model) {
		int t = 0;
		for (nterm term : model.terms) {
			for (uint32_t fm : term.fm) {
				t += std::_Popcount(fm);
			}
		}
		return t;
	}

	vector<ndsm> FindStrictExtSet(const vector<dataset::datapoint>& Ci, const int size, const ndsm& model, const dataset::datapoint miscex) {
		if (miscex.result != model.defaultRule) {
			//can turn to array by using popc to count number of bits in xor mask
			//can prob then spin off creation of each individual extension set to a unique thread
			vector<ndsm> extensions;
			for (int i = 0; i < miscex.featuregroups.size(); i++) {
				uint32_t hmd = miscex.featuregroups[i] ^ model.annotation.featuregroups[i];
				uint32_t curmask = 1;
				for (int j = 0; j < 32; j++) {
					if (hmd & curmask){
						//check if we have copy bugs
						ndsm newmodel = model;
						vector<uint32_t> fm = vector<uint32_t>(miscex.featuregroups.size(), 0);
						vector<uint32_t> vm = vector<uint32_t>(miscex.featuregroups.size(), 0);
						fm[i] |= curmask;
						vm[i] |= curmask & miscex.featuregroups[i];
						nterm newterm({fm,vm, miscex});
						newmodel.terms.push_back(newterm);
						newmodel.size++;
						extensions.push_back(newmodel);
					}
					curmask <<= 1;
				}
			}
			return extensions;
		} else {
			//investigate possibly of modifying all terms that apply to our misclassified example at once
			for (int j = 0; j < model.terms.size(); j++){
				vector<ndsm> extensions;
				for (int i = 0; i < miscex.featuregroups.size(); i++) {
					uint32_t rt = 0;
					rt |= (model.terms[j].fm[i] & miscex.featuregroups[i]) ^ model.terms[j].vm[i];
					if (!rt) {//mb rt==0
						//maybe add if but for now im optimising for non branching code
						uint32_t hmd = miscex.featuregroups[i] ^ model.terms[j].annotation.featuregroups[i];
						uint32_t curmask = 1;
						for (int k = 0; k < 32; k++) {
							if ((hmd & curmask) && (curmask < model.terms[j].annotation.featuregroups[i])) {
								//check if we have copy bugs
								ndsm newmodel = model;
								newmodel.terms[j].fm[i] |= curmask;
								newmodel.terms[j].vm[i] |= curmask & newmodel.terms[j].annotation.featuregroups[i];
								newmodel.size++;
  								extensions.push_back(newmodel);
							}
							curmask <<= 1;
						}
					}
				}
				if (extensions.size() > 0)
					return extensions;
			}
		}
		//if we got here there are no possible extension sets we can generate from the missclassified example so just return an empty list
		return vector<ndsm>();
	}

	bool FindOptExtSet(const vector<dataset::datapoint>& Ci, const int size, ndsm& model) {
		dataset::datapoint miscex;
		evaluateNDSM(Ci,model,miscex);
		if (model.valid)
			return true;
		if (model.size >= size)
			return false;
		for (ndsm A : FindStrictExtSet(Ci, size, model, miscex)) {
			if (!model.valid || A.size < (model.size-1)) {
				if (FindOptExtSet(Ci, size, A)) {
					if (!model.valid || A.size < model.size)
						model = A;
				}
			}
		}
		return model.valid;
	}

	bool FindOptModel(const vector<dataset::datapoint> &Ci, const int size, ndsm &model) {
		model = ndsm();
		for (dataset::datapoint example : Ci) {
			if (example.result) {
				model.valid = false;
				model.annotation = example;
				model.defaultRule = true;
				model.size = 0;
				break;
			}
		}
		if (FindOptExtSet(Ci, size, model))
			return true;

		model = ndsm();
		for (dataset::datapoint example : Ci) {
			if (!example.result) {
				model.valid = false;
				model.annotation = example;
				model.defaultRule = false;
				model.size = 0;
				break;
			}
		}
		if (FindOptExtSet(Ci, size, model))
			return true;
		return false;
	}
}