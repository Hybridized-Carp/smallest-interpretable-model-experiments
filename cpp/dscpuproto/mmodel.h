#pragma once

#include <deque>
#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <optional>
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
		bool defaultRule;
		vector<nterm> terms;
		dataset::datapoint annotation;
        //I think we should use size as a priority for evaluation
        //though we will want to evaluate smaller models first
        //and generate from bigger models first
		int size;
        bool evaluated;
        optional<mutex> mtx;
        optional<condition_variable> cv;
        bool valid;
        dataset::datapoint miscex;
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

	void evaluateNDSM(const vector<dataset::datapoint> Ci, ndsm& model) {
		for (dataset::datapoint example : Ci) {
			if (example.result != applyNDSM(example, model)) {
				model.valid = false;
                model.evaluated=true;
                model.miscex = example;
				return;
			}
		}
        model.evaluated = true;
		model.valid=true;
	}

    bool modelValid(const vector<dataset::datapoint>& Ci,ndsm& model){
        if (!model.evaluated)
            evaluateNDSM(Ci, model);
        return model.valid;
    }

	bool FindStrictExtSets(const vector<dataset::datapoint>& Ci, const int size, const ndsm& model, deque<ndsm>& extensionsets) {
		bool foundexts= false;
        if (model.size < size){
            if (model.miscex.result != model.defaultRule) {
                //can turn to array by using popc to count number of bits in xor mask
                //can prob then spin off creation of each individual extension set to a unique thread
                for (int i = 0; i < model.miscex.featuregroups.size(); i++) {
                    uint32_t hmd = model.miscex.featuregroups[i] ^ model.annotation.featuregroups[i];
                    uint32_t curmask = 1;
                    for (int j = 0; j < 32; j++) {
                        if (hmd & curmask){
                            foundexts=true;
                            //check if we have copy bugs
                            ndsm newmodel = model;
                            vector<uint32_t> fm = vector<uint32_t>(model.miscex.featuregroups.size(), 0);
                            vector<uint32_t> vm = vector<uint32_t>(model.miscex.featuregroups.size(), 0);
                            fm[i] |= curmask;
                            vm[i] |= curmask & model.miscex.featuregroups[i];
                            nterm newterm({fm,vm, model.miscex});
                            newmodel.terms.push_back(newterm);
                            newmodel.size++;
                            newmodel.evaluated = false;
                            extensionsets.push_front(newmodel);
                        }
                        curmask <<= 1;
                    }
                }
            } else {
                //investigate possibly of modifying all terms that apply to our misclassified example at once
                for (int j = 0; j < model.terms.size(); j++){
                    for (int i = 0; i < model.miscex.featuregroups.size(); i++) {
                        uint32_t rt = 0;
                        rt |= (model.terms[j].fm[i] & model.miscex.featuregroups[i]) ^ model.terms[j].vm[i];
                        if (!rt) {//mb rt==0
                            //maybe add if but for now im optimising for non branching code
                            uint32_t hmd = model.miscex.featuregroups[i] ^ model.terms[j].annotation.featuregroups[i];
                            uint32_t curmask = 1;
                            for (int k = 0; k < 32; k++) {
                                if ((hmd & curmask) && (curmask < model.terms[j].annotation.featuregroups[i])) {
                                    //check if we have copy bugs
                                    foundexts = true;
                                    ndsm newmodel = model;
                                    newmodel.terms[j].fm[i] |= curmask;
                                    newmodel.terms[j].vm[i] |= curmask & newmodel.terms[j].annotation.featuregroups[i];
                                    newmodel.size++;
                                    newmodel.evaluated = false;
                                    extensionsets.push_front(newmodel);
                                }
                                curmask <<= 1;
                            }
                        }
                    }
                    if (foundexts)
                        return foundexts;
                }
            }
        }
		return foundexts;
	}

	bool FindOptExtSet(const vector<dataset::datapoint>& Ci, const int size, ndsm& model, deque<ndsm> &extsets) {
		//prob need some depth/model max size code
        while (!extsets.empty()){
            ndsm extset = extsets.front();
            extsets.pop_front();
            if ((!model.valid || extset.size < model.size) && (extset.size <= size)){
                //all ext sets generated at a given depth have same size
                if (modelValid(Ci,extset)){
                    //we have a valid model
                    //so we dont need to explore further on this branch
                    model=extset;
                } else {
                    FindStrictExtSets(Ci, size, extset, extsets);
                }
            }
        }
		return model.valid;
	}

	bool FindOptModel(const vector<dataset::datapoint> &Ci, const int size, ndsm &model) {
        deque<ndsm> startextsets;
		ndsm newmodel;
        //we can prob use the number of T vs F to decide which we explore first
		for (dataset::datapoint example : Ci) {
			if (example.result) {
				newmodel.valid = false;
				newmodel.annotation = example;
				newmodel.defaultRule = true;
				newmodel.size = 0;
                startextsets.push_back(newmodel);
				break;
			}
		}
        newmodel = ndsm();
        for (dataset::datapoint example : Ci) {
			if (!example.result) {
				newmodel.valid = false;
				newmodel.annotation = example;
				newmodel.defaultRule = false;
				newmodel.size = 0;
                startextsets.push_back(newmodel);
				break;
			}
		}
        if (startextsets.size() == 1){
            //we only have 1 starting ext set
            //-> all of the datapoints have the same result
            //so we can simply return the base rule
            model = startextsets[0];
            model.valid = true;
            return true;
        }
        //we have 2 starting ext sets (or we have an error but fuck error handling)
        //->the starting example in each ext set is misclassified in the other
        startextsets[0].miscex = startextsets[1].annotation;
        startextsets[1].miscex = startextsets[0].annotation;
        if (FindOptExtSet(Ci, size, newmodel,startextsets)){
            model=newmodel;
            return true;
        }
		return false;
	}
}