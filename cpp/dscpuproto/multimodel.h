#pragma once

#include <deque>
#include <hpx/future.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/components.hpp>
#include "dataset.h"
#include <tuple>
#include <optional>;
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

    struct ndsmeval : public hpx::components::component_base<ndsmeval> {
        bool valid;
        std::optional<dataset::datapoint> miscex;
    };

	struct ndsm {
		bool defaultRule;
		vector<nterm> terms;
		dataset::datapoint annotation;
		int size;
        bool evaluated;
        bool valid;
        hpx::shared_future<ndsmeval> eval;
        dataset::datapoint miscex;
	};

#define lol 1

#ifdef lol

	bool applyNDSM(dataset::datapoint example, const ndsm& model) {
        for (int j = 0; j < model.terms.size(); j++) {
            uint32_t rt = 0;
            for (int i = 0; i < example.featuregroups.size(); i++) {
                uint32_t ft = (model.terms[j].fm[i] & example.featuregroups[i]);
                ft ^= model.terms[j].vm[i];
                rt |= ft;
            }
            if (!rt)//mb rt==0
                return !model.defaultRule;
        }
        return model.defaultRule;
	}

#else
    bool applyNDSM(dataset::datapoint example, const ndsm& model) {
        for (nterm t : model.terms) {
            uint32_t rt = 0;
            for (int i = 0; i < example.featuregroups.size(); i++) {
                rt |= (t.fm[i] & example.featuregroups[i]) ^ t.vm[i];
            }
            if (!rt)//mb rt==0
                return !model.defaultRule;
        }
        return model.defaultRule;
    }
#endif
    std::uint64_t fibonacci(std::uint64_t n)
    {
        if (n < 2)
            return n;

        hpx::future<std::uint64_t> n1 = hpx::async(fibonacci, n - 1);
        std::uint64_t n2 = fibonacci(n - 2);

        return n1.get() + n2;    // wait for the Future to return their values
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

    ndsmeval evaluateNDSM(const vector<dataset::datapoint> Ci, const ndsm model) {
        for (dataset::datapoint example : Ci) {
            if (example.result != applyNDSM(example, model)) {
                return {false, example};
            }
        }
        return {true};
    }

    /*bool modelValid(const vector<dataset::datapoint>& Ci, ndsm& model) {
        if (!model.evaluated)
            evaluateNDSM(Ci, model);
        return model.valid;
    }*/

    bool FindStrictExtSetsND(const vector<dataset::datapoint>& Ci, const int size, const ndsm& model, deque<ndsm>& extensionsets) {
        //can turn to array by using popc to count number of bits in xor mask
        //can prob then spin off creation of each individual extension set to a unique thread
        ndsm newmodel = model;
        nterm newterm;
        newterm.annotation = model.miscex;
        newterm.fm = vector<uint32_t>(model.miscex.featuregroups.size(), 0);
        newterm.vm = vector<uint32_t>(model.miscex.featuregroups.size(), 0);
        newmodel.terms.push_back(newterm);
        newmodel.size++;
        newmodel.evaluated = false;
        int k = model.terms.size();
        for (int i = 0; i < model.miscex.featuregroups.size(); i++) {
            uint32_t hmd = model.miscex.featuregroups[i] ^ model.annotation.featuregroups[i];
            uint32_t curmask = 1;
            for (int j = 0; j < 32; j++) {
                if (hmd & curmask) {
                    //check if we have copy bugs
                    newmodel.terms[k].fm[i] = curmask;  
                    newmodel.terms[k].vm[i] = curmask & model.miscex.featuregroups[i];
                    newmodel.eval = hpx::async(evaluateNDSM, Ci, newmodel);
                    extensionsets.push_front(newmodel);
                    //evaluateNDSM(Ci,newmodel);
                    /*if (newmodel.valid) {
                        //we find a valid extensionset
                        //we wont grow deeper than this
                        //so we dont need to return anymore extension sets
                        return true;
                    }*/
                }
                curmask <<= 1;
            }
            newmodel.terms[k].fm[i] = 0;
            newmodel.terms[k].vm[i] = 0;
        }
    }

    void FindStrictExtSetsDF(const vector<dataset::datapoint>& Ci, const int size, const ndsm& model, deque<ndsm>& extensionsets) {
        ndsm newmodel = model;
        newmodel.size++;
        newmodel.evaluated = false;
        //investigate possibly of modifying all terms that apply to our misclassified example at once
        for (int j = 0; j < model.terms.size(); j++) {
            uint32_t rt = 0;
            for (int i = 0; i < model.miscex.featuregroups.size(); i++) {
                rt |= (model.terms[j].fm[i] & model.miscex.featuregroups[i]) ^ model.terms[j].vm[i];
            }
            if (!rt) {//mb rt==0
                //maybe add if but for now im optimising for non branching code
                for (int i = 0; i < model.miscex.featuregroups.size(); i++) {
                    uint32_t hmd = model.miscex.featuregroups[i] ^ model.terms[j].annotation.featuregroups[i];
                    uint32_t curmask = 1;
                    for (int k = 0; k < 32; k++) {
                        if ((hmd & curmask) && (curmask < model.terms[j].annotation.featuregroups[i])) {
                            newmodel.terms[j].fm[i] = model.terms[j].fm[i] | curmask;
                            newmodel.terms[j].vm[i] = model.terms[j].vm[i] | (curmask & model.terms[j].annotation.featuregroups[i]);
                            extensionsets.push_front(newmodel);
                            //evaluateNDSM(Ci,newmodel);
                            /*if (newmodel.valid) {
                                //we find a valid extensionset
                                //we wont grow deeper than this
                                //so we dont need to return anymore extension sets
                                return true;
                            }*/
                        }
                        curmask <<= 1;
                    }
                }
                return;
            }
        }
        return;
    }


	bool FindOptExtSet(const vector<dataset::datapoint>& Ci, const int size, ndsm& model, deque<ndsm> &extsets) {
		//prob need some depth/model max size code
        while (!extsets.empty()){
            ndsm extset = extsets.front();
            extsets.pop_front();
            if ((!model.eval.get().valid || extset.size < model.size) && (extset.size <= size)){
                //all ext sets generated at a given depth have same size
                if (extset.eval.get().valid){
                    //we have a valid model
                    //so we dont need to explore further on this branch
                    model=extset;
                } else {
                    if (model.size < size && (!model.valid || (extset.size < (model.size-1)))) {
                        if (extset.miscex.result != extset.defaultRule) {
                            FindStrictExtSetsND(Ci, size, extset, extsets);
                        }
                        else {
                            FindStrictExtSetsDF(Ci, size, extset, extsets);
                        }
                    }
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