#include <stdio.h>
#include <random>
#include <iostream>
#include <vector>
#include <hpx/hpx_main.hpp>
#include <hpx/iostream.hpp>
#include "dataset.h"
#include "omodel.h"
#include "ClassificationFunctions.h"

int main(){
    hpx::cout << "Hello World!" <<std::endl;
    array<char, 8> map = {27,4,18,20,14,23,6,28};
    std::default_random_engine re(06032026);
    for (int i = 0; i < 5; i++) {
        hpx::cout << re() << endl;
    }

    std::vector<dataset::datapoint> examples;
    //4254
    for (int i = 0; i <  1234; i++) {
        uint32_t temp = re();
        //temp <<= 16;
        //temp >>= 16;
        vector<uint32_t> vfg;
        vfg.push_back(temp);
        examples.push_back(dataset::datapoint({ vfg, classificationFunctions::clsf2(temp)}));
    }
    model::ndsm m;
    if (model::FindOptModel(examples, 5, m)){
        hpx::cout << "Yay" << endl;
        hpx::cout << "Model Valid: " << (m.valid ? "True" : "False") << endl;
        hpx::cout << "Model Size: " << m.size << endl;
        hpx::cout << "Default Rule: " << (m.defaultRule ? "True": "False") << endl;
        hpx::cout << "Number of Terms: " << m.terms.size() << endl;
    }
    else {
        hpx::cout << "Boo";
    }

    
}