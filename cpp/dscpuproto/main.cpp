#include <stdio.h>
#include <random>
#include <iostream>
#include <vector>
#include "dataset.h"
#include "omodel.h"
#include "ClassificationFunctions.h"

int main(){
    std::cout << "Hello World!" <<std::endl;
    array<char, 8> map = {27,4,18,20,14,23,6,28};
    std::default_random_engine re(06032026);
    for (int i = 0; i < 5; i++) {
        cout << re() << endl;
    }

    std::vector<dataset::datapoint> examples;
    //4254
    for (int i = 0; i <  2048; i++) {
        uint32_t temp = re();
        //temp <<= features;
        //temp >>= features;
        vector<uint32_t> vfg;
        vfg.push_back(temp);
        examples.push_back(dataset::datapoint({ vfg, classificationFunctions::clsf2(temp)}));
    }
    model::ndsm m;
    if (model::FindOptModel(examples, 5, m)){
        cout << "Yay" << endl;
        cout << "Model Valid: " << (m.valid ? "True" : "False") << endl;
        cout << "Model Size: " << m.size << endl;
        cout << "Default Rule: " << (m.defaultRule ? "True": "False") << endl;
        cout << "Number of Terms: " << m.terms.size() << endl;
    }

    
}