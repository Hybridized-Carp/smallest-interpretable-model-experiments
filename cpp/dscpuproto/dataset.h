#pragma once
#include <stdio.h>
#include <vector>
#include <array>
#define uint32_t unsigned int

namespace dataset {
    using namespace std;

    struct datapoint {
        //each bit represents a feature
        vector<uint32_t> featuregroups;
          bool result;
    };

    struct feature {
        uint32_t datapointset;
    };

    //host representation
    struct exampleset32 {
        vector<uint32_t> featureset;
        uint32_t resultset;
    };
    struct exampleset {
        vector<exampleset32> batchedexamples;
        uint32_t num_examples;
    };
}