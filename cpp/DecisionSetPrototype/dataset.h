#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <vector>
#include <array>

namespace dataset {
    using namespace std;
    struct featuresetbitmask {
        uint32_t features;
        uint16_t numfeats;
    };

    struct datapoint {
        //each bit represents a feature
        vector<featuresetbitmask> featuresets;
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