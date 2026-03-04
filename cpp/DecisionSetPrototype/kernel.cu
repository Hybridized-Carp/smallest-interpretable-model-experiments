
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "ClassificationFunctions.h"
#include "examples.cuh"
#include "dataset.h"
#include "awfulkernels.cuh"

#include <stdio.h>
#include <random>
#include <iostream>
#include <vector>






int main()
{
    //map bit representing
    // 2^(x-1)
    // to
    //f1,f2,f3,...
    array<char, 8> map = {27,4,18,20,14,23,6,28};
    std::default_random_engine re(06032026);
    for (int i = 0; i < 5; i++) {
        cout << re() << endl;
    }
    std::vector<dataset::datapoint> examples;
    for (int i = 0; i < 1234; i++) {
        uint32_t temp = re();
        //temp <<= 4;
        //temp >>= 4;
        vector<dataset::featuresetbitmask> vfsbm;
        vfsbm.push_back({ temp,32 });
        examples.push_back(dataset::datapoint({ vfsbm, classificationFunctions::clsf2(temp) }));
    } 

    dataset::exampleset es;
    es.num_examples = examples.size();

    //if we dont have example size % 32 = 0
    //repeat examples to pad to next largest size
    for (int i = 0; i < examples.size() % 32; i++) {
        examples.push_back(examples.back());
    }
    
    for (int i = 0; i < examples.size(); i += 32) {
        dataset::exampleset32 e32;
        int rt = 0;
        for (int j = 0; j < 32; j++) {
            rt <<= 1;
            int t = examples[i+j].result ? 1 : 0;
            rt |= t;
        }
        e32.resultset = rt;
        for (int l = 0; l< examples[i].featuresets.size(); l++){
            for (int k = examples[i].featuresets[l].numfeats - 1; k >= 0; k--) {
                rt = 0;
                for (int j = 0; j < 32; j++)
                {
                    rt <<= 1;
                    uint32_t temp = examples[i + j].featuresets[l].features;
                    temp >>= k;
                    rt |= (temp&1);
                }
                e32.featureset.push_back(rt);
            }
        }
        es.batchedexamples.push_back(e32);
    }
    unsigned int* gpuexamples=0;
    unsigned int* gpuexamplesCompliment = 0;
    unsigned int* gpuresults = 0;
    unsigned int* gpuresultsCompliment = 0;

    //copy dataset once so we dont have to copy each time we want to evaluate a model 
    cudaError_t cudaStatus = LoadDataset(   gpuexamples,  gpuexamplesCompliment, gpuresults, gpuresultsCompliment,es);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    model::term test1;
    //test1oliterals.push_back(28);
    test1.zliterals.push_back(28);
    vector<uint32_t> results;
    cudaStatus = EvaluateDSMTermWithCuda(results, gpuexamples, test1, es.batchedexamples.size(), 32);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }
    
    model::term test2;
    test2.oliterals.push_back(30);
    test2.zliterals.push_back(29);
    vector<uint32_t> results2;
    cudaStatus = EvaluateDSMTermWithCuda(results2, gpuexamples, test2, es.batchedexamples.size(), 32);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    if (results2.size() == results.size()) {
        for (int i = 0; i < results.size(); i++) {
            results[i] &= results2[i];
        }
    }

    cudaFree(gpuexamples);

    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

