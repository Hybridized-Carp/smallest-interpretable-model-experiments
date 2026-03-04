#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "dataset.h"
#include "model.h"
#include <stdio.h>
#include <map>
#include "cccl/cuda/cmath"


__global__ void GetComplement(unsigned int* datasetCompliment, const unsigned int* dataset, int size) {
    int workid = threadIdx.x + blockDim.x * blockIdx.x;
    if (workid < size) {
        datasetCompliment[workid] = ~dataset[workid];
    }
}

// Helper function for loading our dataset onto the GPU
// We only need to do this once for the lifecycle of our application
cudaError_t LoadDataset(unsigned int*& dev_es, unsigned int*& dev_ecs, unsigned int*& dev_rs, unsigned int*& dev_rcs,  dataset::exampleset es)
{
    //double mem consumption for increased performance
    //test whether it was worth it?
    dev_es = 0;
    dev_ecs = 0;
    dev_rs = 0;
    dev_rcs = 0;
    cudaError_t cudaStatus;
    vector<uint32_t> examples;
    vector<uint32_t> results;
    for (dataset::exampleset32 es32 : es.batchedexamples)
    {
        results.push_back(es32.resultset);
        for (uint32_t feature : es32.featureset)
            examples.push_back(feature);
    }
    unsigned int size = examples.size();

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for examplesets.
    cudaStatus = cudaMalloc((void**)&dev_es, examples.size()* sizeof(uint32_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy batched examplesets
    cudaStatus = cudaMemcpy(dev_es, &examples.front(), examples.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Allocate GPU buffers for examplecomplementsets.
    cudaStatus = cudaMalloc((void**)&dev_ecs, examples.size() * sizeof(uint32_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    GetComplement<<<4,(size+4)/4>>>(dev_ecs, dev_es, size);
    
    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }
    
    // Allocate GPU buffers for resultsets.
    cudaStatus = cudaMalloc((void**)&dev_rs, results.size() * sizeof(uint32_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy batched examplesets
    cudaStatus = cudaMemcpy(dev_rs, &results.front(), results.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Allocate GPU buffers for resultcomplementsets.
    cudaStatus = cudaMalloc((void**)&dev_rcs, results.size() * sizeof(uint32_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    
    GetComplement<<<1, results.size()>>>(dev_rcs, dev_rs, results.size());
    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    return cudaStatus;

Error:
    cudaFree(dev_es);
    cudaFree(dev_ecs);
    cudaFree(dev_rs);
    cudaFree(dev_rcs);

    return cudaStatus;
}

__global__ void EvaluateDecisionSetTermKernel(uint32_t* results, const unsigned int* dev_es, const unsigned int* zlits, const unsigned int* olits, int size, int nzlits, int nolits, int features)
{
    int workid = threadIdx.x + blockDim.x * blockIdx.x;
    if (workid < size) {
        uint32_t rt = 0;
        for (int i = 0; i < nolits; i++) {
            uint32_t ol = olits[i];
            uint32_t temp = ~dev_es[workid * features+ol];
            rt |= temp;
        }
        for (int i = 0; i < nzlits; i++) {
            uint32_t temp = dev_es[workid * features + zlits[i]];
            rt |= temp;
        }
        results[workid] = rt;
    }
}

cudaError_t EvaluateDSMTermWithCuda(vector<uint32_t>& results, const unsigned int* dev_es, model::term term, int size, int features) {
    unsigned int* dev_zlits = 0;
    unsigned int* dev_olits = 0;
    unsigned int* dev_results = 0;
    cudaError_t cudaStatus;
    results = std::vector<uint32_t>(size, 0);

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_results, size * sizeof(unsigned int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_zlits, term.zliterals.size() * sizeof(unsigned int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_olits, term.oliterals.size() * sizeof(unsigned int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    if (term.zliterals.size() > 0) {
        // Copy input vectors from host memory to GPU buffers.
        cudaStatus = cudaMemcpy(dev_zlits, &term.zliterals.front(), term.zliterals.size() * sizeof(unsigned int), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }
    }

    if (term.oliterals.size() > 0) {
        cudaStatus = cudaMemcpy(dev_olits, &term.oliterals.front(), term.oliterals.size() * sizeof(unsigned int), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }
    }

    // Launch a kernel on the GPU with one thread for each element.
    EvaluateDecisionSetTermKernel<<<1, size>>>(dev_results, dev_es, dev_zlits, dev_olits, size, term.zliterals.size(), term.oliterals.size(), features);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(&results.front(), dev_results, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_results);
    cudaFree(dev_zlits);
    cudaFree(dev_olits);

    return cudaStatus;
}

/*cudaError_t EvaluateDSMWithCuda(std::map<model::term, vector<uint32_t>>& results, const unsigned int* dev_es, model::decisionSetModel dsm, int size, int features)
{
    results = std::map<model::term, vector<uint32_t>>();
    cudaError_t cudaStatus;
    for (model::term t : dsm.terms) {
        vector<uint32_t> result = std::vector<uint32_t>(size, 0);
        cudaStatus = EvaluateDSMTermWithCuda(result, dev_es, t, size, features);
        if (cudaStatus != cudaSuccess)
            return cudaStatus;
        if (!dsm.defaultRule) {
            for (int i = 0; i < result.size(); i++)
                result[i] = ~result[i];
        }
        results.insert({ t, result });
    }
    return cudaStatus;
}
*/