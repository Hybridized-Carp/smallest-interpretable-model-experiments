
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "nlohmann/json.hpp"


#include <stdio.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <set>
#include <map>
#include <filesystem>

using namespace std;

using json = nlohmann::json;

//this is kinda arbitrary because I could invent arbitrary coding schemes
enum binaryEncoding {
    OneHot = 1,
    Label = 2,
    Ordinal = 3,
    Unary = 4
};

namespace parser {
    struct feature {
        string name;
        bool target;
        map<string, uint8_t> symbols;
        binaryEncoding encodingType;
    };

    void to_json(json& j, const feature& f) {
        j = json{ {"name", f.name}, {"target", f.target}, {"symbols", f.symbols}, {"encoding", f.encodingType} };
    }

    void from_json(const json& j, feature& f) {
        j.at("name").get_to(f.name);
        j.at("target").get_to(f.target);
        j.at("symbols").get_to(f.symbols);
        j.at("encoding").get_to(f.encodingType);
    }
}

namespace genericClassifier {
    struct feature {
        string name;
        uint8_t value;
        uint8_t size;
        binaryEncoding encodingType;
    };
    struct datapoint {
        vector<feature> data;
        bool result;
    };
    struct genericClassificationInstance {
        vector<datapoint> datapoints;
        vector<parser::feature> features;
    };
}

namespace binaryClassification {
    struct datapoint {
        vector<bool> data;
        bool result;
    };

    struct binaryClassificationInstance {
        vector<datapoint> examples;
        vector<genericClassifier::feature> features;
    };

    binaryClassificationInstance from_genericClassificationInstance(genericClassifier::genericClassificationInstance genericClassifierInstance) {
        vector<datapoint> bdatapoints;
        for (genericClassifier::datapoint orgdatapoint : genericClassifierInstance.datapoints) {
            vector<bool> data;
            for (genericClassifier::feature feature : orgdatapoint.data)
                switch (feature.encodingType) {
                case (binaryEncoding::OneHot):
                    for (int j = 0; j < feature.size; j++)
                        data.push_back(feature.value == j);
                    break;
                case (binaryEncoding::Label):
                    for (int j = 1; j < feature.size; j++)
                        data.push_back(feature.value == j);
                    break;
                case (binaryEncoding::Ordinal):
                    //can't be fucked rn
                    break;
                case(binaryEncoding::Unary):
                    for (int j = 1; j < feature.size; j++)
                        data.push_back(feature.value >= j);
                    break;
                }

            bdatapoints.push_back({ data, orgdatapoint.result });
        }
        return { bdatapoints , genericClassifierInstance.datapoints[0].data };
    }
    int sortBinaryClassificationInstance(binaryClassificationInstance& bCi) {
        vector<datapoint> sortedexamples;
        int zc = 1;
        for (datapoint example : bCi.examples) {
            example.result ? sortedexamples.insert(sortedexamples.end(), example) : sortedexamples.insert(sortedexamples.begin(), example);
            if (!example.result)
                zc++;
        }
        bCi.examples = sortedexamples;
        return zc;
    }

    struct transposedbinaryClassificationInstance {
        vector<bool> results;
        vector<vector<bool>> features;
    };
    
    transposedbinaryClassificationInstance from_binaryClassificationInstance(binaryClassificationInstance bCi) {
        transposedbinaryClassificationInstance tbCi;
        for (datapoint example : bCi.examples)
            tbCi.results.push_back(example.result);
        for (int i = 0; i < bCi.examples[0].data.size(); i++) {
            vector<bool> f;
            for (datapoint example : bCi.examples)
                f.push_back(example.data[i]);
            tbCi.features.push_back(f);
        }
        return tbCi;
    }

    binaryClassificationInstance from_transposedbinaryClassificationInstance(transposedbinaryClassificationInstance tbCi) {
        binaryClassificationInstance bCi;
        for (bool result : tbCi.results) {
            datapoint d;
            d.result = result;
            bCi.examples.push_back(d);
        }

        for (vector<bool> f : tbCi.features) {
            for (int i = 0; i < tbCi.results.size(); i++)
                bCi.examples[i].data.push_back(f[i]);
        }
        return bCi;
    }
}


cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}



int main()
{
    string Filename = "agaricus-lepiota.data";
    cout << "Opening " << Filename << endl;
    ifstream rawDataset(Filename);
    cout << filesystem::current_path();
    if (!rawDataset.is_open()) {
        cout << "Failed to Open " << Filename << endl;
        return -1;
    }

    vector<string> lines;
    string line;
    int items = -1;

    cout << "Reading lines" << endl;
    //read in lines
    while (getline(rawDataset, line)) {
        if (!line.empty()) {
            if (items == -1)
                items = count(line.begin(), line.end(), ',');
            else
                if (items != count(line.begin(), line.end(), ','))
                    return -2;
            lines.push_back(line);
        }
    }

    rawDataset.close();

    //add one to items because we actually count the number of comma's
    items++;

    cout << "Splitting lines into row elements" << endl;
    vector<vector<string>> rows;
    for (string line : lines) {
        vector<string> row;
        string item;
        stringstream linestream(line);
        while (getline(linestream, item, ',')) {
            row.push_back(item);
        }
        rows.push_back(row);
    }

    string schemaFilename = Filename + "_schema.json";
    cout << "Checking for " << schemaFilename << endl;
    vector<parser::feature> features;
    if (!filesystem::exists(schemaFilename))
    {
        cout << "Schema not found, generating template schema" << endl;
        cout << "Finding unique items in columns" << endl;

        vector<set<string>> unique_items_in_column;
        for (int i = 0; i < items; i++) {
            set<string> column;
            for (vector<string> row : rows) {
                column.insert(row[i]);
            }
            unique_items_in_column.push_back(column);
        }

        cout << "Creating feature list" << endl;
        for (set<string> unique_items : unique_items_in_column) {
            //use category toggle if we only have one item
            //might be incorrect but oh well
            uint8_t fi = 0;
            map<string, uint8_t> symbols;
            for (string unique_item : unique_items) {
                symbols.insert({ unique_item, fi });
                fi++;
            }
            parser::feature pf = { "unknown",false,symbols,unique_items.size() != 2 ? binaryEncoding::OneHot : binaryEncoding::Label };
            features.push_back(pf);
        }
        cout << "Writing " << schemaFilename << endl;
        json j = features;
        ofstream schemaStream(schemaFilename);
        schemaStream << j.dump(4);
        schemaStream.close();
        cout << "Wrote " << schemaFilename << endl;
        cout << "Please set a target feature and then rerun the program";
        return -3;
    }
    cout << "Reading " << schemaFilename << endl;
    ifstream fileSchema(schemaFilename);

    if (!fileSchema.is_open()) {
        cout << "Failed to Open " << schemaFilename << endl;
        return -4;
    }
    cout << "Parsing " << schemaFilename << endl;
    json data = json::parse(fileSchema);
    features = data.get<vector<parser::feature>>();
    cout << "Closing " << schemaFilename << endl;
    fileSchema.close();
    cout << "Searching for target feature" << endl;
    int targetFeatureIndex = -1;
    for (int i = 0; i < features.size(); i++)
    {
        if (features[i].target && targetFeatureIndex == -1 && features[i].symbols.size() == 2) {
            targetFeatureIndex = i;
        }
        else if (features[i].target) {
            cout << "Too many target features found or target feature has more than 2 symbols";
            return -5;
        }
    }
    if (targetFeatureIndex == -1) {
        cout << "Please set a target feature and then rerun the program";
        return -6;
    }

    //bit of testing code
    int numbits = 0;
    for (parser::feature feature : features)
    {
        if (feature.target)
            continue;
        switch (feature.encodingType) {
        case (binaryEncoding::OneHot):
            numbits += feature.symbols.size();
            break;
        case (binaryEncoding::Label):
        case (binaryEncoding::Unary):
            numbits += feature.symbols.size() - 1;
            break;
        case (binaryEncoding::Ordinal):
            uint8_t size = feature.symbols.size();
            int bits = 0;
            while (size >>= 1)
                bits++;
            numbits += bits;
            break;
        }

    }

    cout << "applying feature maps to rows to create generic classifier instance" << endl;
    vector<genericClassifier::datapoint> datapoints;
    for (vector<string> row : rows) {
        genericClassifier::datapoint datapoint;
        for (int i = 0; i < row.size(); i++) {
            parser::feature feature = features[i];
            if (feature.target)
                datapoint.result = (bool)feature.symbols[row[i]];
            else {
                genericClassifier::feature clfeature{ feature.name, feature.symbols[row[i]], feature.symbols.size(), feature.encodingType };
                datapoint.data.push_back(clfeature);
            }
        }
        datapoints.push_back(datapoint);
    }
    genericClassifier::genericClassificationInstance gci = { datapoints, features };
    binaryClassification::binaryClassificationInstance bCi = binaryClassification::from_genericClassificationInstance(gci);
    cout << "Hello CMake." << endl;

    int f1pos = binaryClassification::sortBinaryClassificationInstance(bCi);
    //idk how to choose the right data structure so fuck it we run it and fix later

    binaryClassification::transposedbinaryClassificationInstance tbCi = binaryClassification::from_binaryClassificationInstance(bCi);

    vector<char> cresults;
    for (bool b : tbCi.results) {
        cresults.push_back(b);
    }

    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    vector<int> va = { 1,2,3,4,5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    vector<int> vb = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, &va.front(), &vb.front(), arraySize);
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

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output).
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

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
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
