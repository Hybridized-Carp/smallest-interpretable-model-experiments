
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

    int padExamples(vector<datapoint> &examples, unsigned int chunk_size = 32) {
        int pad = (chunk_size - (examples.size() % chunk_size)) % chunk_size;
        datapoint e = examples.back();
        for (int i = 0; i < pad; i++)
            examples.push_back(e);
        return pad;
    }

    int padBinaryClassificationInstance(binaryClassificationInstance& bCi) {
        int pad = padExamples(bCi.examples);
        return pad;
    }

    int padExampleFeatures(datapoint &example) {
        int pad = (128 - (example.data.size() % 128)) % 128;
        example.data.insert(example.data.begin(), pad, 0);
        return pad;
    }

    struct transposedbinaryClassificationInstance {
        vector<bool> results;
        vector<vector<bool>> features;
    };

    struct exampleSet64x128 {
        uint64_t results;
        uint64_t features[128];
    };

    struct exampleSet32x128 {
        uint32_t results;
        uint32_t features[128];
    };

    struct exampleSet16x128 {
        uint16_t results;
        uint16_t features[128];
    };

    struct griddedbinaryClassificationInstance64x128 {
        bool uniformData;
        int fpad;
        int zpad;
        int opad;
        int pzexc;
        int poexc;
        //I should prob rewrite this so its more flexible but works for now :P
        vector<exampleSet64x128> examples;
    };

    struct griddedbinaryClassificationInstance32x128 {
        bool uniformData;
        int fpad;
        int zpad;
        int opad;
        int pzexc;
        int poexc;
        //I should prob rewrite this so its more flexible but works for now :P
        vector<exampleSet32x128> examples;
    };

    struct griddedbinaryClassificationInstance16x128 {
        bool uniformData;
        int fpad;
        int zpad;
        int opad;
        int pzexc;
        int poexc;
        //I should prob rewrite this so its more flexible but works for now :P
        vector<exampleSet16x128> examples;
    };
    
    griddedbinaryClassificationInstance64x128 toGbCi64x128_frombCI(binaryClassificationInstance bCi) {
        //chunksize
        unsigned int cs = 64;
        griddedbinaryClassificationInstance64x128 GbCi;
        GbCi.uniformData = true;
        vector<datapoint> zexs;
        vector<datapoint> oexs;
        for (datapoint example : bCi.examples) {
            example.result ? oexs.push_back(example) : zexs.push_back(example);
        }
        int lfpad = -1;
        GbCi.zpad = padExamples(zexs, cs);
        for (int i = 0; i < zexs.size(); i++) {
            int tfpad = padExampleFeatures(zexs[i]);
            if (lfpad == -1)
                lfpad = tfpad;
            else if (lfpad != tfpad)
                GbCi.uniformData = false;
        }

        GbCi.opad = padExamples(oexs, cs);
        for (int i = 0; i < oexs.size(); i++) {
            int tfpad = padExampleFeatures(oexs[i]);
            if (lfpad != tfpad)
                GbCi.uniformData = false;
        }
        GbCi.pzexc = zexs.size();
        GbCi.poexc = oexs.size();
        GbCi.fpad = lfpad;
        if (GbCi.uniformData) {
            //push results
            for (int i = 0; i < (zexs.size() / cs); i++) {
                exampleSet64x128 te;
                te.results = 0;
#pragma unroll
                for (int j = 0; j < 128; j++) {
                    uint64_t rt = 0;
#pragma unroll
                    for (int k = 0; k < cs; k++) {
                        rt |= zexs[((cs * i) + k)].data[j];
                        rt <<= 1;
                    }
                    te.features[j] = rt;
                }

                GbCi.examples.push_back(te);
            }
            for (int i = 0; i < (oexs.size() / cs); i++) {
                exampleSet64x128 te;
                te.results = 0xFFFF;
#pragma unroll
                for (int j = 0; j < 128; j++) {
                    uint64_t rt = 0;
#pragma unroll
                    for (int k = 0; k < cs; k++) {
                        rt |= oexs[((cs * i) + k)].data[j];
                        rt <<= 1;
                    }
                    te.features[j] = rt;
                }

                GbCi.examples.push_back(te);
            }
        }
        return GbCi;
    }

    griddedbinaryClassificationInstance32x128 toGbCi32x128_frombCI(binaryClassificationInstance bCi) {
        griddedbinaryClassificationInstance32x128 GbCi;
        GbCi.uniformData = true;
        vector<datapoint> zexs;
        vector<datapoint> oexs;
        for (datapoint example : bCi.examples) {
            example.result ? oexs.push_back(example) : zexs.push_back(example);
        }
        int lfpad = -1;
        GbCi.zpad = padExamples(zexs);
        for (int i = 0; i < zexs.size(); i++) {
            int tfpad = padExampleFeatures(zexs[i]);
            if (lfpad == -1)
                lfpad = tfpad;
            else if (lfpad != tfpad)
                GbCi.uniformData = false;
        }

        GbCi.opad = padExamples(oexs);
        for (int i = 0; i < oexs.size(); i++) {
            int tfpad = padExampleFeatures(oexs[i]);
            if (lfpad != tfpad)
                GbCi.uniformData = false;
        }
        GbCi.pzexc = zexs.size();
        GbCi.poexc = oexs.size();
        GbCi.fpad = lfpad;
        if (GbCi.uniformData) {
            //push results
            for (int i = 0; i < (zexs.size() / 32); i++) {
                exampleSet32x128 te;
                te.results = 0;
#pragma unroll
                for (int j = 0; j < 128;j++) {
                    uint32_t rt = 0;
#pragma unroll
                    for (int k = 0; k < 32; k++) {
                        rt |= zexs[((32 * i) + k)].data[j];
                        rt <<= 1;
                    }
                    te.features[j] = rt;
                }
                
                GbCi.examples.push_back(te);
            }
            for (int i = 0; i < (oexs.size() / 32); i++) {
                exampleSet32x128 te;
                te.results = 0xFFFFFFFF;
#pragma unroll
                for (int j = 0; j < 128;j++) {
                    uint32_t rt = 0;
#pragma unroll
                    for (int k = 0; k < 32; k++) {
                        rt |= oexs[((32 * i) + k)].data[j];
                        rt <<= 1;
                    }
                    te.features[j] = rt;
                }

                GbCi.examples.push_back(te);
            }
        }
        return GbCi;
    }

    griddedbinaryClassificationInstance16x128 toGbCi16x128_frombCI(binaryClassificationInstance bCi) {
        //chunksize
        unsigned int cs = 16;
        griddedbinaryClassificationInstance16x128 GbCi;
        GbCi.uniformData = true;
        vector<datapoint> zexs;
        vector<datapoint> oexs;
        for (datapoint example : bCi.examples) {
            example.result ? oexs.push_back(example) : zexs.push_back(example);
        }
        int lfpad = -1;
        GbCi.zpad = padExamples(zexs,cs);
        for (int i = 0; i < zexs.size(); i++) {
            int tfpad = padExampleFeatures(zexs[i]);
            if (lfpad == -1)
                lfpad = tfpad;
            else if (lfpad != tfpad)
                GbCi.uniformData = false;
        }

        GbCi.opad = padExamples(oexs,cs);
        for (int i = 0; i < oexs.size(); i++) {
            int tfpad = padExampleFeatures(oexs[i]);
            if (lfpad != tfpad)
                GbCi.uniformData = false;
        }
        GbCi.pzexc = zexs.size();
        GbCi.poexc = oexs.size();
        GbCi.fpad = lfpad;
        if (GbCi.uniformData) {
            //push results
            for (int i = 0; i < (zexs.size() / cs); i++) {
                exampleSet16x128 te;
                te.results = 0;
#pragma unroll
                for (int j = 0; j < 128; j++) {
                    uint16_t rt = 0;
#pragma unroll
                    for (int k = 0; k < cs; k++) {
                        rt |= zexs[((cs * i) + k)].data[j];
                        rt <<= 1;
                    }
                    te.features[j] = rt;
                }

                GbCi.examples.push_back(te);
            }
            for (int i = 0; i < (oexs.size() / cs); i++) {
                exampleSet16x128 te;
                te.results = 0xFFFF;
#pragma unroll
                for (int j = 0; j < 128; j++) {
                    uint16_t rt = 0;
#pragma unroll
                    for (int k = 0; k < cs; k++) {
                        rt |= oexs[((cs * i) + k)].data[j];
                        rt <<= 1;
                    }
                    te.features[j] = rt;
                }

                GbCi.examples.push_back(te);
            }
        }
        return GbCi;
    }

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

    /*binaryClassificationInstance from_transposedbinaryClassificationInstance(transposedbinaryClassificationInstance tbCi) {
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
    }*/
}

struct featureMask16x8 {
    char features[16];
};

struct featureMask2x8x16 {
    short zf[8];
    short of[8];
};

struct featureMask2x4x32 {
    uint32_t zf[4];
    uint32_t of[4];
};

//cudaError_t evaluateTermWithCuda64x128(vector<uint64_t>& results, vector<binaryClassification::exampleSet64x128> examples, featureMask2x8x16 fm);

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
cudaError_t andWithCuda32x128(vector<uint32_t> &results, vector<binaryClassification::exampleSet32x128> examples, featureMask16x8 fm);
cudaError_t evaluateTermWithCuda32x128(vector<uint32_t>& results, vector<binaryClassification::exampleSet32x128> examples, featureMask2x8x16 fm);
cudaError_t andWithCuda16x128(vector<uint16_t>& results, vector<binaryClassification::exampleSet16x128> examples, featureMask16x8 fm);
cudaError_t evaluateTermWithCuda16x128(vector<uint16_t>& results, vector<binaryClassification::exampleSet16x128> examples, featureMask2x4x32 fm);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

__global__ void andKernel16x128(
    uint16_t* results,
    binaryClassification::exampleSet16x128* examples,
    unsigned char fm1,
    unsigned char fm2,
    unsigned char fm3,
    unsigned char fm4,
    unsigned char fm5,
    unsigned char fm6,
    unsigned char fm7,
    unsigned char fm8,
    unsigned char fm9,
    unsigned char fm10,
    unsigned char fm11,
    unsigned char fm12,
    unsigned char fm13, 
    unsigned char fm14, 
    unsigned char fm15,
    unsigned char fm16,
    int datapoints) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < datapoints) {
        uint16_t rt = 0xFFFF;
        //might need to find a dissasembler to work out what the ptx generated is ?
        //i might also be mixing endianness??? is that even the right word -_-
        unsigned char bm = 1;
#pragma unroll
        for (int j = 0; j < 8; j++) {
            rt &= (fm1 & bm) ? examples[i].features[j] : 0xFFFF;
            rt &= (fm2 & bm) ? examples[i].features[8 + j] : 0xFFFF;
            rt &= (fm3 & bm) ? examples[i].features[16 + j] : 0xFFFF;
            rt &= (fm4 & bm) ? examples[i].features[24 + j] : 0xFFFF;
            rt &= (fm5 & bm) ? examples[i].features[32 + j] : 0xFFFF;
            rt &= (fm6 & bm) ? examples[i].features[40 + j] : 0xFFFF;
            rt &= (fm7 & bm) ? examples[i].features[48 + j] : 0xFFFF;
            rt &= (fm8 & bm) ? examples[i].features[56 + j] : 0xFFFF;
            rt &= (fm9 & bm) ? examples[i].features[64 + j] : 0xFFFF;
            rt &= (fm10 & bm) ? examples[i].features[72 + j] : 0xFFFF;
            rt &= (fm11 & bm) ? examples[i].features[80 + j] : 0xFFFF;
            rt &= (fm12 & bm) ? examples[i].features[88 + j] : 0xFFFF;
            rt &= (fm13 & bm) ? examples[i].features[96 + j] : 0xFFFF;
            rt &= (fm14 & bm) ? examples[i].features[104 + j] : 0xFFFF;
            rt &= (fm15 & bm) ? examples[i].features[112 + j] : 0xFFFF;
            rt &= (fm16 & bm) ? examples[i].features[120 + j] : 0xFFFF;
            bm += bm;
        }
        results[i] = rt;
    }
}

__global__ void andKernel32x128(
    uint32_t* results,
    binaryClassification::exampleSet32x128* examples,
    unsigned char fm1,
    unsigned char fm2,
    unsigned char fm3,
    unsigned char fm4,
    unsigned char fm5,
    unsigned char fm6,
    unsigned char fm7,
    unsigned char fm8,
    unsigned char fm9,
    unsigned char fm10,
    unsigned char fm11,
    unsigned char fm12,
    unsigned char fm13,
    unsigned char fm14,
    unsigned char fm15,
    unsigned char fm16,
    int datapoints) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < datapoints) {
        uint32_t rt = 0xFFFFFFFF;
        //might need to find a dissasembler to work out what the ptx generated is ?
        //i might also be mixing endianness??? is that even the right word -_-
        uint32_t bm = 1;
#pragma unroll
        for (int j = 0; j < 8; j++) {
            rt &= (fm1 & bm) ? examples[i].features[j] : 0xFFFFFFFF;
            rt &= (fm2 & bm) ? examples[i].features[8 + j] : 0xFFFFFFFF;
            rt &= (fm3 & bm) ? examples[i].features[16 + j] : 0xFFFFFFFF;
            rt &= (fm4 & bm) ? examples[i].features[24 + j] : 0xFFFFFFFF;
            rt &= (fm5 & bm) ? examples[i].features[32 + j] : 0xFFFFFFFF;
            rt &= (fm6 & bm) ? examples[i].features[40 + j] : 0xFFFFFFFF;
            rt &= (fm7 & bm) ? examples[i].features[48 + j] : 0xFFFFFFFF;
            rt &= (fm8 & bm) ? examples[i].features[56 + j] : 0xFFFFFFFF;
            rt &= (fm9 & bm) ? examples[i].features[64 + j] : 0xFFFFFFFF;
            rt &= (fm10 & bm) ? examples[i].features[72 + j] : 0xFFFFFFFF;
            rt &= (fm11 & bm) ? examples[i].features[80 + j] : 0xFFFFFFFF;
            rt &= (fm12 & bm) ? examples[i].features[88 + j] : 0xFFFFFFFF;
            rt &= (fm13 & bm) ? examples[i].features[96 + j] : 0xFFFFFFFF;
            rt &= (fm14 & bm) ? examples[i].features[104 + j] : 0xFFFFFFFF;
            rt &= (fm15 & bm) ? examples[i].features[112 + j] : 0xFFFFFFFF;
            rt &= (fm16 & bm) ? examples[i].features[120 + j] : 0xFFFFFFFF;
            bm <<= 1;
        }
        results[i] = rt;
    }
}
__global__ void evaluateTermKernel16x128(
    uint16_t* results,
    binaryClassification::exampleSet16x128* examples,
    uint32_t zm1,
    uint32_t zm2,
    uint32_t zm3,
    uint32_t zm4,
    uint32_t om1,
    uint32_t om2,
    uint32_t om3,
    uint32_t om4,
    int datapoints) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < datapoints) {
        uint16_t ot = 0xFFFF;
        uint16_t zt = 0x0;
        //might need to find a dissasembler to work out what the ptx generated is ?
        //i might also be mixing endianness??? is that even the right word -_-
        uint32_t bm=1;
#pragma unroll
        for (int j = 0; j < 32; j++) {
            zt |= (zm1 & bm) ? examples[i].features[j] : 0;
            ot &= (om1 & bm) ? examples[i].features[j] : 0xFFFF;
        }
#pragma unroll
        for (int j = 0; j < 32; j++) {
            j += 32;
            zt |= (zm2 & bm) ? examples[i].features[j] : 0;
            ot &= (om2 & bm) ? examples[i].features[j] : 0xFFFF;
            j += 32;
            zt |= (zm3 & bm) ? examples[i].features[j] : 0;
            ot &= (om3 & bm) ? examples[i].features[j] : 0xFFFF;
            j += 32;
            zt |= (zm4 & bm) ? examples[i].features[j] : 0;
            ot &= (om4 & bm) ? examples[i].features[j] : 0xFFFF;
            j -= 96;
            bm <<= 1;
        }
        results[i] = ot & (~zt);
    }
}

__global__ void evaluateTermKernel32x128(
    uint32_t* results,
    binaryClassification::exampleSet32x128* examples,
    unsigned short zm1,
    unsigned short zm2,
    unsigned short zm3,
    unsigned short zm4,
    unsigned short zm5,
    unsigned short zm6,
    unsigned short zm7,
    unsigned short zm8,
    unsigned short om1,
    unsigned short om2,
    unsigned short om3,
    unsigned short om4,
    unsigned short om5,
    unsigned short om6,
    unsigned short om7,
    unsigned short om8,
    int datapoints) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < datapoints) {
        uint32_t ot = 0xFFFFFFFF;
        uint32_t zt = 0x0;
        //might need to find a dissasembler to work out what the ptx generated is ?
        //i might also be mixing endianness??? is that even the right word -_-
        uint32_t bm = 1;
#pragma unroll
        for (int j = 0; j < 16; j++) {
            zt |= (zm1 & bm) ? examples[i].features[j] : 0;
            zt |= (zm2 & bm) ? examples[i].features[16 + j] : 0;
            zt |= (zm3 & bm) ? examples[i].features[32 + j] : 0;
            zt |= (zm4 & bm) ? examples[i].features[48 + j] : 0;
            zt |= (zm5 & bm) ? examples[i].features[64 + j] : 0;
            zt |= (zm6 & bm) ? examples[i].features[80 + j] : 0;
            zt |= (zm7 & bm) ? examples[i].features[96 + j] : 0;
            zt |= (zm8 & bm) ? examples[i].features[112 + j] : 0;
            bm <<= 1;
        }
        bm = 1;
#pragma unroll
        for (int j = 0; j < 16; j++) {
            ot &= (om1 & bm) ? examples[i].features[j] : 0xFFFFFFFF;
            ot &= (om2 & bm) ? examples[i].features[16 + j] : 0xFFFFFFFF;
            ot &= (om3 & bm) ? examples[i].features[32 + j] : 0xFFFFFFFF;
            ot &= (om4 & bm) ? examples[i].features[48 + j] : 0xFFFFFFFF;
            ot &= (om5 & bm) ? examples[i].features[64 + j] : 0xFFFFFFFF;
            ot &= (om6 & bm) ? examples[i].features[80 + j] : 0xFFFFFFFF;
            ot &= (om7 & bm) ? examples[i].features[96 + j] : 0xFFFFFFFF;
            ot &= (om8 & bm) ? examples[i].features[112 + j] : 0xFFFFFFFF;
            bm <<= 1;
        }
        results[i] = ot & (~zt);
    }
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

    

    //idk how to choose the right data structure so fuck it we run it and fix later
    //code might be nicer if we pad the front of the features but fuck it that should be easy to switch later
    //I think for models with > 128 features we prob just chunk it then reduce?
    auto GbCi32x128 = binaryClassification::toGbCi32x128_frombCI(bCi);

    vector<uint32_t> results;

    featureMask16x8 fm;
    fill(fm.features, fm.features + 16, 0);
    featureMask2x4x32 fm3;
    fill(fm3.zf, fm3.zf + 4, 0);
    fill(fm3.of, fm3.of + 4, 0);

    cudaError_t cudaStatus = andWithCuda32x128(results, GbCi32x128.examples, fm);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "andWithCuda failed!");
        return 1;
    }
    vector<uint32_t> results2;
    featureMask2x8x16 fm2;
    fill(fm2.zf, fm2.zf + 8, 0);
    fill(fm2.of, fm2.of + 8, 0);

    cudaStatus = evaluateTermWithCuda32x128(results2, GbCi32x128.examples, fm2);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "evaluateTermWithCuda failed!");
        return 1;
    }

    auto GbCi16x128 = binaryClassification::toGbCi16x128_frombCI(bCi);

    vector<uint16_t> results3;
    cudaStatus = andWithCuda16x128(results3, GbCi16x128.examples, fm);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "andWithCuda failed!");
        return 1;
    }
    vector<uint16_t> results4;

    cudaStatus = evaluateTermWithCuda16x128(results4, GbCi16x128.examples, fm3);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "evaluateTermWithCuda failed!");
        return 1;
    }
    /*
    const int arraySize = 5;
    vector<int> va = { 1,2,3,4,5 };
    vector<int> vb = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaStatus = addWithCuda(c, &va.front(), &vb.front(), arraySize);
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
    */
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
// Helper function for using CUDA to add vectors in parallel.
cudaError_t andWithCuda32x128(vector<uint32_t> &results , vector<binaryClassification::exampleSet32x128> examples, featureMask16x8 fm)
{
    size_t size = examples.size();
    results = std::vector<uint32_t>(size, 0);
    binaryClassification::exampleSet32x128* dev_examples = 0;
    uint32_t* dev_results = 0;
    cudaError_t cudaStatus;
    
    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output).
    cudaStatus = cudaMalloc((void**)&dev_results, size * sizeof(uint32_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_examples, size * sizeof(binaryClassification::exampleSet32x128));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_examples, &examples.front(), size * sizeof(binaryClassification::exampleSet32x128), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    andKernel32x128 << <24, 11>> > (dev_results, dev_examples,
        fm.features[0],
        fm.features[1],
        fm.features[2],
        fm.features[3],
        fm.features[4],
        fm.features[5],
        fm.features[6],
        fm.features[7],
        fm.features[8],
        fm.features[9],
        fm.features[10],
        fm.features[11],
        fm.features[12],
        fm.features[13],
        fm.features[14],
        fm.features[15],
        size);

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
    cudaStatus = cudaMemcpy(&results.front(), dev_results, size * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    
Error:
    cudaFree(dev_results);
    cudaFree(dev_examples);

    return cudaStatus;
}

cudaError_t evaluateTermWithCuda32x128(vector<uint32_t>& results, vector<binaryClassification::exampleSet32x128> examples, featureMask2x8x16 fm)
{
    size_t size = examples.size();
    results = std::vector<uint32_t>(size, 0);
    binaryClassification::exampleSet32x128* dev_examples = 0;
    uint32_t* dev_results = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output).
    cudaStatus = cudaMalloc((void**)&dev_results, size * sizeof(uint32_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_examples, size * sizeof(binaryClassification::exampleSet32x128));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_examples, &examples.front(), size * sizeof(binaryClassification::exampleSet32x128), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    evaluateTermKernel32x128 << <24, 11 >> > (dev_results, dev_examples,
        fm.zf[0],
        fm.zf[1],
        fm.zf[2],
        fm.zf[3],
        fm.zf[4],
        fm.zf[5],
        fm.zf[6],
        fm.zf[7],
        fm.of[0],
        fm.of[1],
        fm.of[2],
        fm.of[3],
        fm.of[4],
        fm.of[5],
        fm.of[6],
        fm.of[7],
        size);

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
    cudaStatus = cudaMemcpy(&results.front(), dev_results, size * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_results);
    cudaFree(dev_examples);

    return cudaStatus;
}

cudaError_t andWithCuda16x128(vector<uint16_t>& results, vector<binaryClassification::exampleSet16x128> examples, featureMask16x8 fm)
{
    size_t size = examples.size();
    results = std::vector<uint16_t>(size, 0);
    binaryClassification::exampleSet16x128* dev_examples = 0;
    uint16_t* dev_results = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output).
    cudaStatus = cudaMalloc((void**)&dev_results, size * sizeof(uint16_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_examples, size * sizeof(binaryClassification::exampleSet16x128));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_examples, &examples.front(), size * sizeof(binaryClassification::exampleSet16x128), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    andKernel16x128 << <24, 22 >> > (dev_results, dev_examples,
        fm.features[0],
        fm.features[1],
        fm.features[2],
        fm.features[3],
        fm.features[4],
        fm.features[5],
        fm.features[6],
        fm.features[7],
        fm.features[8],
        fm.features[9],
        fm.features[10],
        fm.features[11],
        fm.features[12],
        fm.features[13],
        fm.features[14],
        fm.features[15],
        size);

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
    cudaStatus = cudaMemcpy(&results.front(), dev_results, size * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_results);
    cudaFree(dev_examples);

    return cudaStatus;
}

cudaError_t evaluateTermWithCuda16x128(vector<uint16_t>& results, vector<binaryClassification::exampleSet16x128> examples, featureMask2x4x32 fm)
{
    size_t size = examples.size();
    results = std::vector<uint16_t>(size, 0);
    binaryClassification::exampleSet16x128* dev_examples = 0;
    uint16_t* dev_results = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output).
    cudaStatus = cudaMalloc((void**)&dev_results, size * sizeof(uint16_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_examples, size * sizeof(binaryClassification::exampleSet16x128));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_examples, &examples.front(), size * sizeof(binaryClassification::exampleSet16x128), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    evaluateTermKernel16x128 << <24, 22 >> > (dev_results, dev_examples,
        fm.zf[0],
        fm.zf[1],
        fm.zf[2],
        fm.zf[3],
        fm.of[0],
        fm.of[1],
        fm.of[2],
        fm.of[3],
        size);

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
    cudaStatus = cudaMemcpy(&results.front(), dev_results, size * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_results);
    cudaFree(dev_examples);

    return cudaStatus;
}