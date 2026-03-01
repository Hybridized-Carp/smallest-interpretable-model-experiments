// BDSCMAKE.cpp : Defines the entry point for the application.
//

#include "BDSCMAKE.h"
#include "nlohmann/json.hpp"

#include <gf2/namespace.h>
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
enum binaryEncoding{
    OneHot = 1,
    Label = 2,
    Ordinal = 3,
    Unary = 4
};

namespace parser {
    struct feature {
        string name;
        bool target;
        map<string,uint8_t> symbols;
        binaryEncoding encodingType;
    };

    void to_json(json& j, const feature& f) {
        j = json{ {"name", f.name}, {"target", f.target}, {"symbols", f.symbols}, {"encoding", f.encodingType}};
    }

    void from_json(const json& j, feature& f) {
        j.at("name").get_to(f.name);
        j.at("target").get_to(f.target);
        j.at("symbols").get_to(f.symbols);
        j.at("encoding").get_to(f.encodingType);
    }
}

namespace decisionSet {
    struct literal {
        boost::dynamic_bitset<uint8_t> featureMask;
        boost::dynamic_bitset<uint8_t> value;
        boost::dynamic_bitset<uint8_t> annotation;
    };
    struct baseCase {
        bool defaultValue;
        boost::dynamic_bitset<uint8_t> annotation;
    };
    struct model {
        vector<literal> literals;
        baseCase bc;
    };


    bool EvaluateModel(vector<binaryClassification::datapoint> &examples, model& dsm, binaryClassification::datapoint& misclassifiedExample) {
        for (binaryClassification::datapoint example : examples) {
            for (literal l : dsm.literals) {
                //if ((example.data & l.featureMask)==l.value)
            }
        }
        return true;
    }

    size_t modelSize(model model) {
        size_t size = 0;
        for (literal literal : model.literals)
            size += literal.featureMask.count();
        return size;
    }

    bool FindOptModelStr(vector<binaryClassification::datapoint> examples, size_t size, model& dsm) {
        boost::dynamic_bitset<uint8_t> iA;
        for (binaryClassification::datapoint example : examples) {
            if (!example.result) {
                iA = example.data;
                break;
            }
        }
        dsm.bc = {false, iA};
        if (findOptExtSet(examples, size, dsm))
            return true;

        for (binaryClassification::datapoint example : examples) {
            if (example.result) {
                iA = example.data;
                break;
            }
        }

        return findOptExtSet(examples, size, dsm);
    }

    bool findOptExtSet(vector<binaryClassification::datapoint> examples, size_t size, model& dsm) {
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
    struct genericClassificationInstance{
        vector<datapoint> datapoints;
        vector<parser::feature> features;
    };
}

namespace binaryClassification {
    struct datapoint {
        boost::dynamic_bitset<uint8_t> data;
        bool result;
    };

    struct binaryClassificationInstance {
        vector<datapoint> examples;
        vector<genericClassifier::feature> features;
    };

    binaryClassificationInstance to_binaryClassificationInstance(genericClassifier::genericClassificationInstance genericClassifierInstance) {
        vector<datapoint> bdatapoints;
        for (genericClassifier::datapoint orgdatapoint: genericClassifierInstance.datapoints) {
            boost::dynamic_bitset<uint8_t> data;
            for (genericClassifier::feature feature: orgdatapoint.data)
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
            
            bdatapoints.push_back({ data, orgdatapoint.result});
        }
        return { bdatapoints , genericClassifierInstance.datapoints[0].data};
    }
}

int main()
{
    string Filename = "agaricus-lepiota.data";
    cout << "Opening " << Filename << endl;
    ifstream rawDataset(Filename);

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
    cout << "Checking for "<< schemaFilename << endl;
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
            features.push_back(
                parser::feature(
                    "unknown",
                    false,
                    symbols,
                    unique_items.size() != 2 ? binaryEncoding::OneHot : binaryEncoding::Label
                )
            );
        }
        cout << "Writing "<< schemaFilename << endl;
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
        if (features[i].target && targetFeatureIndex ==-1 && features[i].symbols.size() == 2) {
            targetFeatureIndex = i;
        } else if (features[i].target) {
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

    cout << "applying feature maps to rows to create boolean vectors" << endl;
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
    vector<binaryClassification::datapoint> bdatapoints = binaryClassification::to_binaryClassificationInstance(datapoints);
	cout << "Hello CMake." << endl;
	return 0;
}
