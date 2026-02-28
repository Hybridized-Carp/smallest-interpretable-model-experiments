// BDSCMAKE.cpp : Defines the entry point for the application.
//

#include "BDSCMAKE.h"
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

namespace parser {
    //this is kinda arbitrary because I could invent arbitrary coding schemes
    enum encoding {
        OneHot = 1,
        Label = 2,
        Ordinal=3,
        Unary=4
    };
    struct feature {
        string name;
        bool target;
        map<string,int> symbols;
        encoding encodingType;
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

namespace dsm {
    struct model {
        vector<literal> literals;
        bool defaultValue;
    };
    struct literal {
        vector<bool> terms;
        vector<bool> annotation;
    };
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
            int fi = 0;
            map<string, int> symbols;
            for (string unique_item : unique_items) {
                symbols.insert({ unique_item, fi });
                fi++;
            }
            features.push_back(
                parser::feature(
                    "unknown",
                    false,
                    symbols,
                    unique_items.size() == 2 ? parser::encoding::Label : parser::encoding::OneHot
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
        if (features[i].target && targetFeatureIndex ==-1 && features[i].symbols.size() != 2) {
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
    int numbits = 0;
    for (parser::feature feature : features)
    {
        if (feature.target)
            continue;
        switch (feature.encodingType) {
            case (parser::encoding::OneHot):
                numbits += feature.symbols.size();
                break;
            case (parser::encoding::Label):
            case (parser::encoding::Unary):
                numbits += feature.symbols.size() - 1;
                break;
            case (parser::encoding::Ordinal):
                uint8_t size = feature.symbols.size();
                int bits = 0;
                while (size >>= 1)
                    bits++;
                numbits += bits;
                break;
        }
        
    }
    /*cout << "applying feature maps to rows to create boolean vectors" << endl;
    vector<vector<bool>> datapoints;
    for (vector<string> row : rows) {
        vector<bool> datapoint;
        for (int i = 0; i < row.size(); i++) {
            parser::feature feature = features[i];
            map<string, int> categorymap = features[i].symbols;
            string feature = row[i];
            if (categorymap.size() == 2)
                datapoint.push_back(categorymap[feature] == 1);
            else {
                for (int j = 0; j < categorymap.size(); j++)
                {
                    datapoint.push_back(categorymap[feature] == j);
                }
            }
        }
        datapoints.push_back(datapoint);
    }*/
	cout << "Hello CMake." << endl;
	return 0;
}
