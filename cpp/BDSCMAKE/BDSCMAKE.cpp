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
    enum encoding {
        OneHot = 1,
        Label = 2,

    };
    struct symbol {
        string identifier;
        vector<char> mapping;
    };
    struct column {
        vector<symbol> symbols;
        bool target;
        string Title;
    };
}


int main()
{
    string Filename = "agaricus-lepiota.data";
    string schemaFilename = Filename + "_schema.json";
    cout << "Opening " << Filename << endl;
    ifstream rawDataset("agaricus-lepiota.data");

    if (!rawDataset.is_open()) { return -1; }

    vector<string> lines;
    string line;
    int items = -1;

    cout << "reading lines" << endl;
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

    cout << "turning lines into rows" << endl;
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

    cout << "finding unique items in column" << endl;

    vector<set<string>> unique_items_in_column;
    for (int i = 0; i < items; i++) {
        set<string> column;
        for (vector<string> row : rows) {
            column.insert(row[i]);
        }
        unique_items_in_column.push_back(column);
    }

    cout << "creating feature maps" << endl;
    vector<map<string, int>> categorymaps;
    for (set<string> unique_items : unique_items_in_column) {
        //use category toggle if we only have one item
        //might be incorrect but oh well
        map<string, int> categorymap;
        int fm = 0;
        for (string unique_item : unique_items) {
            categorymap.insert({ unique_item, fm });
            fm++;
        }
        categorymaps.push_back(categorymap);
    }

    cout << "applying feature maps to rows to create boolean vectors" << endl;
    vector<vector<bool>> datapoints;
    for (vector<string> row : rows) {
        vector<bool> datapoint;
        for (int i = 0; i < row.size(); i++) {
            map<string, int> categorymap = categorymaps[i];
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
    }
	cout << "Hello CMake." << endl;
	return 0;
}
