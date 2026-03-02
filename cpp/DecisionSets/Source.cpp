/*#include <stdio.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <map>

using namespace std;


namespace parser{

    struct symbol {
        string identifier;
        vector<bool> mapping;
    };
    struct column {
        vector<symbol> symbols;
        bool target;
        string Title;
    };
}

int main()
{
    ifstream rawDataset("agaricus-lepiota.data");

    if (rawDataset.bad()) { return -1; }

    vector<string> lines;
    string line;
    int items = -1;

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

    vector<vector<string>> rows;
    for (string line : lines) {
        vector<string> row;
        string item;
        stringstream linestream(line);
        while (getline(linestream, item, ',')) {
            row.push_back(item);
        }
    }

    vector<vector<string>> unique_items_in_column;
    for (int i = 0; i < items; i++) {
        vector<string> column;
        for (vector<string> row : rows) {
            column.push_back(row[i]);
        }
        sort(column.begin(), column.end());
        unique(column.begin(), column.end());
        unique_items_in_column.push_back(column);
    }

    vector<map<string, int>> categorymaps;
    for (vector<string> unique_items : unique_items_in_column) {
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
    }
    return 0;
}*/