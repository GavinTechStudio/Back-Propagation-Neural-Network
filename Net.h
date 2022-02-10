//
// Created by gavin on 2022/2/10.
//

#ifndef BPNN_NET_H
#define BPNN_NET_H

#include <vector>
#include "Config.h"

using std::vector;

struct Node {
    double value, bias, bias_delta;
    vector<double> weight, weight_delta;

    Node(int size);
};

class Net {
public:
    Node *inputLayer[Config::INNODE]{};
    Node *hideLayer[Config::HIDENODE]{};
    Node *outputLayer[Config::OUTNODE]{};

    Net();
};


#endif //BPNN_NET_H
