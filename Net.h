//
// Created by gavin on 2022/2/10.
//

#ifndef BPNN_NET_H
#define BPNN_NET_H

#include <vector>
#include "Config.h"
#include "Sample.h"

using std::vector;

struct Node {
    double value{}, bias{}, bias_delta{};
    vector<double> weight, weight_delta;

    Node(int size);
};

class Net {
public:
    Node *inputLayer[Config::INNODE]{};
    Node *hideLayer[Config::HIDENODE]{};
    Node *outputLayer[Config::OUTNODE]{};

    Net();

    /**
     * 初始化所有的梯度积累，包括所有节点的 "有意义" 的 weight_delta 和 bias_delta
     * 消除本次迭代的梯度修正，以便进行下一次的迭代
     */
    void grad_zero();

    /**
     * 前向传播
     */
    void forward();

    /**
     * 计算 Loss，以便向下计算梯度
     */
    double CalculateLoss(const vector<double> &out);

    /**
     * 反向传播
     */
     void backward(const vector<double> &out);
};


#endif //BPNN_NET_H
