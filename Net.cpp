//
// Created by gavin on 2022/2/10.
//

#include "Net.h"
#include <random>

Net::Net() {
    std::mt19937 rd;
    rd.seed(std::random_device()());

    std::uniform_real_distribution<double> distribution(-1, 1);

    // 初始化输入层
    for (size_t i = 0; i < Config::INNODE; ++i) {
        inputLayer[i] = new Node(Config::HIDENODE);
        // 输入层的神经元节点不需要偏置值
        // inputLayer[i]->bias = distribution(rd);
        for (size_t j = 0; j < Config::HIDENODE; ++j) {
            // 初始化输入层第 i 个神经元到隐藏层第 j 个神经元的权重值
            inputLayer[i]->weight[j] = distribution(rd);
            // 初始化输入层第 i 个神经元到隐藏层第 j 个神经元的权重修正值
            inputLayer[i]->weight_delta[j] = 0.f;
        }
    }

    // 初始化隐藏层
    for (size_t j = 0; j < Config::HIDENODE; ++j) {

    }
}

Node::Node(int size) {
    weight.resize(size);
}
