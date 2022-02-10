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
            // 初始化输入层第 i 个神经元到隐藏层第 j 个神经元的权重
            inputLayer[i]->weight[j] = distribution(rd);
            // 初始化输入层第 i 个神经元到隐藏层第 j 个神经元的权重修正值
            inputLayer[i]->weight_delta[j] = 0.f;
        }
    }

    // 初始化隐藏层
    for (size_t j = 0; j < Config::HIDENODE; ++j) {
        hideLayer[j] = new Node(Config::OUTNODE);
        // 初始化隐藏层神经元系节点偏置值
        hideLayer[j]->bias = distribution(rd);
        // 初始化隐藏层神经元系节点偏置值修正值
        hideLayer[j]->bias_delta = 0.f;
        for (size_t k = 0; k < Config::OUTNODE; ++k) {
            // 初始化隐藏层第 j 个神经元到输出层第 k 个神经元的权重
            hideLayer[j]->weight[k] = distribution(rd);
            // 初始化隐藏层第 j 个神经元到输出层第 k 个神经元的权重修正值
            hideLayer[j]->weight_delta[k] = 0.f;
        }
    }

    // 初始化输出层
    for (size_t k = 0; k < Config::OUTNODE; ++k) {
        outputLayer[k] = new Node(0);
        // 初始化输出层神经元系节点偏置值
        outputLayer[k]->bias = distribution(rd);
        // 初始化输出层神经元系节点偏置值偏置值
        outputLayer[k]->bias_delta = 0.f;
    }
}

Node::Node(int size) {
    weight.resize(size);
}
