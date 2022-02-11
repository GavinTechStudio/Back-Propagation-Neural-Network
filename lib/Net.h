//
// Created by gavin on 2022/2/10.
//

#ifndef BPNN_NET_H
#define BPNN_NET_H

#include <vector>
#include "Config.h"

using std::vector;

struct Sample {
    vector<double> in, out;

    Sample() = default;

    Sample(const vector<double> &in, const vector<double> &out) {
        this->in = in;
        this->out = out;
    }
};

struct Node {
    double value{}, bias{}, bias_delta{};
    vector<double> weight, weight_delta;

    explicit Node(size_t nextLayerSize);
};

class Net {
public:
    Node *inputLayer[Config::INNODE];
    Node *hideLayer[Config::HIDENODE];
    Node *outputLayer[Config::OUTNODE];

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
     * 计算损失函数 Loss 值
     * @param out 对应本组样本真实的输出向量/值
     * @return Loss 值
     */
    double calculateLoss(const vector<double> &out);

    /**
     * 反向传播
     * @param out 对应本组样本真实的输出向量/值
     */
    void backward(const vector<double> &out);

    /**
     * 训练网络
     * @param trainDataSet 训练数据集
     * @return 是否训练成功（收敛）
     */
    bool train(const vector<Sample> &trainDataSet);

    /**
     * 利用反向传播计算的修正值调整各参数值
     * 对于一个 batch 的修正值作用效果，采取平均值
     * @param batch_size batch 大小
     */
    void adjust(size_t batch_size);

    /**
     * 利用训练好的网络进行预测
     * @param in 预测样本的输入向量
     * @return 通过网路预测后完整的预测样本（包含输入向量和输出向量）
     */
    Sample predict(const vector<double> &in);

    /**
     * 利用训练好的网络进行预测
     * @param predictDataSet 预测样本的vector容器
     * @return 所有预测完成的样本的vector容器
     */
    vector<Sample> predict(const vector<Sample> &predictDataSet);

};


#endif //BPNN_NET_H
