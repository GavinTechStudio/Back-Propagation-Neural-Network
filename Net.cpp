//
// Created by gavin on 2022/2/10.
//

#include "Net.h"
#include "Utils.h"
#include <random>
#include <iostream>

using std::cout;
using std::endl;

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

void Net::grad_zero() {

    // 清零输入层所有节点的 weight_delta
    for (auto &node_input: inputLayer) {
        node_input->weight_delta.assign(node_input->weight_delta.size(), 0.f);
    }

    // 清零隐藏层所有节点的 bias_delta 和 weight_delta
    for (auto &node_hide: hideLayer) {
        node_hide->bias_delta = 0.f;
        node_hide->weight_delta.assign(node_hide->weight_delta.size(), 0.f);
    }

    // 清零输出层所有节点的 bias_delta
    for (auto &node_output: outputLayer) {
        node_output->bias_delta = 0.f;
    }
}

void Net::forward() {

    // 输入层向隐藏层传播
    // h_j = \sigma( \sum_i x_i w_{ij} - \beta_j )
    for (size_t j = 0; j < Config::HIDENODE; ++j) {
        // 计算第 j 个隐藏层节点的值
        double sum = 0;
        for (size_t i = 0; i < Config::INNODE; ++i) {
            // 第 i 个输入层节点对第 j 个隐藏层节点的贡献
            sum += inputLayer[i]->value * inputLayer[i]->weight[j];
        }
        sum -= hideLayer[j]->bias;

        // 激活函数选用 sigmoid
        hideLayer[j]->value = Utils::sigmoid(sum);
    }

    // 隐藏层向输出层传播
    // \hat{y_k} = \sigma( \sum_j h_j v_{jk} - \lambda_k )
    for (size_t k = 0; k < Config::OUTNODE; ++k) {
        // 计算第 k 个输出层节点的值
        double sum = 0;
        for (size_t j = 0; j < Config::HIDENODE; ++j) {
            // 第 j 个隐藏层节点对第 k 个输出层节点的恭喜
            sum += hideLayer[j]->value * hideLayer[j]->weight[k];
        }
        sum -= outputLayer[k]->bias;

        // 激活函数选用 sigmoid
        outputLayer[k]->value = Utils::sigmoid(sum);
    }
}

double Net::CalculateLoss(const vector<double> &out) {
    double loss = 0.f;

    // Loss = \frac{1}{2}\sum_k ( y_k - \hat{y_k} )^2
    for (size_t k = 0; k < Config::OUTNODE; ++k) {
        double tmp = std::fabs(outputLayer[k]->value - out[k]);
        loss += tmp * tmp / 2;
    }

    return loss;
}

void Net::backward(const vector<double> &out) {

    // 计算输出层节点的偏置值修正值
    // \Delta \lambda_k = - \eta (y_k - \hat{y_k}) \hat{y_k} (1 - \hat{y_k})
    for (size_t k = 0; k < Config::OUTNODE; ++k) {
        double bias_delta =
                -(out[k] - outputLayer[k]->value)
                * outputLayer[k]->value * (1.0 - outputLayer[k]->value);

        outputLayer[k]->bias_delta += bias_delta;
    }

    // 计算隐藏层节点到输出层节点权重修正值
    // \Delta v_{jk} = \eta \sum_k ( y_k - \hat{y_k} ) \hat{y_k} ( 1 - \hat{y_k} ) h_j
    for (size_t j = 0; j < Config::HIDENODE; ++j) {
        for (size_t k = 0; k < Config::OUTNODE; ++k) {
            double weight_delta =
                    (out[k] - outputLayer[k]->value)
                    * outputLayer[k]->value * (1.0 - outputLayer[k]->value)
                    * hideLayer[j]->value;

            hideLayer[j]->weight_delta[k] += weight_delta;
        }
    }

    // 计算隐藏层节点的偏置值修正值
    // \Delta \beta_j = - \eta \sum_k ( y_k - \hat{y_k} ) \hat{y_k} ( 1 - \hat{y_k} ) v_{jk} h_j ( 1 - h_j )
    for (size_t j = 0; j < Config::HIDENODE; ++j) {
        double bias_delta = 0.f;
        for (size_t k = 0; k < Config::OUTNODE; ++k) {
            bias_delta +=
                    -(out[k] - outputLayer[k]->value)
                    * outputLayer[k]->value * (1.0 - outputLayer[k]->value)
                    * hideLayer[j]->weight[k];
        }
        bias_delta *=
                hideLayer[j]->value * (1 - hideLayer[j]->value);

        hideLayer[j]->bias_delta += bias_delta;
    }

    // 计算输入层节点到隐藏层节点权重修正值
    // \Delta w_{ij} = \eta \sum_k ( y_k - \hat{y_k} ) \hat{y_k} ( 1 - \hat{y_k} ) v_{jk} h_j ( 1 - h_j ) x_i
    for (size_t i = 0; i < Config::INNODE; ++i) {
        for (size_t j = 0; j < Config::HIDENODE; ++j) {
            double weight_delta = 0.f;
            for (size_t k = 0; k < Config::OUTNODE; ++k) {
                weight_delta +=
                        (out[k] - outputLayer[k]->value)
                        * outputLayer[k]->value * (1.0 - outputLayer[k]->value)
                        * hideLayer[j]->weight[k];
            }
            weight_delta *=
                    hideLayer[j]->value * (1.0 - hideLayer[j]->value)
                    * inputLayer[i]->value;

            inputLayer[i]->weight_delta[j] += weight_delta;
        }
    }
}

bool Net::train(const vector<Sample> &trainDataSet) {
    for (size_t epoch = 0; epoch <= Config::max_epoch; ++epoch) {
        // 清零上一个 epoch 的梯度
        grad_zero();

        double max_loss = 0.f;

        for (const Sample &sample: trainDataSet) {

            // 将本组样本加载到网络中
            for (size_t i = 0; i < Config::INNODE; ++i)
                inputLayer[i]->value = sample.in[i];

            // 前向传播
            forward();

            // 记录 Loss
            double loss = CalculateLoss(sample.out);
            max_loss = std::max(max_loss, loss);

            // 反向传播
            backward(sample.out);
        }

        // 判断是否停止训练
        if (max_loss < Config::threshold) {
            cout << "Success in " << epoch << " epoch." << endl;
            cout << "Final maximum error(loss): " << max_loss << endl;
            return true;
        } else if (epoch % 5000 == 0) {
            cout << "#epoch " << epoch << " max_loss: " << max_loss << endl;
        }

        // 各参数修正值作用
        adjust(trainDataSet.size());
    }

    cout << "Failed within " << Config::max_epoch << " epoch." << endl;

    return false;
}

void Net::adjust(size_t batch_size) {

    auto batch_size_double = (double) batch_size;

    for (size_t i = 0; i < Config::INNODE; ++i) {
        for (size_t j = 0; j < Config::HIDENODE; ++j) {
            // 调整输入层节点到隐藏层节点权重
            inputLayer[i]->weight[j] +=
                    Config::lr * inputLayer[i]->weight_delta[j] / batch_size_double;
        }
    }

    for (size_t j = 0; j < Config::HIDENODE; ++j) {
        // 调整隐藏层节点偏置值
        hideLayer[j]->bias +=
                Config::lr * hideLayer[j]->bias_delta / batch_size_double;
        for (size_t k = 0; k < Config::OUTNODE; ++k) {
            // 调整隐藏层节点到输出层节点权重
            hideLayer[j]->weight[k] +=
                    Config::lr * hideLayer[j]->weight_delta[k] / batch_size_double;
        }
    }

    for (size_t k = 0; k < Config::OUTNODE; ++k) {
        // 调整输出层节点偏置值
        outputLayer[k]->bias +=
                Config::lr * outputLayer[k]->bias_delta / batch_size_double;
    }
}

vector<double> Net::predict(const vector<double> &in) {
    vector<double> pred(Config::OUTNODE);

    for (size_t i = 0; i < Config::INNODE; ++i)
        inputLayer[i]->value = in[i];

    forward();

    for (size_t k = 0; k < Config::OUTNODE; ++k)
        pred[k] = outputLayer[k]->value;

    return pred;
}

Node::Node(size_t nextLayerSize) {
    weight.resize(nextLayerSize);
    weight_delta.resize(nextLayerSize);
}
