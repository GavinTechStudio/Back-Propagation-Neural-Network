#include <iostream>
#include "lib/Net.h"
#include "lib/Utils.h"

using std::cout;
using std::endl;

int main(int argc, char *argv[]) {

    // 创建网络
    Net net;

    // 读取训练数据
    const vector<Sample> trainDataSet = Utils::getTrainData("../data/traindata.txt");

    // 训练网络
    net.train(trainDataSet);

    // 利用网络预测
    const vector<Sample> testDataSet = Utils::getTestData("../data/testdata.txt");
    vector<Sample> predSet = net.predict(testDataSet);
    for (auto &pred: predSet) {
        pred.display();
    }

    return 0;
}
