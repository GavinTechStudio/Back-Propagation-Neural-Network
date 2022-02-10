#include <iostream>
#include "Net.h"
#include "Utils.h"

using std::cout;
using std::endl;

int main() {

    // 创建网络
    Net net;

    // 读取训练数据
    const vector<Sample> trainDataSet = Utils::getTrainData("traindata.txt");
    cout << trainDataSet.size() << endl;

    net.train(trainDataSet);

    return 0;
}
