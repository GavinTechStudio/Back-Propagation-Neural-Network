#include <iostream>
#include "lib/Net.h"
#include "lib/Utils.h"

using std::cout;
using std::endl;

int main(int argc, char *argv[]) {

    // Create neural network object
    Net net;

    // Read training data
    const vector<Sample> trainDataSet = Utils::getTrainData("../data/traindata.txt");

    // Training neural network
    net.train(trainDataSet);

    // Prediction of samples using neural network
    const vector<Sample> testDataSet = Utils::getTestData("../data/testdata.txt");
    vector<Sample> predSet = net.predict(testDataSet);
    for (auto &pred: predSet) {
        pred.display();
    }

    return 0;
}
