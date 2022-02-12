/**
 * @author  Gavin
 * @date    2022/2/10
 * @Email   gavinsun0921@foxmail.com
 */

#ifndef BPNN_UTILS_H
#define BPNN_UTILS_H


#include <cmath>
#include <vector>
#include <string>
#include "Net.h"

using std::vector;
using std::string;

namespace Utils {
    static double sigmoid(double x) {
        return 1.0 / (1.0 + std::exp(-x));
    }

    vector<double> getFileData(const string &filename);

    vector<Sample> getTrainData(const string &filename);

    vector<Sample> getTestData(const string &filename);
};


#endif //BPNN_UTILS_H
