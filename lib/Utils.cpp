//
// Created by gavin on 2022/2/10.
//

#include <iostream>
#include <fstream>
#include <unistd.h>
#include "Utils.h"
#include "Config.h"

vector<double> Utils::getFileData(const string &filename) {
    vector<double> res;

    std::ifstream in(filename);
    if (in.is_open()) {
        while (!in.eof()) {
            double val;
            in >> val;
            res.push_back(val);
        }
        in.close();
    } else {
        // 未成功读取到数据文件
        printf("[ERROR] '%s' not found.\n", filename.c_str());
//        cout << "[Error] " << filename << "' not found." << endl;
        // 输出当前的可执行文件路径
        char path[256];
        getcwd(path, sizeof(path));
        printf("Please check the path '%s' is relative to '%s'.\n", filename.c_str(), path);
//        cout << "Please check the path '" << filename << "' is relative to '" << path << "'." << endl;
        exit(1);
    }

    return res;
}

vector<Sample> Utils::getTrainData(const string &filename) {
    vector<Sample> trainDataSet;

    vector<double> buffer = getFileData(filename);

    for (size_t i = 0; i < buffer.size(); i += Config::INNODE + Config::OUTNODE) {
        Sample trainSample;
        // 读入训练样本输入
        for (size_t j = 0; j < Config::INNODE; ++j)
            trainSample.in.push_back(buffer[i + j]);
        // 读入训练样本输出
        for (size_t k = 0; k < Config::OUTNODE; ++k)
            trainSample.out.push_back(buffer[i + Config::INNODE + k]);
        // 将样本加入到训练集
        trainDataSet.push_back(trainSample);
    }
    return trainDataSet;
}

vector<Sample> Utils::getTestData(const string &filename) {
    vector<Sample> testDataSet;

    vector<double> buffer = getFileData(filename);

    for (size_t i = 0; i < buffer.size(); i += Config::INNODE) {
        Sample testSample;
        // 读入测试样本输入
        for (size_t j = 0; j < Config::INNODE; ++j)
            testSample.in.push_back(buffer[i + j]);
        // 将样本加入到测试集
        testDataSet.push_back(testSample);
    }

    return testDataSet;
}
