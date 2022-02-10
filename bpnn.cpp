#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <random>

#define INNODE 2
#define HIDENODE 4
#define OUTNODE 1

double rate = 0.8;
double threshold = 1e-4;
size_t mosttimes = 1e6;

struct Sample {
    std::vector<double> in, out;
};

struct Node {
    double value{}, bias{}, bias_delta{};
    std::vector<double> weight, weight_delta;

    Node(size_t nextLayerSize) {
        weight.resize(nextLayerSize);
        weight_delta.resize(nextLayerSize);
    }
};

namespace utils {

    inline double sigmoid(double x) {
        double res = 1.0 / (1.0 + std::exp(-x));
        return res;
    }

    std::vector<double> getFileData(std::string filename) {
        std::vector<double> res;

        std::ifstream in(filename);
        if (in.is_open()) {
            while (!in.eof()) {
                double buffer;
                in >> buffer;
                res.push_back(buffer);
            }
            in.close();
        } else {
            std::cout << "Error in reading " << filename << std::endl;
        }

        return res;
    }

    std::vector<Sample> getTrainData(std::string filename) {
        std::vector<Sample> res;

        std::vector<double> buffer = getFileData(filename);

        for (size_t i = 0; i < buffer.size(); i += INNODE + OUTNODE) {
            Sample tmp;
            for (size_t t = 0; t < INNODE; t++) {
                tmp.in.push_back(buffer[i + t]);
            }
            for (size_t t = 0; t < OUTNODE; t++) {
                tmp.out.push_back(buffer[i + INNODE + t]);
            }
            res.push_back(tmp);
        }

        return res;
    }

    std::vector<Sample> getTestData(std::string filename) {
        std::vector<Sample> res;

        std::vector<double> buffer = getFileData(filename);

        for (size_t i = 0; i < buffer.size(); i += INNODE) {
            Sample tmp;
            for (size_t t = 0; t < INNODE; t++) {
                tmp.in.push_back(buffer[i + t]);
            }
            res.push_back(tmp);
        }

        return res;
    }

}

Node *inputLayer[INNODE], *hideLayer[HIDENODE], *outLayer[OUTNODE];

inline void init() {
    std::mt19937 rd;
    rd.seed(std::random_device()());

    std::uniform_real_distribution<double> distribution(-1, 1);

    for (size_t i = 0; i < INNODE; i++) {
        ::inputLayer[i] = new Node(HIDENODE);
        for (size_t j = 0; j < HIDENODE; j++) {
            ::inputLayer[i]->weight[j] = distribution(rd);
            ::inputLayer[i]->weight_delta[j] = 0.f;
        }
    }

    for (size_t i = 0; i < HIDENODE; i++) {
        ::hideLayer[i] = new Node(OUTNODE);
        ::hideLayer[i]->bias = distribution(rd);
        for (size_t j = 0; j < OUTNODE; j++) {
            ::hideLayer[i]->weight[j] = distribution(rd);
            ::hideLayer[i]->weight_delta[j] = 0.f;
        }
    }

    for (size_t i = 0; i < OUTNODE; i++) {
        ::outLayer[i] = new Node(0);
        ::outLayer[i]->bias = distribution(rd);
    }

}

inline void reset_delta() {

    for (auto &i: inputLayer) {
        i->weight_delta.assign(i->weight_delta.size(), 0.f);
    }

    for (auto &i: hideLayer) {
        i->bias_delta = 0.f;
        i->weight_delta.assign(i->weight_delta.size(), 0.f);
    }

    for (auto &i: outLayer) {
        i->bias_delta = 0.f;
    }

}

int main(int argc, char *argv[]) {

    init();

    std::vector<Sample> train_data = utils::getTrainData("traindata.txt");

    // training
    for (size_t times = 0; times < mosttimes; times++) {

        reset_delta();

        double error_max = 0.f;

        for (auto &idx : train_data) {

            for (size_t i = 0; i < INNODE; i++) {
                ::inputLayer[i]->value = idx.in[i];
            }

            // 正向传播
            for (size_t j = 0; j < HIDENODE; j++) {
                double sum = 0;
                for (size_t i = 0; i < INNODE; i++) {
                    sum += ::inputLayer[i]->value * ::inputLayer[i]->weight[j];
                }
                sum -= ::hideLayer[j]->bias;

                ::hideLayer[j]->value = utils::sigmoid(sum);
            }

            for (size_t k = 0; k < OUTNODE; k++) {
                double sum = 0;
                for (size_t j = 0; j < HIDENODE; j++) {
                    sum += ::hideLayer[j]->value * ::hideLayer[j]->weight[k];
                }
                sum -= ::outLayer[k]->bias;

                ::outLayer[k]->value = utils::sigmoid(sum);
            }

            // 计算误差
            double error = 0.f;
            for (size_t i = 0; i < OUTNODE; i++) {
                double tmp = std::fabs(::outLayer[i]->value - idx.out[i]);
                error += tmp * tmp / 2;
            }

            error_max = std::max(error_max, error);

            // 反向传播

            for (size_t k = 0; k < OUTNODE; k++) {
                double bias_delta = 
                        -(idx.out[k] - ::outLayer[k]->value) *
                        ::outLayer[k]->value * (1.0 - ::outLayer[k]->value);

                ::outLayer[k]->bias_delta += bias_delta;
            }

            for (size_t j = 0; j < HIDENODE; j++) {
                for (size_t k = 0; k < OUTNODE; k++) {
                    double weight_delta = 
                            (idx.out[k] - ::outLayer[k]->value) *
                            ::outLayer[k]->value * (1.0 - ::outLayer[k]->value) *
                            ::hideLayer[j]->value;
                    
                    ::hideLayer[j]->weight_delta[k] += weight_delta;
                }
            }

            for (size_t j = 0; j < HIDENODE; j++) {
                double sum = 0;
                for (size_t k = 0; k < OUTNODE; k++) {
                    sum +=
                            -(idx.out[k] - ::outLayer[k]->value) *
                            ::outLayer[k]->value * (1.0 - ::outLayer[k]->value) *
                            ::hideLayer[j]->weight[k];
                }
                sum *=
                        ::hideLayer[j]->value * (1.0 - ::hideLayer[j]->value);

                ::hideLayer[j]->bias_delta += sum;
            }

            for (size_t i = 0; i < INNODE; i++) {
                for (size_t j = 0; j < HIDENODE; j++) {
                    double sum = 0.f;
                    for (size_t k = 0; k < OUTNODE; k++) {
                        sum +=
                                (idx.out[k] - ::outLayer[k]->value) *
                                ::outLayer[k]->value * (1.0 - ::outLayer[k]->value) *
                                ::hideLayer[j]->weight[k];
                    }
                    sum *=
                        ::hideLayer[j]->value * (1.0 - ::hideLayer[j]->value) *
                        ::inputLayer[i]->value;

                    ::inputLayer[i]->weight_delta[j] += sum;
                }
            }

        }

        if (error_max < ::threshold) {
            std::cout << "Success with " << times + 1 << " times training." << std::endl;
            std::cout << "Maximum error: " << error_max << std::endl;
            break;
        } else if (times % 10000 == 0) {
            std::cout << "#epoch " << times << " max_loss: " << error_max << std::endl;
        }

        auto train_data_size = double(train_data.size());

        for (size_t i = 0; i < INNODE; i++) {
            for (size_t j = 0; j < HIDENODE; j++) {
                ::inputLayer[i]->weight[j] +=
                        rate * ::inputLayer[i]->weight_delta[j] / train_data_size;
            }
        }

        for (size_t j = 0; j < HIDENODE; j++) {
            ::hideLayer[j]->bias +=
                    rate * ::hideLayer[j]->bias_delta / train_data_size;
            for (size_t k = 0; k < OUTNODE; k++) {
                ::hideLayer[j]->weight[k] +=
                        rate * ::hideLayer[j]->weight_delta[k] / train_data_size;
            }
        }

        for (size_t k = 0; k < OUTNODE; k++) {
            ::outLayer[k]->bias +=
                    rate * ::outLayer[k]->bias_delta / train_data_size;
        }

    }

    std::vector<Sample> test_data = utils::getTestData("testdata.txt");

    // predict
    for (auto &idx : test_data) {

        for (size_t i = 0; i < INNODE; i++) {
            ::inputLayer[i]->value = idx.in[i];
        }

        for (size_t j = 0; j < HIDENODE; j++) {
            double sum = 0;
            for (size_t i = 0; i < INNODE; i++) {
                sum += ::inputLayer[i]->value * inputLayer[i]->weight[j];
            }
            sum -= ::hideLayer[j]->bias;

            ::hideLayer[j]->value = utils::sigmoid(sum);
        }

        for (size_t j = 0; j < OUTNODE; j++) {
            double sum = 0;
            for (size_t i = 0; i < HIDENODE; i++) {
                sum += ::hideLayer[i]->value * ::hideLayer[i]->weight[j];
            }
            sum -= ::outLayer[j]->bias;

            ::outLayer[j]->value = utils::sigmoid(sum);

            idx.out.push_back(::outLayer[j]->value);

            for (auto &tmp : idx.in) {
                std::cout << tmp << " ";
            }
            for (auto &tmp : idx.out) {
                std::cout << tmp << " ";
            }
            std::cout << std::endl;
        }

    }

    return 0;
}
