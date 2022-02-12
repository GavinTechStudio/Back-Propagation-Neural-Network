//
// Created by gavin on 2022/2/10.
//

#ifndef BPNN_NET_H
#define BPNN_NET_H

#include <vector>
#include "Config.h"

using std::vector;

struct Sample {
    vector<double> feature, label;

    Sample();

    Sample(const vector<double> &feature, const vector<double> &label);

    void display();
};

struct Node {
    double value{}, bias{}, bias_delta{};
    vector<double> weight, weight_delta;

    explicit Node(size_t nextLayerSize);
};

class Net {
private:
    Node *inputLayer[Config::INNODE];
    Node *hiddenLayer[Config::HIDENODE];
    Node *outputLayer[Config::OUTNODE];

    /**
     * Clear all gradient accumulation
     *
     * Set 'weight_delta'(the weight correction value) and
     * 'bias_delta'(the bias correction value) to 0 of nodes
     */
    void grad_zero();

    /**
     * Forward propagation
     */
    void forward();

    /**
     * Calculate the value of Loss Function
     * @param label the label of sample (vector / numeric)
     * @return loss
     */
    double calculateLoss(const vector<double> &label);

    /**
     * Back propagation
     * @param label label of sample (vector / numeric)
     */
    void backward(const vector<double> &label);

    /**
     * Revise 'weight' and 'bias according to
     * 'weight_delta'(the weight correction value) and
     * 'bias_weight'(the bias correction value)
     * @param batch_size
     */
    void adjust(size_t batch_size);

public:

    Net();

    /**
     * Training network with training data
     * @param trainDataSet The sample set
     * @return Training convergence
     */
    bool train(const vector<Sample> &trainDataSet);

    /**
     * Using network to predict sample
     * @param feature The feature of sample (vector)
     * @return Sample with 'feature' and 'label'(predicted)
     */
    Sample predict(const vector<double> &feature);

    /**
     * Using network to predict the sample set
     * @param predictDataSet The sample set
     * @return The sample set, in which each sample has 'feature' and 'label'(predicted)
     */
    vector<Sample> predict(const vector<Sample> &predictDataSet);

};


#endif //BPNN_NET_H
