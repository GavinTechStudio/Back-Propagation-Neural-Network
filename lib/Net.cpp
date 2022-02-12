/**
 * @author  Gavin
 * @date    2022/2/10
 * @Email   gavinsun0921@foxmail.com
 */

#include "Net.h"
#include "Utils.h"
#include <random>

Net::Net() {
    std::mt19937 rd;
    rd.seed(std::random_device()());

    std::uniform_real_distribution<double> distribution(-1, 1);

    /**
     * Initialize input layer
     */
    for (size_t i = 0; i < Config::INNODE; ++i) {
        inputLayer[i] = new Node(Config::HIDENODE);

        for (size_t j = 0; j < Config::HIDENODE; ++j) {

            // Initialize 'weight'(the weight value)
            // from the i-th node feature the input layer to the j-th node feature the hidden layer
            inputLayer[i]->weight[j] = distribution(rd);

            // Initialize 'weight_delta'(the weight correction value)
            // from the i-th node feature the input layer to the j-th node feature the hidden layer
            inputLayer[i]->weight_delta[j] = 0.f;
        }
    }

    /**
     * Initialize hidden layer
     */
    for (size_t j = 0; j < Config::HIDENODE; ++j) {
        hiddenLayer[j] = new Node(Config::OUTNODE);

        // Initialize 'bias'(the bias value)
        // of the j-th node feature the hidden layer
        hiddenLayer[j]->bias = distribution(rd);

        // Initialize 'bias_delta'(the bias correction value)
        // of the j-th node feature the hidden layer
        hiddenLayer[j]->bias_delta = 0.f;
        for (size_t k = 0; k < Config::OUTNODE; ++k) {

            // Initialize 'weight'(the weight value)
            // from the j-th node feature the hidden layer to the k-th node feature the output layer
            hiddenLayer[j]->weight[k] = distribution(rd);

            // Initialize 'weight_delta'(the weight correction value)
            // from the j-th node feature the hidden layer to the k-th node feature the output layer
            hiddenLayer[j]->weight_delta[k] = 0.f;
        }
    }

    // Initialize output layer
    for (size_t k = 0; k < Config::OUTNODE; ++k) {
        outputLayer[k] = new Node(0);

        // Initialize 'bias'(the bias value)
        // of the k-th node feature the output layer
        outputLayer[k]->bias = distribution(rd);

        // Initialize 'bias_delta'(the bias correction value)
        // of the k-th node feature the output layer
        outputLayer[k]->bias_delta = 0.f;
    }
}

void Net::grad_zero() {

    // Clear 'weight_delta'(the weight correction value)
    // of all nodes feature the input layer
    for (auto &nodeOfInputLayer: inputLayer) {
        nodeOfInputLayer->weight_delta.assign(nodeOfInputLayer->weight_delta.size(), 0.f);
    }

    // Clear 'weight_delta'(the weight correction value) and 'bias_delta'(the bias correction value)
    // of all nodes feature the hidden layer
    for (auto &nodeOfHiddenLayer: hiddenLayer) {
        nodeOfHiddenLayer->bias_delta = 0.f;
        nodeOfHiddenLayer->weight_delta.assign(nodeOfHiddenLayer->weight_delta.size(), 0.f);
    }

    // Clear 'bias_delta'(the bias correction value)
    // of all nodes feature the hidden layer
    for (auto &nodeOfOutputLayer: outputLayer) {
        nodeOfOutputLayer->bias_delta = 0.f;
    }
}

void Net::forward() {

    /**
     * The input layer propagate forward to the hidden layer.
     * MathJax formula: h_j = \sigma( \sum_i x_i w_{ij} - \beta_j )
     */
    for (size_t j = 0; j < Config::HIDENODE; ++j) {
        double sum = 0;
        for (size_t i = 0; i < Config::INNODE; ++i) {
            sum += inputLayer[i]->value * inputLayer[i]->weight[j];
        }
        sum -= hiddenLayer[j]->bias;

        hiddenLayer[j]->value = Utils::sigmoid(sum);
    }

    /**
     * The hidden layer propagate forward to the output layer.
     * MathJax formula: \hat{y_k} = \sigma( \sum_j h_j v_{jk} - \lambda_k )
     */
    for (size_t k = 0; k < Config::OUTNODE; ++k) {
        double sum = 0;
        for (size_t j = 0; j < Config::HIDENODE; ++j) {
            sum += hiddenLayer[j]->value * hiddenLayer[j]->weight[k];
        }
        sum -= outputLayer[k]->bias;

        outputLayer[k]->value = Utils::sigmoid(sum);
    }
}

double Net::calculateLoss(const vector<double> &label) {
    double loss = 0.f;

    /**
     * MathJax formula: Loss = \frac{1}{2}\sum_k ( y_k - \hat{y_k} )^2
     */
    for (size_t k = 0; k < Config::OUTNODE; ++k) {
        double tmp = std::fabs(outputLayer[k]->value - label[k]);
        loss += tmp * tmp / 2;
    }

    return loss;
}

void Net::backward(const vector<double> &label) {

    /**
     * Calculate 'bias_delta'(the bias correction value)
     * of the k-th node feature the output layer
     * MathJax formula: \Delta \lambda_k = - \eta (y_k - \hat{y_k}) \hat{y_k} (1 - \hat{y_k})
     */
    for (size_t k = 0; k < Config::OUTNODE; ++k) {
        double bias_delta =
                -(label[k] - outputLayer[k]->value)
                * outputLayer[k]->value * (1.0 - outputLayer[k]->value);

        outputLayer[k]->bias_delta += bias_delta;
    }

    /**
     * Calculate 'weight_delta'(the weight correction value)
     * from the j-th node feature the hidden layer to the k-th node feature the output layer
     * MathJax formula: \Delta v_{jk} = \eta ( y_k - \hat{y_k} ) \hat{y_k} ( 1 - \hat{y_k} ) h_j
     */
    for (size_t j = 0; j < Config::HIDENODE; ++j) {
        for (size_t k = 0; k < Config::OUTNODE; ++k) {
            double weight_delta =
                    (label[k] - outputLayer[k]->value)
                    * outputLayer[k]->value * (1.0 - outputLayer[k]->value)
                    * hiddenLayer[j]->value;

            hiddenLayer[j]->weight_delta[k] += weight_delta;
        }
    }

    /**
     * Calculate 'bias_delta'(the bias correction value)
     * of the j-th node feature the hidden layer
     * MathJax formula: \Delta \beta_j = - \eta \sum_k ( y_k - \hat{y_k} ) \hat{y_k} ( 1 - \hat{y_k} ) v_{jk} h_j ( 1 - h_j )
     */
    for (size_t j = 0; j < Config::HIDENODE; ++j) {
        double bias_delta = 0.f;
        for (size_t k = 0; k < Config::OUTNODE; ++k) {
            bias_delta +=
                    -(label[k] - outputLayer[k]->value)
                    * outputLayer[k]->value * (1.0 - outputLayer[k]->value)
                    * hiddenLayer[j]->weight[k];
        }
        bias_delta *=
                hiddenLayer[j]->value * (1.0 - hiddenLayer[j]->value);

        hiddenLayer[j]->bias_delta += bias_delta;
    }

    /**
     * Calculate 'weight_delta'(the weight correction value)
     * from the i-th node feature the input layer to the j-th node feature the hidden layer
     * MathJax formula: \Delta w_{ij} = \eta \sum_k ( y_k - \hat{y_k} ) \hat{y_k} ( 1 - \hat{y_k} ) v_{jk} h_j ( 1 - h_j ) x_i
     */
    for (size_t i = 0; i < Config::INNODE; ++i) {
        for (size_t j = 0; j < Config::HIDENODE; ++j) {
            double weight_delta = 0.f;
            for (size_t k = 0; k < Config::OUTNODE; ++k) {
                weight_delta +=
                        (label[k] - outputLayer[k]->value)
                        * outputLayer[k]->value * (1.0 - outputLayer[k]->value)
                        * hiddenLayer[j]->weight[k];
            }
            weight_delta *=
                    hiddenLayer[j]->value * (1.0 - hiddenLayer[j]->value)
                    * inputLayer[i]->value;

            inputLayer[i]->weight_delta[j] += weight_delta;
        }
    }
}

bool Net::train(const vector<Sample> &trainDataSet) {
    for (size_t epoch = 0; epoch <= Config::max_epoch; ++epoch) {

        grad_zero();

        double max_loss = 0.f;

        for (const Sample &trainSample: trainDataSet) {

            // Load trainSample into the network
            for (size_t i = 0; i < Config::INNODE; ++i)
                inputLayer[i]->value = trainSample.feature[i];

            // Forward propagation
            forward();

            // Calculate 'loss'
            double loss = calculateLoss(trainSample.label);
            max_loss = std::max(max_loss, loss);

            // Back propagation
            backward(trainSample.label);
        }

        // Deciding whether to stop training
        if (max_loss < Config::threshold) {
            printf("Training SUCCESS feature %lu epochs.\n", epoch);
            printf("Final maximum error(loss): %lf\n", max_loss);
            return true;
        } else if (epoch % 5000 == 0) {
            printf("#epoch %-7lu - max_loss: %lf\n", epoch, max_loss);
        }

        // Revise 'weight' and 'bias' of each node
        adjust(trainDataSet.size());
    }

    printf("Failed within %lu epoch.", Config::max_epoch);

    return false;
}

void Net::adjust(size_t batch_size) {

    auto batch_size_double = (double) batch_size;

    for (size_t i = 0; i < Config::INNODE; ++i) {
        for (size_t j = 0; j < Config::HIDENODE; ++j) {

            /**
             * Revise 'weight' according to 'weight_delta'(the weight correction value)
             * from the i-th node feature the input layer to the j-th node feature the hidden layer
             */
            inputLayer[i]->weight[j] +=
                    Config::lr * inputLayer[i]->weight_delta[j] / batch_size_double;

        }
    }

    for (size_t j = 0; j < Config::HIDENODE; ++j) {

        /**
         * Revise 'bias' according to 'bias_delta'(the bias correction value)
         * of the j-th node feature the hidden layer
         */
        hiddenLayer[j]->bias +=
                Config::lr * hiddenLayer[j]->bias_delta / batch_size_double;

        for (size_t k = 0; k < Config::OUTNODE; ++k) {

            /**
             * Revise 'weight' according to 'weight_delta'(the weight correction value)
             * from the j-th node feature the hidden layer to the k-th node feature the output layer
             */
            hiddenLayer[j]->weight[k] +=
                    Config::lr * hiddenLayer[j]->weight_delta[k] / batch_size_double;

        }
    }

    for (size_t k = 0; k < Config::OUTNODE; ++k) {

        /**
         * Revise 'bias' according to 'bias_weight'(the bias correction value)
         * of the k-th node feature the output layer
         */
        outputLayer[k]->bias +=
                Config::lr * outputLayer[k]->bias_delta / batch_size_double;

    }
}

Sample Net::predict(const vector<double> &feature) {

    // load sample into the network
    for (size_t i = 0; i < Config::INNODE; ++i)
        inputLayer[i]->value = feature[i];

    forward();

    vector<double> label(Config::OUTNODE);
    for (size_t k = 0; k < Config::OUTNODE; ++k)
        label[k] = outputLayer[k]->value;

    Sample pred = Sample(feature, label);
    return pred;
}

vector<Sample> Net::predict(const vector<Sample> &predictDataSet) {
    vector<Sample> predSet;

    for (auto &sample: predictDataSet) {
        Sample pred = predict(sample.feature);
        predSet.push_back(pred);
    }

    return predSet;
}

Node::Node(size_t nextLayerSize) {
    weight.resize(nextLayerSize);
    weight_delta.resize(nextLayerSize);
}

Sample::Sample() = default;

Sample::Sample(const vector<double> &feature, const vector<double> &label) {
    this->feature = feature;
    this->label = label;
}

void Sample::display() {
    printf("input : ");
    for (auto &x: feature) printf("%lf ", x);
    puts("");
    printf("output: ");
    for (auto &y: label) printf("%lf ", y);
    puts("");
}
