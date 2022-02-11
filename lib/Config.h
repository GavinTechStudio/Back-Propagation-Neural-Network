//
// Created by gavin on 2022/2/10.
//

#ifndef BPNN_CONFIG_H
#define BPNN_CONFIG_H

using std::size_t;

namespace Config {
    const size_t INNODE = 2;
    const size_t HIDENODE = 4;
    const size_t OUTNODE = 1;

    const double lr = 0.8;
    const double threshold = 1e-4;
    const size_t max_epoch = 1e7;
};


#endif //BPNN_CONFIG_H
