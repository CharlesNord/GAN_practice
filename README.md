# GAN_practice

## gan_demo
gan_demo 是最基础的 gan。它的网络结构采用了如下模块：
* LeakyReLU
* BatchNormalization
* 每个 mini-batch 中 real 和 fake 是单一的，而不能混合在一起
* 最后一层 generator 采用 tanh 激活函数

LeakyReLU 在 GAN 中被广泛运用，它的作用是防止普通 ReLU 当 W 出现的大梯度更新后，对所有的训练数据输出都是小于零的数，最终造成该神经元只输出0，也就是 dead neuron。
BatchNormalization 是为了加速收敛，第一层网络和输出层不加 Batch normalization，之后的顺序是 Conv->BN->LeakyReLU
tanh 用作 Generator 最后一层输出的 activation，而不是 sigmoid 函数，原因是， 1. tanh 比 sigmoid 可以提供更大的梯度值。 2.tanh(x) 在 [-1,1]之间是对称的，防止了梯度的 bias。 参考 LeCun 的[Efficient BackProp](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)

其余的训练技巧，参考[How to Train a GAN? Tips and tricks to make GANs work](https://github.com/soumith/ganhacks#authors)
