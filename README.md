## TensorFlow implementation of "Iterative Pruning" and revised from revised from: [impl-pruning-TF](https://github.com/garion9013/impl-pruning-TF).

This work is based on "Learning both Weights and Connections for Efficient
Neural Network." [Song et al.](http://arxiv.org/pdf/1506.02626v3.pdf) @ NIPS '15.

Besides, I also designed a matrix-based pruning, which has less storage overhead and higher computation efficiency. 
For more details, please refer to the attached Final Project Report.

I applied Iterative Pruning on a small MNIST CNN model (13MB, originally), which can be
accessed from [TensorFlow Tutorials](https://www.tensorflow.org/versions/r0.8/tutorials/mnist/pros/index.html).

## File descriptions and usages

model_ckpt_dense: original model<br>
model_ckpt_dense_pruned: pruned-only model<br>
model_ckpt_sparse_retrained: pruned and retrained model<br>

#### Python package requirements
```bash
sudo apt-get install python-scipy python-numpy python-matplotlib
```

To regenerate different sparse model, edit ```config.py``` first. You can decide which layer you want to prune,
and then run training with different pruning mode as depicted below.

Train convolutional neural network model,

```bash
python3 train.py -1 0
```

Apply pruning and additional retraining,

```bash
./train.py -2 [pruning ratio]
```

Apply iterative pruning and additional retraining,

```bash
./train.py -3 [pruning ratio]
```

Apply iterative pruning for each layer and retraining,

```bash
./train.py -4 [pruning ratio]
```

Apply proposed matrix-based pruning iteratively and retraining,

```bash
./train.py -5 [pruning ratio]
```

To inference single image (seven.png) and measure its computation time,

```bash
./deploy_test.py -d
./deploy_test_sparse.py -d
./deploy_test_struct.py -d
```

To draw histogram that shows the weight distribution,

```bash
# After running train.py (it generates .dat files)
./draw_histogram.py
```

