# Batch Normalization

This branch provides implementation of Batch Normalization (BN). Most of the codes are adpated from Chenglong Chen's [caffe-windows](https://github.com/ChenglongChen/caffe-windows).

## Usage

Just add a BN layer before each activation function. The configuration of a BN layer looks like:

    layer {
      name: "conv1_bn"
      type: "BN"
      bottom: "conv1"
      top: "conv1_bn"
      param {
        lr_mult: 1
        decay_mult: 0
      }
      param {
        lr_mult: 1
        decay_mult: 0
      }
      bn_param {
        slope_filler {
          type: "constant"
          value: 1
        }
        bias_filler {
          type: "constant"
          value: 0
        }
      }
    }

We also implement a simple version of local data shuffling in the data layer. It's recommended to set `shuffle_pool_size: 10` in `data_param` of the training data layer.
