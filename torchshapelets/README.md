# torchshapelets

A differentiable implementation of the generalised shapelet transform, using PyTorch, parallelised via OpenMP.

# What is the shapelet transform?
It's a feature extraction method for time series, in which a time series is described by its similarity to a small 'shapelet'. Given lots of well-chosen shapelets, then you can now look at those similarities and conclude that "This time series is probably of class X, because it has a very high similarity to shapelet Y." (In practice of course you feed these features into a machine learning model, but the point is that these features are interpretable.)

For more details, see the paper: TODO.

Despite the name, it has nothing to do with wavelets.

# Installation

`pip install "git+https://github.com/jambo6/generalised_shapelets/#egg=torchshapelets&subdirectory=torchshapelets"`

Make sure you include the quotation marks. Tested to work on Linux. If on other operating systems then you must have a C++ compiler available and known to `pip`. (If you're on a Linux system then this should already be the case.)

If you want to compute logsignature discrepancies then install [Signatory](https://github.com/patrick-kidger/signatory) first. If that's not installed then `torchshapelets` will still work, but `torchshapelets.LogsignatureDiscrepancy` will not be available.

# Usage

Once installed, then `import torchshapelets` to get everything.

The key class is `torchshapelets.GeneralisedShapeletTransform`, which computes the generalised shapelet transform.

### Example:
```python
import torch
import torchshapelets

in_channels = 3
num_shapelets = 4
num_shapelet_samples = 5
discrepancy_fn = torchshapelets.L2Discrepancy(in_channels)
max_shapelet_length = 5.0

transform = torchshapelets.GeneralisedShapeletTransform(in_channels,
                                                        num_shapelets,
                                                        num_shapelet_samples,
                                                        discrepancy_fn,
                                                        max_shapelet_length)

batch_size = 8
time_sampling = 10
# In particular the time difference 9 - 0 is greater than the
# max_shapelet_length. (Note that this is unrelated to how often
# times are sampled.)
times = torch.linspace(0, 9, time_sampling)
path = torch.rand(batch_size, time_sampling, in_channels)

shapelet_similarity = transform(times, path)
# This will now be a tensor of shape (batch_size, num_shapelets),
# describing the similarity between each batch element and each shapelet.
```

### CPU vs GPU
This is specifically written to operate on the CPU, and will probably crash if you try to run it on the GPU.

That this is the case is just an implementation limitation - computing shapelets is a massively parallel and therefore GPU-friendly operation, with the potential to parallelise over batch, over different shapelets, and over the continuous-minimum operation. But there's no easy way to write this parallelisation using just PyTorch (for the general case of irregularly sampled data), so this would need a custom GPU kernel.

If you do want to build this into a deep learning model which is mostly on the GPU, then that is doable by passing the tensors between the CPU and GPU in the usual way for PyTorch.

### Full API
Available objects are:
```python
torchshapelets.GeneralisedShapeletTransform

torchshapelets.CppDiscrepancy
torchshapelets.L2Discrepanc
torchshapelets.LogsignatureDiscrepancy

torchshapelets.similarity_regularisation
```
In brief: a discrepancy function such as `L2Discrepancy` or `LogsignatureDiscrepancy` describes how to compute the similarity between a shapelet, and a small piece of a path of equal length.

This is then an argument to the `GeneralisedShapeletTransform`. This is then capable of computing the the discrepancy between a full path and a shapelet, for each shapelet, in a parallelisable manner.

The regularisation will help ensure that the shapelets that are learnt are in fact interpretable (rather than just being random-looking).

Check their docstrings and the paper TODO for more details.

# Citation

TODO