# torchshapelets

A differentiable implementation of the generalised shapelet transform, using PyTorch, parallelised via OpenMP.

# What is the shapelet transform?
It's a feature extraction method for time series, in which a time series is described by its similarity to a small 'shapelet'. Given lots of well-chosen shapelets, then you can now look at those similarities and conclude that "This time series is probably of class X, because it has a very high similarity to shapelet Y." (In practice of course you feed these features into a machine learning model, but the point is that these features are interpretable.)

For more details, see the paper: TODO.

Despite the similar names, it has nothing to do with wavelets.

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

### Full API
Available objects are:
```python
torchshapelets.GeneralisedShapeletTransform

torchshapelets.CppDiscrepancy
torchshapelets.L2Discrepancy
torchshapelets.LogsignatureDiscrepancy

torchshapelets.similarity_regularisation
```
In brief: a discrepancy function such as `L2Discrepancy` or `LogsignatureDiscrepancy` describes how to compute the similarity between a shapelet, and a small piece of a path of equal length.

This is then an argument to the `GeneralisedShapeletTransform`. This is then capable of computing the the discrepancy between a full path and a shapelet, for each shapelet, in a parallelisable manner.

The regularisation will help ensure that the shapelets that are learnt are in fact interpretable (rather than just being random-looking).

Check their docstrings and the paper TODO for more details.

### CPU vs GPU

The code may or may not crash if ran on the GPU. OpenMP and CUDA don't always seem to play well with each other.

# Limitations

There are a number of limitations with this implementation - this definitely constitutes research code, not production code!

In brief, the code is slow and memory inefficient. We're pretty sure that this is mostly just a limitation of the implementation though - in principle the shapelet transform shouldn't suffer from any of these issues.

(Footnote - if anyone writes a better `torchshapelets` then we'll happily link to it as superseding this code.)

If you're curious, having a look at `LIMITATIONS.md` for more information.

# Citation

TODO