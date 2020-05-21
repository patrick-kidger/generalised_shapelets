# Limitations

As stated in `README.MD`, this code is pretty slow, and pretty memory inefficient. It was sufficient for our purposes in writing the paper, but it's not "production ready". Here we discuss a bit about why these limitations exist.

Writing better versions is possible but takes more time than we really have!

## Speed

This code isn't very fast, despite the parallelisation via OpenMP. These are some of the reasons why.

#### High level PyTorch tensors
The hot loops are still written in terms of PyTorch tensors, rather than C++ primitives.

#### CPU vs GPU
The code is practically speaking CPU-only, and doesn't really work on the GPU. This is because OpenMP and CUDA don't seem to play well with each other.

This is a shame, as in principle shapelets are a massively parallel and therefore GPU-friendly operation, with the potential to parallelise over batch, over different shapelets, and over the continuous-minimum operation. But we don't think there's an easy way to write this parallelisation using just PyTorch (for the general case of irregularly sampled data), so this would need a custom GPU kernel.

#### Generality of implementation
This code is written to handle the general case, but in many practical cases things can be simplified (a lot!), and furthermore there is an obvious corner that can be cut to speed things up.

The code handles irregular-sampled data, with arbitrarily good approximations to the minimum-over-region. (See the bit in the paper about approximating the minimum.) In this case both the shapelet and the time series are piecewise linear, but on a _different set of knots_. Therefore every single time we compute a discrepancy, we take the union of their knots and rewrite them each as piecewise linear functions on this enlarged set of knots.

In practice, if the data is regularly sampled and the approximation-to-minimum uses a step that coincides with the sampling rate, then the knots will coincide, and this whole very expensive procedure can be avoided.

(With one caveat: using learnt lengths means that each shapelet doesn't have a length that is a multiple of the sampling rate, so there may be one knot that needs to be inserted for each comparison, partway in between two samples.)

Even in the general case, an obvious corner that could be cut to speed things up is not to put both the shapelet and the slice-of-time-series on the same set of knots, but just to project one onto the set of knots used by the other. Depending on the application, this may be a viable approximation.

## Memory usage

The memory usage can be a bit much.

We don't completely understand why this is. It doesn't seem like it's an intrinsic part of the algorithm, as this occurs to a certain extent even during inference, which in principle can obviously be done with very little memory.

It might just be a particular class of memory leak, but we do observe the memory usage being constant across batches, so it's not a really bad one that sticks around forever.

We think we have tracked down one of the reasons for high memory usage during training: it's because of the minimum-over-region. (`continuous_min` in the C++ code.) We take a minimum over a _lot_ of things, and in order to backpropagate later, PyTorch holds all of it in memory.

Now suppose `a < b` and then compute  `c = min(a, b)`. Then the gradient on `c` backpropagates to the gradient on `a`, and `b` gets a zero gradient. So we see that in order to compute gradients, we actually don't need to hold `b` in memory at all.

Thus in our case, where we have `c = min(lots and lots and lots of things)`, it should be possible to write a more-efficient `min` which only retains the single entry that actually achieved the minimum.

For a single element this is easy: compute the argmin over the lots-and-lots-of-things, then extract that entry and use that as `c`. argmin isn't a differentiable operation, so as long as the lots-and-lots-of-things is something like `std::vector` rather than a `torch::Tensor`, then all of the other entries get free'd if we don't keep them around.

The caveat that makes this tricky is batching: each batch element is likely to have a different minimum, so really the tensors that need to be recorded have to be on a per-batch-element basis, which makes writing fast code for that somewhat trickier.

That said it seems unlikely that this completely explains the high memory usage, especially given that this should be a training-time-only problem not affecting inference time.