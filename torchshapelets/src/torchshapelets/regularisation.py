def similarity_regularisation(shapelet_similarity):
    """The regularisation penalty for ensuring that shapelets look like the input data.

    Arguments:
        shapelet_similarity: A tensor of shape (..., num_shapelets), where ... is any number of batch dimensions,
            representing the similarity between each sample and each of the shapelets.

    Returns:
        A scalar for the regularisation penalty.
    """
    return shapelet_similarity.sum(dim=-1).min()
