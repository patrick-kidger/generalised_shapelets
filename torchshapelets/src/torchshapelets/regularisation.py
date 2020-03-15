def similarity_regularisation(shapelet_similarity):
    """The regularisation penalty for ensuring that shapelets look like the input data.

    Arguments:
        shapelet_similarity: A tensor of shape (..., num_shapelets), where ... is any number of batch dimensions,
            representing the similarity between each sample and each of the shapelets.

    Returns:
        A scalar for the regularisation penalty.
    """
    return shapelet_similarity.sum(dim=-1).min()


def length_regularisation(shapelet_lengths):
    """The regularisation penalty for ensuring that shapelets are of nontrivial length.

    Arguments:
        shapelet_lengths: A one dimensional tensor of shape (num_shapelets,), representing the length of each shapelet.

    Returns:
        A scalar for the regularisation penalty.
    """
    return shapelet_lengths.reciprocal().sum()


def pseudometric_regularisation(discrepancy_fn):
    """The regularisation penalty for ensuring that the pseudometric is nontrivial.

    Arguments:
        discrepancy_fn: A torch.nn.Module representing the discrepancy function.

    Returns:
        A scalar for the regularisation penalty.
    """
    loss = 0
    for parameter in discrepancy_fn.parameters():
        if parameter.requires_grad:
            loss = loss + parameter.norm().reciprocal()
    return loss
