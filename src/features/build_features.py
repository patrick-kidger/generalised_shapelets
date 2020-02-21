from src.features.signatures.augmentations import *


# Dictionary of augmentations
AUGMENTATIONS = {
    'leadlag': LeadLag(),
    'penoff': PenOff(),
    'addtime': AddTime(),
    'cumsum': CumulativeSum(),
    'appendzero': AppendZero()
}


def apply_augmentation_list(data, aug_list):
    """Applies augmentations to the data if specified in list format with keys corresponding to AUGMENTATIONS.keys().

    This will build a sklearn pipeline from the augmentation list, as such, each augmentation must operate a fit and
    a transform method.

    Example:
        >>> out_data = apply_augmentation_list(data, ['addtime', 'leadlag'])
        # Is equivalent to
        >>> out_data = LeadLag().transform(AddTime().transform(data))

    Args:
        data (torch.Tensor): [N, L, C] shaped data.
        aug_list (list): A list of augmentation strings that correspond to an element of AUGMENTATIONS.

    Returns:
        torch.Tensor: Data with augmentations applied in order.
    """
    # Get correspondenet classes
    augs = [
        (tfm_str, AUGMENTATIONS[tfm_str]) for tfm_str in aug_list
    ]

    try:
        pipeline = Pipeline(augs)
    except:
        print('h')

    # Tranform
    data_tfmd = pipeline.fit_transform(data)

    return data_tfmd


