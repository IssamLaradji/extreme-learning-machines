# Authors: Andreas Mueller
#          Manoj Kumar
#          Issam H. Laradji
# License: BSD 3 clause

import numpy as np


def compute_class_weight(class_weight, classes, y):
    """Estimate class weights for unbalanced datasets.

    Parameters
    ----------
    class_weight : dict, 'auto' or None
        If 'auto', class weights will be given inverse proportional
        to the frequency of the class in the data.
        If a dictionary is given, keys are classes and values
        are corresponding class weights.
        If None is given, the class weights will be uniform.

    classes : ndarray
        Array of the classes occurring in the data, as given by
        ``np.unique(y_org)`` with ``y_org`` the original class labels.

    y : array-like, shape (n_samples,)
        Array of original class labels per sample;

    Returns
    -------
    class_weight_vect : ndarray, shape (n_classes,)
        Array with class_weight_vect[i] the weight for i-th class
    """
    # Import error caused by circular imports.
    from sklearn.preprocessing import LabelEncoder

    if class_weight is None or len(class_weight) == 0:
        # uniform class weights
        weight = np.ones(classes.shape[0], dtype=np.float64, order='C')
    elif class_weight == 'auto':
        # Find the weight of each class as present in y.
        le = LabelEncoder()
        y_ind = le.fit_transform(y)
        if not all(np.in1d(classes, le.classes_)):
            raise ValueError("classes should have valid labels that are in y")

        # inversely proportional to the number of samples in the class
        recip_freq = 1. / np.bincount(y_ind)
        weight = recip_freq[le.transform(classes)] / np.mean(recip_freq)
    else:
        # user-defined dictionary
        weight = np.ones(classes.shape[0], dtype=np.float64, order='C')
        if not isinstance(class_weight, dict):
            raise ValueError("class_weight must be dict, 'auto', or None,"
                             " got: %r" % class_weight)
        for c in class_weight:
            i = np.searchsorted(classes, c)
            if classes[i] != c:
                raise ValueError("Class label %d not present." % c)
            else:
                weight[i] = class_weight[c]

    return weight


def compute_sample_weight(class_weight, classes, y):
    """Compute sample weights for unbalanced datasets.

    Parameters
    ----------
    class_weight : dict, 'auto' or None
        If 'auto', class weights will be given inverse proportional
        to the frequency of the class in the data.
        If a dictionary is given, keys are classes and values
        are corresponding class weights.
        If None is given, the class weights will be uniform.

    classes : ndarray
        Array of the classes occurring in the data, as given by
        ``np.unique(y_org)`` with ``y_org`` the original class labels.

    y : array-like, shape (n_samples,)
        Array of original class labels per sample;

    Returns
    -------
    sample_weight : ndarray, shape (n_samples,)
        Array where sample_weight[i] denotes the weight for the i-th sample
    """
    # Import error caused by circular imports.
    from sklearn.preprocessing import LabelEncoder

    if class_weight is None:
        return None

    weight = compute_class_weight(class_weight, classes, y)
    sample_weight = np.zeros(y.shape[0])

    # Check if every y sample belongs to a class in classes
    if not all(np.in1d(np.unique(y), classes)):
        raise ValueError("'y' has classes not in 'classes'.")

    le = LabelEncoder()
    y_ind = le.fit_transform(y)

    for class_ in np.unique(y_ind):
        indices = np.where(y_ind == class_)[0]
        sample_weight[indices] = weight[class_]

    return sample_weight