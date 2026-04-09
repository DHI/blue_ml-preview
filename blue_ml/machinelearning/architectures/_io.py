from blue_ml.machinelearning.architectures.base_class import BlueMLModel


def read(filename: str) -> BlueMLModel:
    """Read a `BlueMLModel` from a file.

    Parameters
    ----------
    filename : str
        File to read from

    Returns
    -------
    BlueMLModel
        Generic Blue_ML machine learning model

    Raises
    ------
    ValueError
        If the object loaded from the file is not a `BlueMLModel`
    """
    return BlueMLModel.read(filename)
