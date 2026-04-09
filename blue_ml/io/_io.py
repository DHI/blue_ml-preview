"""Functions to serialize and deserialize objects to and from json format.

The files are then packed in zip file, including some meta data.

Model decoding:

- The model is first loaded from the zip file.
- A meta.json file contains the version number of blue_ml and points to the blue_ml model in JSON format.
- The blue_ml model is then deserializd using BlueEncoder and BlueDecoder.
    - if OBJNAME = "__obj__.__name__" and OBJMODULE = "__obj__.__module__" then the object is deserialized using
        SerializeObject.decode(dct). This is where the BlueML model is deserialized.
    - For keras or sklearn, a subclass of deserialzing is used. This is encountered


"""

import json
import sys
import zipfile
from abc import ABC, abstractmethod
from datetime import datetime
from io import BytesIO
from typing import Any, Dict

import numpy as np
import pandas as pd
import xarray as xr
from keras.src.models.functional import Functional  # type: ignore[import-untyped]
from keras.src.models.sequential import Sequential  # type: ignore[import-untyped]
from keras.src.saving import saving_lib  # type: ignore[import-untyped]
from sklearn.base import BaseEstimator  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]


def get_version():
    # Import here to avoid circular imports
    from blue_ml import __version__

    return __version__


class _EncodeDecode(ABC):
    OBJNAME = "__obj__.__name__"
    OBJMODULE = "__obj__.__module__"
    TYPENAME = "__type__.__name__"
    TYPEMODULE = "__type__.__module__"
    KERASMODEL = "__keras__model__"
    ONNXMODEL = "__onnx__model__"
    SLICEOBJ = "__obj__.__slice__"
    MODELPATH = "FILEPATH:"

    _safe_modules = {
        "xarray.Dataset",
        "xarray.DataArray",
        "blue_ml.timeseries.transforms._transforms",
        "mikeio.eum._eum",
        "numpy.core.multiarray",
        "numpy",
        "blue_ml.timeseries.pipeline._pipeline",
        "blue_ml.machinelearning.architectures.keras.keras",
        "blue_ml.machinelearning.architectures.onnx.onnx",
        "blue_ml.machinelearning.architectures.regression.gradient_boosting",
        "blue_ml.machinelearning.architectures.regression.linear",
        "blue_ml.machinelearning.windowgenerator",
        "builtins",
        "keras.src.models.functional",
        "keras.src.models.sequential",
        "tensorflow.python.trackable.data_structures",
        "sklearn.preprocessing._data",
        "_io",
    }

    @abstractmethod
    def encode(obj): ...

    @abstractmethod
    def decode(dct): ...


class SerializeObject(_EncodeDecode):
    def encode(obj):
        """Serialize object to json dict."""
        dct_props = {}
        dct_props[_EncodeDecode.OBJNAME] = obj.__class__.__name__
        dct_props[_EncodeDecode.OBJMODULE] = obj.__class__.__module__
        dct_props["__dict__"] = obj.__dict__.copy()

        return dct_props

    def decode(dct):
        module = dct[_EncodeDecode.OBJMODULE]
        name = dct[_EncodeDecode.OBJNAME]
        if module in _EncodeDecode._safe_modules:
            if "__dict__" in dct.keys():
                obj_dict = dct["__dict__"].copy()
                obj_value = None
                if "_value_" in obj_dict.keys():
                    obj_value = obj_dict["_value_"]
            else:
                obj_dict = obj_value = None

            cls = getattr(sys.modules[module], name)
            if obj_value is not None:
                obj = cls.__new__(cls, obj_value)
            else:
                obj = cls.__new__(cls)
            obj.__dict__.update(**obj_dict)
            return obj
        else:
            raise ValueError(f"Module {module} is not in safe modules")


class SerializeType(_EncodeDecode):
    def encode(obj):
        """Serialize object type to json dict."""
        dct_props = {}
        dct_props[_EncodeDecode.TYPENAME] = obj.__name__
        dct_props[_EncodeDecode.TYPEMODULE] = obj.__module__

        return dct_props

    def decode(dct):
        module = dct[_EncodeDecode.TYPEMODULE]
        name = dct[_EncodeDecode.TYPENAME]
        if module in _EncodeDecode._safe_modules:
            cls = getattr(sys.modules[module], name)
            return cls
        else:
            raise ValueError(f"Module {module} is not in safe modules")


class SerializeKerasModel(_EncodeDecode):
    def _filepath(model):
        """Get the filepath of the model."""
        model_name = model.name
        filepath = f"{model_name}.keras"
        return filepath

    def encode_to_filename(obj):
        filepath = SerializeKerasModel._filepath(obj)
        return {_EncodeDecode.KERASMODEL: f"{_EncodeDecode.MODELPATH}{filepath}"}

    def write_to_zip(model, zipf):
        # Serialize object to json
        filepath = SerializeKerasModel._filepath(model)

        byte_io = BytesIO()
        saving_lib._save_model_to_fileobj(model, byte_io, weights_format="h5")
        with zipf.open(filepath, "w") as fwrite:
            # Save to zip file
            fwrite.write(byte_io.getvalue())

    def decode(dct, zipf):
        """Find the filename of the model in the zip file and load it.

        Replace the filename entry with the actual deserialized object.
        """
        filepath = dct.pop(_EncodeDecode.KERASMODEL).split(_EncodeDecode.MODELPATH)[1]
        with zipf.open(filepath, "r") as fopen:
            byte_io = BytesIO(fopen.read())

        model = saving_lib._load_model_from_fileobj(
            byte_io, custom_objects=None, compile=True, safe_mode=True
        )

        return model


class SerializeSklearnModel(_EncodeDecode):
    def encode_as_onnx(obj):
        # We just need to convert to BlueOnnx - the encoding will then catch
        # the model as a SerializeOnnxModel inside BlueML serialization
        model = obj.to_onnx_model()
        return model


class SerializeSlice(_EncodeDecode):
    def encode(obj):
        """Serialize slice to json dict."""
        dct_props = {}
        dct_props["start"] = obj.start
        dct_props["stop"] = obj.stop
        dct_props["step"] = obj.step

        return {_EncodeDecode.SLICEOBJ: dct_props}

    def decode(dct):
        slice_properties = dct.pop(_EncodeDecode.SLICEOBJ)
        return slice(
            slice_properties["start"],
            slice_properties["stop"],
            slice_properties.get("step", None),
        )


class BlueEncoder(json.JSONEncoder):
    """Subclass to extend JSONEncoder to handle more types."""

    def default(self, obj):
        if isinstance(obj, range):
            return list(obj)
        # Handling for numpy types
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return {"__ndarray__": obj.tolist()}
        elif isinstance(obj, slice):
            return SerializeSlice.encode(obj)
        # Handling for datetime objects
        elif isinstance(obj, datetime):
            return {"__datetime__": obj.isoformat()}
        elif isinstance(obj, pd.Timestamp):
            return {"__pd_timestamp__": obj.isoformat()}
        # Handling for xarray objects
        elif isinstance(obj, xr.Dataset):
            return {"__xr_dataset__": obj.to_dict()}
        elif isinstance(obj, xr.DataArray):
            return {"__xr_dataarray__": obj.to_dict()}
        # Handling for pandas objects
        elif isinstance(obj, pd.DatetimeIndex):
            return {"__pd_datetimeindex__": obj.to_list()}

        # Handling for custom classes
        if Pipeline in obj.__class__.__mro__ or any(
            module in obj.__class__.__module__
            for module in (
                "blue_ml.timeseries.transforms._transforms",
                "blue_ml.machinelearning.architectures.onnx.onnx",
                "blue_ml.machinelearning.architectures.keras.keras",
                "blue_ml.machinelearning.windowgenerator",
                "sklearn.preprocessing._data",
            )
        ):
            return SerializeObject.encode(obj)

        # "Raw" models - we replace the model with the filename in the json object
        # and save the model seeprately
        ## Raw Keras models
        elif any(
            module in obj.__class__.__mro__ for module in (Sequential, Functional)
        ):
            return SerializeKerasModel.encode_to_filename(obj)

        # Handling for class types
        elif isinstance(obj, type):
            return SerializeType.encode(obj)

        # All other "normal" data types
        else:
            return super(BlueEncoder, self).default(obj)

    def encode(self, obj):
        return super().encode(self.wrap_obj(obj))

    def wrap_obj(self, obj: Any, **metakwargs) -> Dict:
        """Wrap object in a dictionary with meta data."""
        dct_obj = {}
        # Automatically add version
        dct_obj["__version__"] = get_version()
        # Add meta kwargs
        dct_obj.update(metakwargs)
        # The object is the main
        dct_obj["__object__"] = obj
        return dct_obj


class BlueDecoder(json.JSONDecoder):
    """Subclass to extend JSONDecoder to handle more types."""

    def __init__(self, zipf=None, *args, **kwargs):
        self.zipf = zipf
        super().__init__(object_hook=self.deserialize_json, *args, **kwargs)

    def deserialize_json(self, dct):
        if _EncodeDecode.TYPENAME and _EncodeDecode.TYPEMODULE in dct:
            cls = SerializeType.decode(dct)
            return cls

        # Generic object
        if _EncodeDecode.OBJNAME and _EncodeDecode.OBJMODULE in dct:
            obj = SerializeObject.decode(dct)
            return obj

        if _EncodeDecode.KERASMODEL in dct:
            dct = SerializeKerasModel.decode(dct, self.zipf)
            return dct

        # Slice
        if _EncodeDecode.SLICEOBJ in dct:
            dct = SerializeSlice.decode(dct)
            return dct

        # datetime objects
        if "__datetime__" in dct:
            return datetime.fromisoformat(dct["__datetime__"])

        if "__pd_timestamp__" in dct:
            return pd.Timestamp(dct["__pd_timestamp__"])

        # xarray objects
        if "__xr_dataset__" in dct:
            return xr.Dataset.from_dict(dct["__xr_dataset__"])

        if "__xr_dataarray__" in dct:
            return xr.DataArray.from_dict(dct["__xr_dataarray__"])

        # pandas objects
        if "__pd_datetimeindex__" in dct:
            return pd.DatetimeIndex(dct["__pd_datetimeindex__"])

        if "__ndarray__" in dct:
            dct = np.array(dct["__ndarray__"])
            return dct

        return dct

    def decode(self, s, zipf=NotImplemented):
        obj = super().decode(s)
        return obj


class ZipIO:
    """Class to handle zip files.

    Can pack multiple objects in a zip file + some meta data.
    """

    @staticmethod
    def verify_zip(zipfilestream):
        if "meta.json" not in [f.filename for f in zipfilestream.filelist]:
            raise ValueError("Not a bmf object")

    @staticmethod
    def write_zip(filepath: str, objects: Dict[str, Any]) -> None:
        meta = {
            "blue_ml.__version__": get_version(),
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_objects": len(objects),
            "objects": list(objects.keys()),
        }

        # Create a new zip file or open an existing one
        with zipfile.ZipFile(filepath, "w") as zipf:
            with zipf.open("meta.json", "w") as fwrite:
                # Meta data to JSON
                json_obj = json.dumps(meta, indent=4)
                # Write in zip file
                fwrite.write(json_obj.encode("utf-8"))

            # Write a file directly into the zip file
            for key, value in objects.items():
                # Serialize object to json
                # We have a special case were we seperate the model from the json object

                is_blueml_model = isinstance(
                    value,
                    getattr(
                        sys.modules["blue_ml.machinelearning.architectures.base_class"],
                        "BlueMLModel",
                    ),
                )

                # BlueML models store the train and test data. Storing it as a Timeseries causes
                # problems during serialization, so they are stored as xarray.Dataset, which needs to be
                # converted back when reading
                if is_blueml_model:
                    blue_model = value.copy()
                    blue_model._train_ts = value._train_ts.as_xarray()
                    blue_model._test_ts = value._test_ts.as_xarray()
                    json_obj = json.dumps(blue_model, indent=4, cls=BlueEncoder)
                else:
                    json_obj = json.dumps(value, indent=4, cls=BlueEncoder)

                # Part 1: json_obj
                with zipf.open(f"{key}.json", "w") as fwrite:
                    # Save to zip file
                    fwrite.write(json_obj.encode("utf-8"))

                if is_blueml_model:
                    model = value.model
                    ZipIO._write_model_to_zip(model, zipf, key)

    @staticmethod
    def _write_model_to_zip(model, zipf, key):
        """Write the model to the zip file."""
        if isinstance(model, (Functional, Sequential)):
            SerializeKerasModel.write_to_zip(model, zipf)

        else:
            raise ValueError(
                f"Model type {type(model)} not supported for serialization"
            )

    @staticmethod
    def read_zip(filepath: str) -> Dict:
        # Import here to avoid circular imports
        from blue_ml import Timeseries
        from blue_ml.timeseries import TimeseriesFactory

        with zipfile.ZipFile(filepath, "r") as zipf:
            # Ensure that the zip file is a blue_ml file (.bmf) object
            ZipIO.verify_zip(zipf)
            filelist = [f.filename for f in zipf.filelist]
            objects = {}

            meta = json.loads(zipf.read("meta.json"))
            objects["meta"] = meta

            for name in meta["objects"]:
                filename_json = f"{name}.json"
                if filename_json not in filelist:
                    raise ValueError(f"Object {name} not found in zip file")
                else:
                    zip_obj = zipf.read(filename_json).decode("utf-8")
                    data = BlueDecoder(zipf=zipf).decode(zip_obj)

                    # BlueML models store the train and test data. Storing it as a Timeseries causes
                    # problems during serialization, so they are stored as xarray.Dataset, which needs to be
                    # converted back when reading
                    obj = data["__object__"]
                    if name == "blue_ml_model":
                        train_ds = obj._train_ts
                        test_ds = obj._test_ts
                        feature_names = obj._feature_names
                        target_names = obj._target_names
                        obj._train_ts = Timeseries(
                            train_ds,
                            features=feature_names,
                            targets=target_names,
                        )
                        if len(test_ds["time"]) > 0:
                            obj._test_ts = Timeseries(
                                test_ds,
                                features=feature_names,
                                targets=target_names,
                            )
                        else:
                            obj._test_ts = TimeseriesFactory.empty()

                    objects[name] = obj

            return objects
