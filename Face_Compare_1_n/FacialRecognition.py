from abc import ABC, abstractmethod
from typing import Any, Union, List, Tuple
import numpy as np
import tensorflow as tf

tf_version = int(tf.__version__.split(".", maxsplit=1)[0])
if tf_version == 2:
    from tensorflow.keras.models import Model
else:
    from keras.models import Model

# Notice that all facial recognition models must be inherited from this class

# pylint: disable=too-few-public-methods
class FacialRecognition(ABC):
    model: Union[Model, Any]
    model_name: str
    input_shape: Tuple[int, int]
    output_shape: int


    @abstractmethod
    def find_embeddings(img: np.ndarray) -> List[float]:
        pass
