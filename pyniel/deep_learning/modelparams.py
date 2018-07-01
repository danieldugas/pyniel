import numpy as np
import tensorflow as tf


class ModelParams:
    """
  Base class for storing parameters of tensorflow models.
  """

    def __init__(self):
        """
    initializer provided as an example.
    should be overriden with specific default parameters for child classes.
    """
        self.INPUT_SHAPE = [16]
        self.OUTPUT_SHAPE = [3]
        self.LAYERS = [{"shape": [128]}, {"shape": [128]}, {"shape": [128]}]
        self.LEARNING_RATE = 0.001
        self.DROPOUT = 0.90  # Keep-prob
        self.FLOAT_TYPE = tf.float32
        self.DISABLE_SUMMARY = False

    def __str__(self):
        return str(self.__dict__)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not self.__eq__(other)
