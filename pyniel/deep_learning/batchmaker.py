import numpy as np

# TODO: pregenerate batches, etc...


def basic_extractor(input_data, pos):
    """
    A basic extractor for extracting values from the input_data
    provided as an example.
    """
    return input_data[pos]


class Batchmaker:
    """
    Generates batches based on input_data

    Generated batches have shape (examples_per_batch, example_shape)
    and are extracted from the input_data according to extractor_func.
    In the simplest case the extractor function takes the form of basic_extractor.
    """

    def __init__(
        self,
        input_data,
        examples_per_batch,
        example_shape,
        shuffle_examples=True,
        extractor_func=basic_extractor,
    ):
        self.input_data = input_data
        self.example_shape = example_shape
        self.extractor_func = extractor_func
        # examples per batch
        if examples_per_batch is "max":
            examples_per_batch = len(input_data)
        assert type(examples_per_batch) is int
        if examples_per_batch > len(input_data):
            print(
                "WARNING: more examples per batch than possible examples in all input_data"
            )
            self.examples_per_batch = len(input_data)
        else:
            self.examples_per_batch = examples_per_batch
        # initialize example indices list
        self.remaining_example_indices = list(range(len(input_data)))
        # shuffle list if required
        if shuffle_examples:
            from random import shuffle

            shuffle(self.remaining_example_indices)
        self.batches_consumed_counter = 0

    def next_batch(self):
        assert not self.is_depleted()
        # Create a single batch
        batch_input_values = np.zeros([self.examples_per_batch] + self.example_shape)
        for i_example in range(self.examples_per_batch):
            # Create training example at index 'pos' in input_data.
            pos = self.remaining_example_indices.pop(0)
            #   input.
            batch_input_values[i_example] = np.reshape(
                self.extractor_func(self.input_data, pos), self.example_shape
            )

        self.batches_consumed_counter += 1

        return batch_input_values

    def is_depleted(self):
        return len(self.remaining_example_indices) < self.examples_per_batch

    def n_batches_remaining(self):
        return len(self.remaining_example_indices) / self.examples_per_batch

    def n_batches_consumed(self):
        return self.batches_consumed_counter


class FieldBatchmaker(Batchmaker):
    """
    A Batch contains one or several fields,
    which each are an array of shape (examples_per_batch, field_shape)
    field_shape being the shape of a single example in this field

    for example, take the case where a batch needs to be created with the following fields:
    x: shape = (16)        e.g. feature vector of size 16
    y: shape = (3)         e.g. one_hot encoding with 3 labels

    each example in this batch has a corresponding x and y field,
    for tensorflow we batch together all the x fields for several examples into an array
    of shape (examples_per_batch, 16)
    and all the y fields for the same examples into an array of shape (examples_per_batch, 3)

    In this case fields_format would be {'x': {'shape': (16,), 'extractor_func': x_extractor_func},
                                         'y': {'shape': (3,),  'extractor_func': y_extractor_func}}

    An extractor function should be provided to extract each field from the provided input data
    based on an example position.

    In the simplest case an extractor function takes the form of basic_extractor(input_data, pos).
    """

    def __init__(
        self, input_data, examples_per_batch, fields_format, shuffle_examples=True
    ):
        self.input_data = input_data
        self.fields_format = fields_format
        # examples per batch
        if examples_per_batch is "max":
            examples_per_batch = len(input_data)
        assert type(examples_per_batch) is int
        if examples_per_batch > len(input_data):
            print(
                "WARNING: more examples per batch than possible examples in all input_data"
            )
            self.examples_per_batch = len(input_data)
        else:
            self.examples_per_batch = examples_per_batch
        # initialize example indices list
        self.remaining_example_indices = list(range(len(input_data)))
        # shuffle list if required
        if shuffle_examples:
            from random import shuffle

            shuffle(self.remaining_example_indices)
        self.batches_consumed_counter = 0

    def next_batch(self):
        assert not self.is_depleted()
        batch = {
            field_name: np.zeros((self.examples_per_batch,) + field["shape"])
            for field_name, field in self.fields_format.items()
        }
        # Create a single batch
        for i_example in range(self.examples_per_batch):
            # Create training example at index 'pos' in input_data.
            pos = self.remaining_example_indices.pop(0)
            #   fill values for each field
            for field_name, field in self.fields_format.items():
                batch[field_name][i_example] = field["extractor_func"](
                    self.input_data, pos
                )
                assert batch[field_name][i_example].shape == field["shape"]

        self.batches_consumed_counter += 1

        return batch
