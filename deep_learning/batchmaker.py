import numpy as np

# Default data processing function - extract the element at pos in input_data.
# data processing functions should take input_data and a position as input,
# and return an array of shape [example_shape]
def extract_single_example(input_data, pos):
    return input_data[pos]

class Batchmaker:
    def __init__(self, input_data, examples_per_batch, example_shape,
                 shuffle_examples=True, data_processing_func=extract_single_example):
        self.input_data = input_data
        self.example_shape = example_shape
        self.data_processing_func = data_processing_func
        # examples per batch
        if examples_per_batch is "max":
            examples_per_batch = len(input_data)
        assert type(examples_per_batch) is int
        if examples_per_batch > len(input_data):
            print("WARNING: more examples per batch than possible examples in all input_data")
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
        batch_input_values  =  np.zeros([self.examples_per_batch] + self.example_shape)
        for i_example in range(self.examples_per_batch):
          # Create training example at index 'pos' in input_data.
          pos = self.remaining_example_indices.pop(0)
          #   input.
          batch_input_values[i_example] = np.reshape(self.data_processing_func(self.input_data, pos),
                                                     self.example_shape)

        self.batches_consumed_counter += 1

        return batch_input_values

    def is_depleted(self):
        return len(self.remaining_example_indices) < self.examples_per_batch

    def n_batches_remaining(self):
        return len(self.remaining_example_indices) / self.examples_per_batch

    def n_batches_consumed(self):
        return self.batches_consumed_counter

def progress_bar(batchmaker):
  from matplotlib import pyplot as plt  
  import time
  plt.figure('progress_bar')
  plt.scatter(time.time(), batchmaker.n_batches_consumed())
  plt.ylim([0, batchmaker.n_batches_consumed()+batchmaker.n_batches_remaining()])
  plt.show()
  plt.gcf().canvas.draw()
  time.sleep(0.0001)
