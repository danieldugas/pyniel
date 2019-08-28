import pickle
import os

def freeze_test_case(func, args, kwargs={}, folder="./", case_name="testcase", omit_input=False):
    """ Automated storage of test cases

    Takes an input, and function
    runs the input through the function to get an output
    then saves both input and output to a test_data.npy file

    Func should take ndarray as input, and as output
    """
    # generate reference test output
    outputs = func(*args, **kwargs)

    # omit inputs if desired (for example if they should be loaded separately and are memory-heavy)
    if omit_input:
        args = 'omitted'
        kwargs = 'omitted'
    # save inputs and outputs to file
    data = {'args':args, 'kwargs':kwargs, 'outputs':outputs, 'funcname':func.__name__}

    folder = os.path.expandvars(os.path.expanduser(folder))
    filepath = os.path.join(folder, case_name + '.pickle')
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print("Tests result saved to {}".format(filepath))
    return outputs, filepath

def load_test_case(folder="./", case_name="testcase"):
    folder = os.path.expandvars(os.path.expanduser(folder))
    filepath = os.path.join(folder, case_name + '.pickle')
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data

