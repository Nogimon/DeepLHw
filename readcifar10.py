def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

data = unpickle('./cifar-10-batches-py/data_batch_1')
training_inputs = data['data']
training_results = data['labels']