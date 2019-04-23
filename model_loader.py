import os
import numpy as np
import tensorflow as tf

class _ModelLoader(object):
    def __init__(self):
        """Creates a `_ModelLoader` object.
        """
        self._session = tf.Session()

    def load(self):
        raise NotImplementedError('load: Not Implemented')

    def _convert_tensor_name(self, name):
        """replace '/' or ':' in the name with '-'
        """
        return name.replace('/', '_').replace(':', '_')

    def get_conv_tensors(self):
        """Return output tensors of all convolution layers as a list
        """
        conv_tensors = []
        graph = tf.get_default_graph()
        for op in graph.get_operations():
            if op.type == 'Conv2D':
                conv_tensors.append(op.outputs[0])
        return conv_tensors

    def run_tensor(self, tensor_name, inputs):
        """Evaluates a tensor.
        
        Args:
            tensor_name: Name of the tensor
            inputs: Input image, a np array

        Returns:
            A tuple (name, value)
        """
        graph = tf.get_default_graph()
        tensor = graph.get_tensor_by_name(tensor_name)
        outputs = self._session.run(tensor)
        name = self._convert_tensor_name(tensor_name)
        return (name, outputs)

    def default_value_for_input(self, tensor):
        """Feed default values into other Placeholders.
        
        Args:
            tensor: Tensor of Placeholder

        Returns:
            A tuple (tensor, default_value)
        """
        if tensor.shape.dims is None: # Not specific shape.
            default_value = False if tensor.dtype.is_bool else 0
        else:
            # Conver the TensorShape to common shape.
            np_shape = tuple([x if str(x).isdigit() else 0 for x in tensor.shape])
            # Default value compatible with the shape of Placeholder tensor
            default_value = np.zeros(shape=np_shape, dtype=tensor.dtype.as_numpy_dtype)
        return (tensor, default_value)

    def get_input_dict(self, inputs):
        """Finds all the `Placeholders` in the computing graph and searches
        for the best-matched `ones` to feed the image into.
        
        Args:
            inputs: Input image, a numpy array.

        Returns:
            Feed_dict for the corresponding input values.
        """
        graph = tf.get_default_graph()
        fead_list = []

        for tensor in tf.contrib.graph_editor.get_tensors(graph)[:]:
            if tensor.op.type == 'Placeholder':
                # Placeholder for the input image with or without the dimension of batch size.
                if tensor.shape.dims is not None and len(tensor.shape) in [3, 4]:
                    # Whether the shape of the Placeholder is compatible with the shape of the input image.
                    if  tensor.shape[-1].is_compatible_with(tf.Dimension(inputs.shape[-1])) and \
                        tensor.shape[-2].is_compatible_with(tf.Dimension(inputs.shape[-2])) and \
                        tensor.shape[-3].is_compatible_with(tf.Dimension(inputs.shape[-3])):
                        # Added the dimension of batch size for inputs if necessary.
                        if len(tensor.shape) == 4:
                            inputs = inputs.reshape((1, inputs.shape[0], inputs.shape[1], inputs.shape[2]))
                        # Once the target Placeholder was found, feed the input image to it.
                        print 'Found input tensor: {}'.format(tensor.name)
                        fead_list.append((tensor, inputs))
                        continue
                # Found other Placeholders and feed default values to them.
                feed_pair = self.default_value_for_input(tensor)
                fead_list.append(feed_pair)

        return dict(fead_list)

    def run_conv_tensors(self, inputs):
        """Evaluates tensors of all convolution layers

        Args:
            inputs: Input image, a np array

        Returns:
            A list of tuples (name, value)
        """
        conv_tensors = self.get_conv_tensors()
        names = []
        for t in conv_tensors:
            names.append(self._convert_tensor_name(t.name))
        
        # graph = tf.get_default_graph()
        # input_tensor = graph.get_tensor_by_name(name)

        input_dict = self.get_input_dict(inputs)

        if input_dict is None:
            print 'Can not find the input tensor!'
            return
        outputs = self._session.run(conv_tensors, feed_dict=input_dict)
        return zip(names, outputs)


class MetaGraphLoader(_ModelLoader):
    """Class used to restore meta graph
    """
    def __init__(self, restore_from=None):
        """Creates a `MetaGraphLoader` object.

        Args:
            input_name: Name of input tensor
            restore_from: meta graph
        """
        super(MetaGraphLoader, self).__init__()
        self._restore_from = restore_from

    def load(self):
        """Restore from meta graph
        """
        saver = tf.train.import_meta_graph(self._restore_from + '.meta',
                                           clear_devices=True)
        saver.restore(self._session, self._restore_from)
        graph = tf.get_default_graph()

        print 'Restored model parameters from {}'.format(self._restore_from)
