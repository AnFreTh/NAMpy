import tensorflow as tf


def flatten_list(input_list):
    output_list = []
    for item in input_list:
        if isinstance(item, list):
            output_list.extend(flatten_list(item))
        else:
            output_list.append(item)
    return output_list


def check_is_float_type(tensor):
    dtype = tf.as_dtype(tensor.dtype)
    return dtype.is_floating
