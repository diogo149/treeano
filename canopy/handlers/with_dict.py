from . import base


class CallWithDict(base.NetworkHandlerImpl):

    """
    allows calling a function with a map/dict instead of positional arguments
    """

    def transform_compile_function_kwargs(self, state, inputs, **kwargs):
        assert isinstance(inputs, dict)
        self.input_key_order_ = []
        new_inputs = []
        for k, v in inputs.items():
            self.input_key_order_.append(k)
            new_inputs.append(v)

        kwargs["inputs"] = new_inputs
        return kwargs

    def call(self, fn, in_dict, **kwargs):
        assert isinstance(in_dict, dict)
        new_args = [in_dict[k] for k in self.input_key_order_]
        return fn(*new_args, **kwargs)

call_with_dict = CallWithDict


class ReturnDict(base.NetworkHandlerImpl):

    """
    has a function return a map/dict instead of positional outputs
    """

    def transform_compile_function_kwargs(self, state, outputs, **kwargs):
        assert isinstance(outputs, dict)
        self.output_key_order_ = []
        new_outputs = []
        for k, v in outputs.items():
            self.output_key_order_.append(k)
            new_outputs.append(v)

        kwargs["outputs"] = new_outputs
        return kwargs

    def call(self, fn, *args, **kwargs):
        res = fn(*args, **kwargs)
        assert len(res) == len(self.output_key_order_)
        return {k: v for k, v in zip(self.output_key_order_, res)}

return_dict = ReturnDict
