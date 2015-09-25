"""
implementation of "Fractional Max-Pooling" (http://arxiv.org/abs/1412.6071)

from https://github.com/diogo149/theano_fractional_max_pooling
"""

import numpy as np
import theano
import theano.sandbox.cuda as cuda

from pycuda.compiler import SourceModule

import theano.misc.pycuda_init


class DisjointPseudorandomFractionalMaxPooling2DOp(cuda.GpuOp):

    __props__ = ("alpha", "u")

    def __init__(self, alpha, u):
        assert 1 < alpha < 2
        assert 0 < u < 1
        self.alpha = alpha
        # TODO allow separate u for each axis
        # TODO allow u to be randomly generated
        self.u = u

    def make_node(self, inp):
        def to_gpu_contiguous(v):
            v = cuda.basic_ops.as_cuda_ndarray_variable(v)
            v = cuda.basic_ops.gpu_contiguous(v)
            return v

        inp = to_gpu_contiguous(inp)

        assert inp.dtype == "float32"
        return theano.Apply(self, [inp], [self.output_type(inp)()])

    def output_type(self, inp):
        return cuda.CudaNdarrayType(broadcastable=[False] * (inp.type.ndim))

    def output_length(self, input_length):
        return int(np.floor(input_length / self.alpha))

    # TODO add infer_shape

    def make_thunk(self, node, storage_map, _, _2):
        inputs = [storage_map[v] for v in node.inputs]
        outputs = [storage_map[v] for v in node.outputs]

        mod = SourceModule("""
__global__ void fmp(float * input,
                    float * output,
                    float alpha,
                    float u,
                    int batch_size,
                    int num_channels,
                    int old_map_size,
                    int map_size) {
    // feature dim, fastest varying index!
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    // batch dim
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    int map_size_sq = map_size * map_size;
    int example_size = num_channels * map_size_sq;

    // feature indices (channels, height, width)
    int x_channel = x / map_size_sq;
    int x_f0 = (x % map_size_sq) / map_size;
    int x_f1 = x % map_size;

    int a_before = (x_f0 == 0) ? 0 : ceil(alpha * (x_f0 - 1 + u));
    int a_after = (x_f0 == map_size - 1) ? old_map_size : ceil(alpha * (x_f0 + u));
    int b_before = (x_f1 == 0) ? 0 : ceil(alpha * (x_f1 - 1 + u));
    int b_after = (x_f1 == map_size - 1) ? old_map_size : ceil(alpha * (x_f1 + u));

    int old_map_size_sq = old_map_size * old_map_size;
    int old_example_size = num_channels * old_map_size_sq;
    int input_idx_base = y * old_example_size + old_map_size_sq * x_channel;
    if (x < example_size && y < batch_size) {
      float best = input[input_idx_base  + old_map_size * a_before + b_before];
      for (int a = a_before; a < a_after; a++) {
        for (int b = b_before; b < b_after; b++) {
          best = max(best, input[input_idx_base  + old_map_size * a + b]);
        }
      }
      output[y * example_size + x] = best;
    }
}
""")
        kernel = mod.get_function("fmp")

        def thunk():
            inp_shape = inputs[0][0].shape
            batch_size, num_channels, height, width = inp_shape
            assert height > 1
            # might not be necessary, but let's do it anyway
            assert height == width
            new_dim = self.output_length(height)
            out_shape = (batch_size, num_channels, new_dim, new_dim)

            example_size = num_channels * new_dim * new_dim
            map_size = new_dim

            out = outputs[0]

            # only allocate if there is no previous allocation of the right
            # size.
            if out[0] is None or out[0].shape != out_shape:
                out[0] = cuda.CudaNdarray.zeros(out_shape)

            x_block = 16
            y_block = 16
            block = (x_block, y_block, 1)

            x_grid = int(np.ceil(float(example_size) / x_block))
            y_grid = int(np.ceil(float(batch_size) / y_block))
            grid = (x_grid, y_grid, 1)

            kernel(inputs[0][0],
                   out[0],
                   np.float32(self.alpha),
                   np.float32(self.u),
                   np.intc(batch_size),
                   np.intc(num_channels),
                   np.intc(height),
                   np.intc(map_size),
                   block=block,
                   grid=grid)

        thunk.inputs = inputs
        thunk.outputs = outputs
        thunk.lazy = False

        return thunk

    def grad(self, inputs, grads):
        inp, = inputs
        top, = grads
        top = cuda.basic_ops.gpu_contiguous(top)
        return [DisjointPseudorandomFractionalMaxPooling2DGradOp(
            self.alpha,
            self.u
        )(inp, top)]


class DisjointPseudorandomFractionalMaxPooling2DGradOp(cuda.GpuOp):

    __props__ = ("alpha", "u")

    def __init__(self, alpha, u):
        assert 1 < alpha < 2
        assert 0 < u < 1
        self.alpha = alpha
        # TODO allow separate u for each axis
        # TODO allow u to be randomly generated
        self.u = u

    def make_node(self, inp, grad):
        def to_gpu_contiguous(v):
            v = cuda.basic_ops.as_cuda_ndarray_variable(v)
            v = cuda.basic_ops.gpu_contiguous(v)
            return v

        inp = to_gpu_contiguous(inp)
        grad = to_gpu_contiguous(grad)

        assert inp.dtype == "float32"
        return theano.Apply(self, [inp, grad], [self.output_type(inp, grad)()])

    def output_type(self, inp, grad):
        return cuda.CudaNdarrayType(broadcastable=[False] * (inp.type.ndim))

    def make_thunk(self, node, storage_map, _, _2):
        inputs = [storage_map[v] for v in node.inputs]
        outputs = [storage_map[v] for v in node.outputs]

        mod = SourceModule("""
__global__ void fmp(float * input,
                    float * grad,
                    float * output,
                    float alpha,
                    float u,
                    int batch_size,
                    int num_channels,
                    int old_map_size,
                    int map_size) {
    // feature dim, fastest varying index!
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    // batch dim
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    int map_size_sq = map_size * map_size;
    int example_size = num_channels * map_size_sq;

    // feature indices (channels, height, width)
    int x_channel = x / map_size_sq;
    int x_f0 = (x % map_size_sq) / map_size;
    int x_f1 = x % map_size;

    int a_before = (x_f0 == 0) ? 0 : ceil(alpha * (x_f0 - 1 + u));
    int a_after = (x_f0 == map_size - 1) ? old_map_size : ceil(alpha * (x_f0 + u));
    int b_before = (x_f1 == 0) ? 0 : ceil(alpha * (x_f1 - 1 + u));
    int b_after = (x_f1 == map_size - 1) ? old_map_size : ceil(alpha * (x_f1 + u));

    int old_map_size_sq = old_map_size * old_map_size;
    int old_example_size = num_channels * old_map_size_sq;
    int input_idx_base = y * old_example_size + old_map_size_sq * x_channel;
    if (x < example_size && y < batch_size) {
      float best = input[input_idx_base  + old_map_size * a_before + b_before];
      for (int a = a_before; a < a_after; a++) {
        for (int b = b_before; b < b_after; b++) {
          best = max(best, input[input_idx_base  + old_map_size * a + b]);
        }
      }
      for (int a = a_before; a < a_after; a++) {
        for (int b = b_before; b < b_after; b++) {
          int old_idx = input_idx_base  + old_map_size * a + b;
          output[old_idx] = (input[old_idx] == best) ? grad[y * example_size + x] : 0;
        }
      }
    }
}
""")
        kernel = mod.get_function("fmp")

        def thunk():
            inp_shape = inputs[0][0].shape
            batch_size, num_channels, height, width = inp_shape
            # might not be necessary, but let's do it anyway
            assert height == width
            new_dim = np.floor(height / self.alpha)
            out_shape = inp_shape
            example_size = num_channels * new_dim * new_dim
            map_size = new_dim

            out = outputs[0]

            # only allocate if there is no previous allocation of the right
            # size.
            if out[0] is None or out[0].shape != out_shape:
                out[0] = cuda.CudaNdarray.zeros(out_shape)

            x_block = 16
            y_block = 16
            block = (x_block, y_block, 1)

            x_grid = int(np.ceil(float(example_size) / x_block))
            y_grid = int(np.ceil(float(batch_size) / y_block))
            grid = (x_grid, y_grid, 1)

            kernel(inputs[0][0],
                   inputs[1][0],
                   out[0],
                   np.float32(self.alpha),
                   np.float32(self.u),
                   np.intc(batch_size),
                   np.intc(num_channels),
                   np.intc(height),
                   np.intc(map_size),
                   block=block,
                   grid=grid)

        thunk.inputs = inputs
        thunk.outputs = outputs
        thunk.lazy = False

        return thunk
