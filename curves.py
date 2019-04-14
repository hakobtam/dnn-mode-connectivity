import numpy as np
import math
import warnings
import itertools
import numbers
import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter
from torch.nn.modules.utils import _pair
from torch.nn.modules.utils.rnn import PackedSequence
from scipy.special import binom


class Bezier(Module):
    def __init__(self, num_bends):
        super(Bezier, self).__init__()
        self.register_buffer(
            'binom',
            torch.Tensor(binom(num_bends - 1, np.arange(num_bends), dtype=np.float32))
        )
        self.register_buffer('range', torch.arange(0, float(num_bends)))
        self.register_buffer('rev_range', torch.arange(float(num_bends - 1), -1, -1))

    def forward(self, t):
        return self.binom * \
               torch.pow(t, self.range) * \
               torch.pow((1.0 - t), self.rev_range)


class PolyChain(Module):
    def __init__(self, num_bends):
        super(PolyChain, self).__init__()
        self.num_bends = num_bends
        self.register_buffer('range', torch.arange(0, float(num_bends)))

    def forward(self, t):
        t_n = t * (self.num_bends - 1)
        return torch.max(self.range.new([0.0]), 1.0 - torch.abs(t_n - self.range))


class CurveModule(Module):

    def __init__(self, fix_points, parameter_names=()):
        super(CurveModule, self).__init__()
        self.fix_points = fix_points
        self.num_bends = len(self.fix_points)
        self.parameter_names = parameter_names
        self.l2 = 0.0

    def compute_weights_t(self, coeffs_t):
        w_t = [None] * len(self.parameter_names)
        self.l2 = 0.0
        for i, parameter_name in enumerate(self.parameter_names):
            for j, coeff in enumerate(coeffs_t):
                parameter = getattr(self, '%s_%d' % (parameter_name, j))
                if parameter is not None:
                    if w_t[i] is None:
                        w_t[i] = parameter * coeff
                    else:
                        w_t[i] += parameter * coeff
            if w_t[i] is not None:
                self.l2 += torch.sum(w_t[i] ** 2)
        return w_t


class Linear(CurveModule):

    def __init__(self, in_features, out_features, fix_points, bias=True):
        super(Linear, self).__init__(fix_points, ('weight', 'bias'))
        self.in_features = in_features
        self.out_features = out_features

        self.l2 = 0.0
        for i, fixed in enumerate(self.fix_points):
            self.register_parameter(
                'weight_%d' % i,
                Parameter(torch.Tensor(out_features, in_features), requires_grad=not fixed)
            )
        for i, fixed in enumerate(self.fix_points):
            if bias:
                self.register_parameter(
                    'bias_%d' % i,
                    Parameter(torch.Tensor(out_features), requires_grad=not fixed)
                )
            else:
                self.register_parameter('bias_%d' % i, None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.in_features)
        for i in range(self.num_bends):
            getattr(self, 'weight_%d' % i).data.uniform_(-stdv, stdv)
            bias = getattr(self, 'bias_%d' % i)
            if bias is not None:
                bias.data.uniform_(-stdv, stdv)

    def forward(self, input, coeffs_t):
        weight_t, bias_t = self.compute_weights_t(coeffs_t)
        return F.linear(input, weight_t, bias_t)


class Conv2d(CurveModule):

    def __init__(self, in_channels, out_channels, kernel_size, fix_points, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d, self).__init__(fix_points, ('weight', 'bias'))
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        for i, fixed in enumerate(self.fix_points):
            self.register_parameter(
                'weight_%d' % i,
                Parameter(
                    torch.Tensor(out_channels, in_channels // groups, *kernel_size),
                    requires_grad=not fixed
                )
            )
        for i, fixed in enumerate(self.fix_points):
            if bias:
                self.register_parameter(
                    'bias_%d' % i,
                    Parameter(torch.Tensor(out_channels), requires_grad=not fixed)
                )
            else:
                self.register_parameter('bias_%d' % i, None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        for i in range(self.num_bends):
            getattr(self, 'weight_%d' % i).data.uniform_(-stdv, stdv)
            bias = getattr(self, 'bias_%d' % i)
            if bias is not None:
                bias.data.uniform_(-stdv, stdv)

    def forward(self, input, coeffs_t):
        weight_t, bias_t = self.compute_weights_t(coeffs_t)
        return F.conv2d(input, weight_t, bias_t, self.stride,
                        self.padding, self.dilation, self.groups)


class _BatchNorm(CurveModule):
    _version = 2

    def __init__(self, num_features, fix_points, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_BatchNorm, self).__init__(fix_points, ('weight', 'bias'))
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        self.l2 = 0.0
        for i, fixed in enumerate(self.fix_points):
            if self.affine:
                self.register_parameter(
                    'weight_%d' % i,
                    Parameter(torch.Tensor(num_features), requires_grad=not fixed)
                )
            else:
                self.register_parameter('weight_%d' % i, None)
        for i, fixed in enumerate(self.fix_points):
            if self.affine:
                self.register_parameter(
                    'bias_%d' % i,
                    Parameter(torch.Tensor(num_features), requires_grad=not fixed)
                )
            else:
                self.register_parameter('bias_%d' % i, None)

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            for i in range(self.num_bends):
                getattr(self, 'weight_%d' % i).data.uniform_()
                getattr(self, 'bias_%d' % i).data.zero_()

    def _check_input_dim(self, input):
        raise NotImplementedError

    def forward(self, input, coeffs_t):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum
        weight_t, bias_t = self.compute_weights_t(coeffs_t)
        return F.batch_norm(
            input, self.running_mean, self.running_var, weight_t, bias_t,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = metadata.get('version', None)

        if (version is None or version < 2) and self.track_running_stats:
            # at version 2: added num_batches_tracked buffer
            #               this should have a default value of 0
            num_batches_tracked_key = prefix + 'num_batches_tracked'
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)

        super(_BatchNorm, self)._load_from_state_dict(
            state_dict, prefix, metadata, strict,
            missing_keys, unexpected_keys, error_msgs)


class BatchNorm2d(_BatchNorm):

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))


_VF = torch._C._VariableFunctions
_rnn_impls = {
    'LSTM': _VF.lstm,
    'GRU': _VF.gru,
    'RNN_TANH': _VF.rnn_tanh,
    'RNN_RELU': _VF.rnn_relu,
}


class LSTM(CurveModule):

    def __init__(self, input_size, hidden_size, fix_points,
                 num_layers=1, bias=True, batch_first=False,
                 dropout=0, bidirectional=False):

        num_directions = 2 if bidirectional else 1
        all_param_names = []
        for layer in range(num_layers):
            for direction in range(num_directions):
                suffix = '_reverse' if direction == 1 else ''
                param_names = ['weight_ih_l{}{}', 'weight_hh_l{}{}']
                if bias:
                    param_names += ['bias_ih_l{}{}', 'bias_hh_l{}{}']
                param_names = [x.format(layer, suffix) for x in param_names]
                all_param_names += param_names
        super(LSTM, self).__init__(fix_points, tuple(all_param_names))

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional

        if not isinstance(dropout, numbers.Number) or not 0 <= dropout <= 1 or \
                isinstance(dropout, bool):
            raise ValueError("dropout should be a number in range [0, 1] "
                             "representing the probability of an element being "
                             "zeroed")
        if dropout > 0 and num_layers == 1:
            warnings.warn("dropout option adds dropout after all but last "
                          "recurrent layer, so non-zero dropout expects "
                          "num_layers greater than 1, but got dropout={} and "
                          "num_layers={}".format(dropout, num_layers))

        gate_size = 4 * hidden_size
        self._all_weights = []
        for layer in range(num_layers):
            for direction in range(num_directions):
                layer_input_size = input_size if layer == 0 else hidden_size * num_directions

                suffix = '_reverse' if direction == 1 else ''
                param_names = []

                for i, fixed in enumerate(self.fix_points):
                    self.register_parameter(
                        'weight_ih_l{}{}_{}'.format(layer, suffix, i),
                        Parameter(torch.Tensor(gate_size, layer_input_size), requires_grad=not fixed)
                    )
                    param_names += ['weight_ih_l{}{}_{}'.format(layer, suffix, i)]
                for i, fixed in enumerate(self.fix_points):
                    self.register_parameter(
                        'weight_hh_l{}{}_{}'.format(layer, suffix, i),
                        Parameter(torch.Tensor(gate_size, hidden_size), requires_grad=not fixed)
                    )
                    param_names += ['weight_hh_l{}{}_{}'.format(layer, suffix, i)]
                if bias:
                    for i, fixed in enumerate(self.fix_points):
                        self.register_parameter(
                            'bias_ih_l{}{}_{}'.format(layer, suffix, i),
                            Parameter(torch.Tensor(gate_size), requires_grad=not fixed)
                        )
                        param_names += ['bias_ih_l{}{}_{}'.format(layer, suffix, i)]
                    for i, fixed in enumerate(self.fix_points):
                        self.register_parameter(
                            'bias_hh_l{}{}_{}'.format(layer, suffix, i),
                            Parameter(torch.Tensor(gate_size), requires_grad=not fixed)
                        )
                        param_names += ['bias_hh_l{}{}_{}'.format(layer, suffix, i)]
                    # else:
                    #     self.register_parameter('bias_ih_l{}{}_{}'.format(layer, suffix, i), None)
                    #     self.register_parameter('bias_hh_l{}{}_{}'.format(layer, suffix, i), None)

                self._all_weights.append(param_names)

        self.flatten_parameters()
        self.reset_parameters()

    def flatten_parameters(self):
        """Resets parameter data pointer so that they can use faster code paths.

        Right now, this works only if the module is on the GPU and cuDNN is enabled.
        Otherwise, it's a no-op.
        """
        any_param = next(self.parameters()).data
        if not any_param.is_cuda or not torch.backends.cudnn.is_acceptable(any_param):
            return

        # If any parameters alias, we fall back to the slower, copying code path. This is
        # a sufficient check, because overlapping parameter buffers that don't completely
        # alias would break the assumptions of the uniqueness check in
        # Module.named_parameters().
        all_weights = self._flat_weights
        unique_data_ptrs = set(p.data_ptr() for p in all_weights)
        if len(unique_data_ptrs) != len(all_weights):
            return

        with torch.cuda.device_of(any_param):
            import torch.backends.cudnn.rnn as rnn

            # NB: This is a temporary hack while we still don't have Tensor
            # bindings for ATen functions
            with torch.no_grad():
                # NB: this is an INPLACE function on all_weights, that's why the
                # no_grad() is necessary.
                torch._cudnn_rnn_flatten_weight(
                    all_weights, (4 * self.num_bends if self.bias else 2 * self.num_bends),
                    self.input_size, rnn.get_cudnn_mode('LSTM'), self.hidden_size, self.num_layers,
                    self.batch_first, bool(self.bidirectional))

    def _apply(self, fn):
        ret = super(LSTM, self)._apply(fn)
        self.flatten_parameters()
        return ret

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def check_forward_args(self, input, hidden, batch_sizes):
        is_input_packed = batch_sizes is not None
        expected_input_dim = 2 if is_input_packed else 3
        if input.dim() != expected_input_dim:
            raise RuntimeError(
                'input must have {} dimensions, got {}'.format(
                    expected_input_dim, input.dim()))
        if self.input_size != input.size(-1):
            raise RuntimeError(
                'input.size(-1) must be equal to input_size. Expected {}, got {}'.format(
                    self.input_size, input.size(-1)))

        if is_input_packed:
            mini_batch = int(batch_sizes[0])
        else:
            mini_batch = input.size(0) if self.batch_first else input.size(1)

        num_directions = 2 if self.bidirectional else 1
        expected_hidden_size = (self.num_layers * num_directions,
                                mini_batch, self.hidden_size)

        def check_hidden_size(hx, expected_hidden_size, msg='Expected hidden size {}, got {}'):
            if tuple(hx.size()) != expected_hidden_size:
                raise RuntimeError(msg.format(expected_hidden_size, tuple(hx.size())))

        check_hidden_size(hidden[0], expected_hidden_size,
                          'Expected hidden[0] size {}, got {}')
        check_hidden_size(hidden[1], expected_hidden_size,
                          'Expected hidden[1] size {}, got {}')

    def forward(self, input, coeffs_t, hx=None):
        is_packed = isinstance(input, PackedSequence)
        if is_packed:
            input, batch_sizes = input
            max_batch_size = int(batch_sizes[0])
        else:
            batch_sizes = None
            max_batch_size = input.size(0) if self.batch_first else input.size(1)

        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            hx = input.new_zeros(self.num_layers * num_directions,
                                 max_batch_size, self.hidden_size,
                                 requires_grad=False)
            hx = (hx, hx)

        self.check_forward_args(input, hx, batch_sizes)
        _impl = _rnn_impls['LSTM']
        if batch_sizes is None:
            result = _impl(input, hx, self.compute_weights_t(coeffs_t), self.bias, self.num_layers,
                           self.dropout, self.training, self.bidirectional, self.batch_first)
        else:
            result = _impl(input, batch_sizes, hx, self.compute_weights_t(coeffs_t), self.bias,
                           self.num_layers, self.dropout, self.training, self.bidirectional)
        output = result[0]
        hidden = result[1:]

        if is_packed:
            output = PackedSequence(output, batch_sizes)
        return output, hidden

    def extra_repr(self):
        s = '{input_size}, {hidden_size}'
        if self.num_layers != 1:
            s += ', num_layers={num_layers}'
        if self.bias is not True:
            s += ', bias={bias}'
        if self.batch_first is not False:
            s += ', batch_first={batch_first}'
        if self.dropout != 0:
            s += ', dropout={dropout}'
        if self.bidirectional is not False:
            s += ', bidirectional={bidirectional}'
        return s.format(**self.__dict__)

    def __setstate__(self, d):
        super(LSTM, self).__setstate__(d)
        if 'all_weights' in d:
            self._all_weights = d['all_weights']
        if isinstance(self._all_weights[0][0], str):
            return
        num_layers = self.num_layers
        num_directions = 2 if self.bidirectional else 1
        self._all_weights = []
        for layer in range(num_layers):
            for direction in range(num_directions):
                suffix = '_reverse' if direction == 1 else ''
                param_names = []

                for i in range(len(self.fix_points)):
                    param_names += ['weight_ih_l{}{}_{}'.format(layer, suffix, i)]
                for i in range(len(self.fix_points)):
                    param_names += ['weight_hh_l{}{}_{}'.format(layer, suffix, i)]
                if bias:
                    for i in range(len(self.fix_points)):
                        param_names += ['bias_ih_l{}{}_{}'.format(layer, suffix, i)]
                    for i in range(len(self.fix_points)):
                        param_names += ['bias_hh_l{}{}_{}'.format(layer, suffix, i)]
                self._all_weights.append(param_names)
    @property
    def _flat_weights(self):
        return [p for layerparams in self.all_weights for p in layerparams]

    @property
    def all_weights(self):
        return [[getattr(self, weight) for weight in weights] for weights in self._all_weights]



class CurveNet(Module):
    def __init__(self, num_classes, curve, architecture, num_bends, fix_start=True, fix_end=True,
                 architecture_kwargs={}):
        super(CurveNet, self).__init__()
        self.num_classes = num_classes
        self.num_bends = num_bends
        self.fix_points = [fix_start] + [False] * (self.num_bends - 2) + [fix_end]

        self.curve = curve
        self.architecture = architecture

        self.l2 = 0.0
        self.coeff_layer = self.curve(self.num_bends)
        self.net = self.architecture(num_classes, fix_points=self.fix_points, **architecture_kwargs)
        self.curve_modules = []
        for module in self.net.modules():
            if issubclass(module.__class__, CurveModule):
                self.curve_modules.append(module)

    def import_base_parameters(self, base_model, index):
        parameters = list(self.net.parameters())[index::self.num_bends]
        base_parameters = base_model.parameters()
        for parameter, base_parameter in zip(parameters, base_parameters):
            parameter.data.copy_(base_parameter.data)

    def import_base_buffers(self, base_model):
        for buffer, base_buffer in zip(self.net._all_buffers(), base_model._all_buffers()):
            buffer.data.copy_(base_buffer.data)

    def export_base_parameters(self, base_model, index):
        parameters = list(self.net.parameters())[index::self.num_bends]
        base_parameters = base_model.parameters()
        for parameter, base_parameter in zip(parameters, base_parameters):
            base_parameter.data.copy_(parameter.data)

    def init_linear(self):
        parameters = list(self.net.parameters())
        for i in range(0, len(parameters), self.num_bends):
            weights = parameters[i:i+self.num_bends]
            for j in range(1, self.num_bends - 1):
                alpha = j * 1.0 / (self.num_bends - 1)
                weights[j].data.copy_(alpha * weights[-1].data + (1.0 - alpha) * weights[0].data)

    def weights(self, t):
        coeffs_t = self.coeff_layer(t)
        weights = []
        for module in self.curve_modules:
            weights.extend([w for w in module.compute_weights_t(coeffs_t) if w is not None])
        return np.concatenate([w.detach().cpu().numpy().ravel() for w in weights])

    def _compute_l2(self):
        self.l2 = sum(module.l2 for module in self.curve_modules)

    def forward(self, input, t=None):
        if t is None:
            t = input.data.new(1).uniform_()
        coeffs_t = self.coeff_layer(t)
        output = self.net(input, coeffs_t)
        self._compute_l2()
        return output


def l2_regularizer(weight_decay):
    return lambda model: 0.5 * weight_decay * model.l2
