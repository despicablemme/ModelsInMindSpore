"""

"""
# todo 某些地方的广播问题
import numpy as np
import mindspore as ms
from mindspore import nn
import mindspore.ops as o
from mindspore import Parameter, Tensor
from mindspore.common.initializer import initializer, TruncatedNormal


class BpskAwgn(nn.Cell):
    def __init__(self, n, r, batch_size):
        super(BpskAwgn, self).__init__()
        self.n = n
        self.r = r
        self.avg_energy = 1
        self.batch_size = batch_size

        self.split = o.Gather()
        self.split_ind_1 = Tensor(np.arange(n), ms.int32)
        self.split_ind_2 = Tensor([n], ms.int32)
        self.bpsk = o.Select()
        self.sqrt = o.Sqrt()
        self.randn = o.StandardNormal()
        self.ones = o.Ones()

    def construct(self, inputs):
        bpsk = self.split(inputs, self.split_ind_1, 1)
        variances = self.split(inputs, self.split_ind_2, 1)
        noise = self.sqrt(variances) * self.randn((self.batch_size, self.n))

        received = bpsk + noise
        init_node_llr = 2 * received / variances

        return init_node_llr  # (bs, n)


class Node2Edges(nn.Cell):
    def __init__(self, llr_mat):
        super(Node2Edges, self).__init__()
        self.llr_mat = llr_mat

        self.matmul = o.MatMul()

    def construct(self, init_node_llr):
        init_edge_llr = self.matmul(init_node_llr, self.llr_mat)
        return init_edge_llr


class ComputeVC(nn.Cell):
    def __init__(self, vc_mat, num_edges):
        super(ComputeVC, self).__init__()
        self.eye = o.Eye()(num_edges, num_edges, ms.float32)
        self.vc_mat = vc_mat

        self.vc_llr_kernel = Parameter(Tensor(np.ones(shape=(num_edges, num_edges)), dtype=ms.float32),
                                       requires_grad=True)
        self.vc_kernel = Parameter(initializer(TruncatedNormal(1.0), shape=(num_edges, num_edges)),
                                   requires_grad=True)

        self.matmul = o.MatMul()
        self.tanh = o.Tanh()
        self.clip_min = Tensor(-10.0, ms.float32)
        self.clip_max = Tensor(10.0, ms.float32)

    def construct(self, init_llr, cv_llr):   # (bs, num), (bs, num)
        llr_param = self.eye * self.vc_llr_kernel
        l_value = self.matmul(init_llr, llr_param)

        vc_param = self.vc_mat * self.vc_kernel
        v_value = self.matmul(cv_llr, vc_param)

        vc_value = v_value + l_value
        vc_value = o.clip_by_value(vc_value, self.clip_min, self.clip_max)
        return self.tanh(0.5 * vc_value)


class ComputeCV(nn.Cell):
    def __init__(self, cv_mat, num_edges, batch_size):
        super(ComputeCV, self).__init__()
        self.cv_mat = cv_mat
        self.num_edges = num_edges
        self.batch_size = batch_size

        self.expand_dim = o.ExpandDims()
        self.tile = o.Tile()
        self.select = o.Select()
        self.prod = o.ReduceProd()
        self.atanh = o.Atanh()
        self.equal = o.Equal()
        self.float = o.Cast()

    def construct(self, vc_value):
        vc_value = self.expand_dim(vc_value, -1)
        vc_value_tiled = self.tile(vc_value, (1, 1, self.num_edges))
        cv_mat_tiled = self.tile(self.expand_dim(self.cv_mat, 0), (self.batch_size, 1, 1))
        value_filter = self.equal(cv_mat_tiled, 1)
        # value_filtered_float = self.float(value_filter, ms.float32)
        inputs_prod = self.select(value_filter, vc_value_tiled, cv_mat_tiled)
        cv_value = self.prod(inputs_prod, 2)
        cv_value = self.atanh(cv_value) * 2

        return cv_value


class Marginalize(nn.Cell):
    def __init__(self, llr_mat_trans, num_edges, n):
        super(Marginalize, self).__init__()
        self.llr_mat_trans = llr_mat_trans

        self.init_kernel = Parameter(Tensor(np.ones((num_edges,)), dtype=ms.float32), requires_grad=True)
        self.cv_kernel = Parameter(Tensor(np.ones((num_edges,)), dtype=ms.float32), requires_grad=True)

        self.matmul = o.MatMul()

    def construct(self, init_edge_llr, cv_llr):
        cv_value = cv_llr * self.cv_kernel
        init_value = init_edge_llr * self.init_kernel
        final_llr = init_value + cv_value
        logits = self.matmul(final_llr, self.llr_mat_trans)

        return logits


class BPRNN(nn.Cell):
    def __init__(self, n, r, num_edges, batch_size, llr_mat, llr_mat_trans, vc_mat, cv_mat, iteration):
        super(BPRNN, self).__init__()
        self.bpskawgn = BpskAwgn(n=n, r=r, batch_size=batch_size)
        self.node2edges = Node2Edges(llr_mat=Tensor(llr_mat, ms.float32))
        self.vc = ComputeVC(vc_mat=Tensor(vc_mat, ms.float32), num_edges=num_edges)
        self.cv = ComputeCV(cv_mat=Tensor(cv_mat, ms.float32), num_edges=num_edges, batch_size=batch_size)
        self.margin = Marginalize(llr_mat_trans=Tensor(llr_mat_trans, ms.float32), num_edges=num_edges, n=n)

        self.iteration = iteration
        self.batch_size = batch_size
        self.n = n

        self.concat = o.Concat(axis=-1)
        self.zeros_like = o.ZerosLike()

    def construct(self, inputs):
        init_node_llr = self.bpskawgn(inputs)
        init_edge_llr = self.node2edges(init_node_llr)

        cv_value = self.zeros_like(init_edge_llr)
        logits_list = []                                     # graph mode 支持列表创建与append
        for iter in range(self.iteration):
            vc_value = self.vc(init_edge_llr, cv_value)
            cv_value = self.cv(vc_value)
            logits = self.margin(init_edge_llr, cv_value)
            logits_list.append(logits)
        logits_ = self.concat(logits_list)
        return -logits_
