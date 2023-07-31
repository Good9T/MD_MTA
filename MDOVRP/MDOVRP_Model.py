import torch
import torch.nn as nn
import torch.nn.functional as F


class MDOVRPModel(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params

        self.encoder = Encoder(**model_params)
        self.decoder = Decoder(**model_params)
        self.encoded_nodes = None
        # shape: (batch_size, node_size, embedding_dim)
        self.depot_size = None
        self.encoded_depots = None
        self.encoded_customers = None

    def pre_forward(self, reset_state):
        depot_x_y = reset_state.depot_x_y
        # shape: (batch_size, depot_size, 2)
        customer_x_y = reset_state.customer_x_y
        # shape: (batch_size, customer_size, 2)
        customer_demand = reset_state.customer_demand
        # shape: (batch_size, customer_size)
        self.depot_size = reset_state.depot_size
        customer_x_y_demand = torch.cat((customer_x_y, customer_demand[:, :, None]), dim=2)
        # shape: (batch_size, customer_size, 3)
        self.encoded_nodes, self.encoded_depots, self.encoded_customers = self.encoder(depot_x_y, customer_x_y_demand)
        # shape: (batch_size, node_size, embedding_dim)
        self.decoder.set_k_v(self.encoded_nodes, self.encoded_depots, self.encoded_customers, reset_state)

    def forward(self, state):
        batch_size = state.batch_idx.size(0)
        mt_size = state.mt_idx.size(1)
        if state.selected_count == 0:  # first move depot
            selected = torch.zeros(size=(batch_size, mt_size), dtype=torch.long)
            probability = torch.ones(size=(batch_size, mt_size))
        elif state.selected_count == 1:  # second move mt
            selected = torch.arange(1, mt_size + 1)[None, :].expand(batch_size, mt_size)
            probability = torch.ones(size=(batch_size, mt_size))
        else:
            prob = self.decoder(self.encoded_nodes, state, mask=state.mask)
            # shape: (batch_size, mt_size, node_size)
            if self.training or self.model_params['sample']:
                while True:
                    with torch.no_grad():
                        selected = prob.reshape(batch_size * mt_size, -1).multinomial(1).squeeze(dim=1) \
                            .reshape(batch_size, mt_size)
                        # shape: (batch_size, mt_size)
                    probability = prob[state.batch_idx, state.mt_idx, selected].reshape(batch_size, mt_size)
                    # shape: (batch_size, mt_size)
                    if (probability != 0).all():
                        break
            else:
                selected = prob.argmax(dim=2)
                # shape: (batch_size, mt_size)
                probability = None
        return selected, probability

class Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.embedding_dim = self.model_params['embedding_dim']
        self.encoder_layer_num = self.model_params['encoder_layer_num']

        self.embedding_depot = nn.Linear(2, self.embedding_dim)
        self.embedding_customer = nn.Linear(3, self.embedding_dim)
        self.layers1 = nn.ModuleList([EncoderLayer(**model_params) for _ in range(self.encoder_layer_num)])
        self.layers2 = nn.ModuleList([EncoderLayer(**model_params) for _ in range(self.encoder_layer_num)])
        self.layers3 = nn.ModuleList([EncoderLayer(**model_params) for _ in range(self.encoder_layer_num)])

    def forward(self, depot_x_y, customer_x_y_demand):
        # depot_x_y shape: (batch_size, depot_size, 2)
        # customer_x_y_demand shape: (batch_size, customer_size, 3)
        embedded_depots = self.embedding_depot(depot_x_y)
        # shape: (batch_size, depot_size, embedding_dim)
        embedded_customers = self.embedding_customer(customer_x_y_demand)
        # shape: (batch_size, customer_size, embedding_dim)
        embedded_nodes = torch.cat((embedded_depots, embedded_customers), dim=1)
        # shape: (batch_size, node_size, embedding_dim)
        for layer in self.layers1:
            embedded_nodes = layer(embedded_nodes)
        for layer in self.layers2:
            embedded_depot = layer(embedded_depots)
        for layer in self.layers3:
            embedded_customers = layer(embedded_customers)

        return embedded_nodes, embedded_depot, embedded_customers
        # shape: (batch_size, node_size, embedding_dim)


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.embedding_dim = self.model_params['embedding_dim']
        self.head_num = self.model_params['head_num']
        self.qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(self.embedding_dim, self.head_num * self.qkv_dim, bias=False)
        self.Wk = nn.Linear(self.embedding_dim, self.head_num * self.qkv_dim, bias=False)
        self.Wv = nn.Linear(self.embedding_dim, self.head_num * self.qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(self.head_num * self.qkv_dim, self.embedding_dim)

        self.norm1 = Norm(**model_params)
        self.ff = FF(**model_params)
        self.norm2 = Norm(**model_params)

    def forward(self, out):
        # shape: (batch_size, node_size, embedding_dim)
        q = multi_head_qkv(self.Wq(out), head_num=self.head_num)
        k = multi_head_qkv(self.Wk(out), head_num=self.head_num)
        v = multi_head_qkv(self.Wv(out), head_num=self.head_num)
        # shape: (batch_size, head_num, node_size, qkv_dim)
        out_concat = multi_head_attention(q, k, v)
        # shape: (batch_size, node_size, head_num * qkv_dim)
        multi_head_out = self.multi_head_combine(out_concat)
        # shape: (batch_size, node_size, embedding_dim)
        out1 = self.norm1(out, multi_head_out)
        out2 = self.ff(out1)
        out3 = self.norm2(out1, out2)
        return out3
        # shape :(batch_size, node_size, embedding_dim)


class Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.embedding_dim = self.model_params['embedding_dim']
        self.head_num = self.model_params['head_num']
        self.qkv_dim = self.model_params['qkv_dim']
        self.clip = self.model_params['clip']
        self.Wq = nn.Linear(self.embedding_dim, self.head_num * self.qkv_dim, bias=False)
        self.Wk = nn.Linear(self.embedding_dim, self.head_num * self.qkv_dim, bias=False)
        self.Wv = nn.Linear(self.embedding_dim, self.head_num * self.qkv_dim, bias=False)
        self.Wk_depot = nn.Linear(self.embedding_dim, self.head_num * self.qkv_dim, bias=False)
        self.Wv_depot = nn.Linear(self.embedding_dim, self.head_num * self.qkv_dim, bias=False)
        self.Wk_customer = nn.Linear(self.embedding_dim, self.head_num * self.qkv_dim, bias=False)
        self.Wv_customer = nn.Linear(self.embedding_dim, self.head_num * self.qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(self.head_num * self.qkv_dim, self.embedding_dim)
        self.k = None
        self.v = None
        self.depots_key = None
        self.add_key = None
        self.nodes_key = None
        self.q = None
        self.k_depot = None
        self.v_depot = None
        self.k_customer = None
        self.v_customer = None
        self.depot_size = None
        self.customer_size = None
        self.node_size = None

    def set_k_v(self, encoded_nodes, encoded_depots, encoded_customers, reset_state):
        self.depot_size = reset_state.depot_size
        self.customer_size = reset_state.customer_size
        self.node_size = self.depot_size + self.customer_size

        self.k = multi_head_qkv(self.Wk(encoded_nodes), head_num=self.head_num)
        self.v = multi_head_qkv(self.Wv(encoded_nodes), head_num=self.head_num)
        # shape: (batch_size, head_num, node_size, qkv_dim)

        self.nodes_key = encoded_nodes.transpose(1, 2)
        # shape: (batch_size, embedding_dim, node_size)

        # encoded_depots = encoded_nodes[:, :self.depot_size, :].contiguous()
        # shape: (batch_size, depot_size, embedding_dim)
        self.k_depot = multi_head_qkv(self.Wk_depot(encoded_depots), head_num=self.head_num)
        self.v_depot = multi_head_qkv(self.Wv_depot(encoded_depots), head_num=self.head_num)
        # shape: (batch_size, head_num, depot_size, qkv_dim)

        self.k_customer = multi_head_qkv(self.Wk_depot(encoded_customers), head_num=self.head_num)
        self.v_customer = multi_head_qkv(self.Wv_depot(encoded_customers), head_num=self.head_num)
        # shape: (batch_size, head_num, customer_size, qkv_dim)

    def forward(self, encoded_nodes, state, mask):
        # mask shape: (batch_size, mt_size, node_size)
        q = get_encoding(encoded_nodes, state)
        # shape: (batch_size, mt_size, embedding_dim)
        self.q = multi_head_qkv(self.Wq(q), head_num=self.head_num)
        # shape: (batch_size, head_num, mt_size, qkv_dim)
        attention_nodes = multi_head_attention(self.q, self.k, self.v, rank3_mask=mask)
        attention_depots = multi_head_attention(self.q, self.k_depot, self.v_depot)
        attention_customers = multi_head_attention(self.q, self.k_customer, self.v_customer)
        attention_combine = attention_nodes + attention_depots + attention_customers
        # shape: (batch_size, mt_size, head_num * qkv_dim)
        score = self.multi_head_combine(attention_combine)
        # shape: (batch_size, mt_size, embedding_dim)
        score_nodes = torch.matmul(score, self.nodes_key)
        # shape: (batch_size, mt_size, node_size)
        sqrt_embedding_dim = self.embedding_dim ** (1 / 2)
        score_scaled = score_nodes / sqrt_embedding_dim
        # shape: (batch_size, mt_size, node_size)
        score_clipped = self.clip * torch.tanh(score_scaled)
        score_masked = score_clipped + mask
        prob = F.softmax(score_masked, dim=2)
        # shape: (batch_size, mt_size, node_size)
        return prob


class Norm(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.embedding_dim = model_params['embedding_dim']
        self.norm = nn.InstanceNorm1d(self.embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input1, input2):
        # shape: (batch_size, node_size, embedding_dim)
        input_added = input1 + input2
        # shape: (batch_size, node_size, embedding_dim)
        input_transposed = input_added.transpose(1, 2)
        # shape: (batch_size, embedding_dim, node_size)
        input_normed = self.norm(input_transposed)
        # shape: (batch_size, embedding_dim, node_size)
        output_transposed = input_normed.transpose(1, 2)
        # shape: (batch_size, node_size, embedding_dim)
        return output_transposed


class FF(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.embedding_dim = model_params['embedding_dim']
        self.ff_hidden_dim = model_params['ff_hidden_dim']
        self.W1 = nn.Linear(self.embedding_dim, self.ff_hidden_dim)
        self.W2 = nn.Linear(self.ff_hidden_dim, self.embedding_dim)

    def forward(self, input1):
        # shape: (batch_size, node_size, embedding_dim)
        return self.W2(F.relu(self.W1(input1)))


def get_encoding(encoded_nodes, state):
    # encoded_customers shape: (batch_size, node_size, embedding_dim)
    # index_to_pick shape: (batch_size, mt_size)
    index_to_pick = state.current_node
    batch_size = index_to_pick.size(0)
    mt_size = index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)
    index_to_gather = index_to_pick[:, :, None].expand(batch_size, mt_size, embedding_dim)
    # shape: (batch_size, mt_size, embedding_dim)
    picked_customers = encoded_nodes.gather(dim=1, index=index_to_gather)
    # shape: (batch_size, mt_size, embedding_dim)
    return picked_customers


def multi_head_qkv(qkv, head_num):
    # shape: (batch_size, n, embedding_dim) : n can be 1 or node_size
    batch_size = qkv.size(0)
    n = qkv.size(1)
    qkv_multi_head = qkv.reshape(batch_size, n, head_num, -1)
    qkv_transposed = qkv_multi_head.transpose(1, 2)
    # shape: (batch_size, head_num, n, key_dim)
    return qkv_transposed


def multi_head_attention(q, k, v, rank2_mask=None, rank3_mask=None):
    # q shape: (batch_size, head_num, n, key_dim)
    # k,v shape: (batch_size, head_num, node_size, key_dim)
    # rank2_mask shape: (batch_size, node_size)
    # rank3_mask shape: (batch_size, group, node_size)
    batch_size = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)
    depot_customer_size = k.size(2)
    score = torch.matmul(q, k.transpose(2, 3))
    # shape :(batch_size, head_num, n, node_size)
    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))
    if rank2_mask is not None:
        score_scaled = score_scaled + rank2_mask[:, None, None, :].expand(batch_size, head_num, n, depot_customer_size)
    if rank3_mask is not None:
        score_scaled = score_scaled + rank3_mask[:, None, :, :].expand(batch_size, head_num, n, depot_customer_size)
    weights = nn.Softmax(dim=3)(score_scaled)
    # shape: (batch_size, head_num, n, node_size)
    out = torch.matmul(weights, v)
    # shape: (batch_size, head_num. n, key_dim)
    out_transposed = out.transpose(1, 2)
    # shape: (batch_size, n, head_num, key_dim)
    out_concat = out_transposed.reshape(batch_size, n, head_num * key_dim)
    # shape: (batch_size, n, head_num * key_dim)
    return out_concat
