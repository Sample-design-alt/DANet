import time
from functools import reduce
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from einops import rearrange
from torch import nn, einsum
from torch.nn import ModuleList

visual_feature_map = False


# wa_bottom=True
# Prob = False
# mask_bottom = True   #true mask,   false 没有mask
class CyclicShift(nn.Module):
    def __init__(self, displacement):
        super().__init__()
        self.displacement = displacement

    def forward(self, x):
        return torch.roll(x, shifts=self.displacement, dims=1)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


# def create_mask(window_size, displacement, upper_lower, left_right):
def create_mask(window_size, displacement, MASK):
    mask = torch.zeros(window_size, window_size)
    if not MASK:
        mask[-displacement * window_size:, :-displacement * window_size] = float('-inf')
        mask[:-displacement * window_size, -displacement * window_size:] = float('-inf')  # 不带mask

    # 带mask
    else:
        mask[-displacement:, :-displacement] = float('-inf')
        mask[:-displacement, -displacement:] = float('-inf')
    return mask


def get_relative_distances(window_size):
    indices = torch.arange(window_size)
    distances = indices[None, :] - indices[:, None]
    return distances


class WindowAttention(nn.Module):
    def __init__(self, dim, heads, head_dim, shifted, window_size, relative_pos_embedding, wa_dim, wa, prob, mask):
        super().__init__()
        inner_dim = head_dim * heads

        self.heads = heads
        self.scale = head_dim ** -0.5
        self.window_size = window_size
        self.relative_pos_embedding = relative_pos_embedding
        self.shifted = shifted
        self.prob = prob
        self.dropout = nn.Dropout(0.1)

        if self.shifted:
            displacement = window_size // 2
            self.cyclic_shift = CyclicShift(-displacement)
            self.cyclic_back_shift = CyclicShift(displacement)
            self.left_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement, MASK=mask),
                                          requires_grad=False)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        if self.relative_pos_embedding:
            self.relative_indices = get_relative_distances(window_size) + window_size - 1
            self.pos_embedding = nn.Parameter(torch.randn(2 * window_size - 1))
        else:

            self.pos_embedding = nn.Parameter(torch.randn(window_size ** 2, window_size ** 2))

        self.to_out = nn.Linear(inner_dim, dim)

        self.window_attention = nn.Linear(wa_dim, wa_dim)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.activation = nn.ReLU()
        self.wa = wa

    def forward(self, x):
        if self.shifted:
            x = self.cyclic_shift(x)

        # b, n, t,c, h = *x.shape, self.heads
        b, p, f, h = *x.shape, self.heads  # 32,128,96,3     #128代表分了多少段，96是特征
        if p <= self.window_size:
            self.window_size = p
        # b batch_size   p : patch number f: feature

        if self.wa:
            y = self.activation(self.window_attention(self.avgpool(x).reshape(b, p)))
            x = x * y.unsqueeze(2).repeat(1, 1, f)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        new_p = p // self.window_size
        # q, k, v = map(
        #     lambda t: rearrange(t, 'b (nw_h w_h) (nw_w w_w) (h d) -> b h (nw_h nw_w) (w_h w_w) d',
        #                         h=h, w_h=self.window_size, w_w=self.window_size), qkv)

        q, k, v = map(
            lambda t: rearrange(t, 'b (new_w new_p) (head dim) -> b head new_p new_w dim',
                                head=h, new_w=self.window_size), qkv)
        # q  {batch_size,head,patch_num,window_size,feature}
        start_time = time.time()
        if not self.prob:
            dots = einsum('b h w i d, b h w j d -> b h w i j', q, k) * self.scale
            if visual_feature_map:
                import matplotlib.pyplot as plt
                import seaborn as sns
                for i in range(int(dots.shape[1])):
                    sns.heatmap(torch.softmax(dots[0][i][0], dim=-1).cpu().detach().numpy())
                    # plt.savefig('heatmap_{0}.pdf'.format(i), format='pdf')
                    plt.show()

            if self.relative_pos_embedding:
                dots += self.pos_embedding[self.relative_indices[:, :].to(torch.long)]
            else:
                dots += self.pos_embedding

            if self.shifted:
                dots[:, :, -new_p:] += self.left_mask

            attn = self.dropout(dots.softmax(dim=-1))
            out = einsum('b h w i j, b h w j d -> b h w i d', attn, v)
            # print('attention的时间为:',time.time()-start_time)
        else:
            scores_top, index = self._compute_q_k(q, k)
            # print('计算top-K的时间为：',time.time()-start_time)
            scores_top = scores_top * self.scale
            if self.relative_pos_embedding:
                scores_top += self.pos_embedding[self.relative_indices[index].to(torch.long)]
            else:
                scores_top += self.pos_embedding

            if self.shifted:
                scores_top[:, :, -new_p:] += self.left_mask[index]
            context = self._get_initial_context(v, self.window_size)
            out = self._update_context(context, v, scores_top, index)
            # print('prob attention的时间为：',time.time()-start_time)
        out = rearrange(out, 'b head patch window_size dim -> b (patch window_size) (head dim)',
                        head=h)

        out = self.to_out(out)

        if self.shifted:
            out = self.cyclic_back_shift(out)
        return out

    def _compute_q_k(self, q, k):
        B, Head, patch, L_Q, D = q.shape
        _, _, _, L_K, _ = k.shape

        U_part = 5 * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = 5 * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)   #u是采样频率

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(q, k, sample_k=U_part, n_top=u)

        return scores_top, index

    def _prob_QK(self, Q, K, sample_k, n_top):
        B, H, p, L_K, D = K.shape
        _, _, p, L_Q, _ = Q.shape

        K_expand = K.unsqueeze(-3).expand(B, H, p, L_Q, L_K, D)  # B ,H ,P ,L_Q, L_K
        index_sample = torch.randint(L_K, (
            L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q             #从0~96中选出随机L_Q*sample_k个数
        K_sample = K_expand[:, :, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]  # 32 8 96 96 64  -> 32 8 96 25 64
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)
        # 32 8 96 25  = 32 8 96 1 64  * 32 8 96 64 25
        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]
        Q_reduce = Q[torch.arange(B)[:, None, None, None],
                   torch.arange(H)[None, :, None, None],
                   torch.arange(p)[None, None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, p, L_V, D = V.shape

        V_sum = V.mean(dim=-2)
        contex = V_sum.unsqueeze(-2).expand(B, H, p, L_Q, V_sum.shape[-1]).clone()

        return contex

    def _update_context(self, context_in, V, scores, index):
        B, H, P, L_V, D = V.shape

        attn = self.dropout(torch.softmax(scores, dim=-1))  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None, None],
        torch.arange(H)[None, :, None, None],
        torch.arange(P)[None, None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)  # 选取那top-K个的index，不是top-K的用均值代替

        return context_in


class SwinBlock(nn.Module):
    def __init__(self, dim, heads, head_dim, mlp_dim, shifted,
                 window_size, relative_pos_embedding, wa_dim, wa, prob, mask):
        super().__init__()
        self.attention_block = Residual(PreNorm(dim, WindowAttention(dim=dim,
                                                                     heads=heads,
                                                                     head_dim=head_dim,
                                                                     shifted=shifted,
                                                                     window_size=window_size,
                                                                     relative_pos_embedding=relative_pos_embedding,
                                                                     wa_dim=wa_dim,
                                                                     wa=wa,
                                                                     prob=prob,
                                                                     mask=mask,
                                                                     )))
        self.mlp_block = Residual(PreNorm(dim, FeedForward(dim=dim, hidden_dim=mlp_dim)))

    def forward(self, x):
        x = self.attention_block(x)
        x = self.mlp_block(x)
        return x


class PatchMerging(nn.Module):
    def __init__(self, in_channels=1, out_channels=64, downscaling_factor=4):
        super(PatchMerging, self).__init__()
        self.downscaling_factor = downscaling_factor
        self.linear = nn.Linear(in_channels * downscaling_factor, out_channels)

    def forward(self, x):
        x = rearrange(x, 'b (p f) c -> b f (p c)', p=self.downscaling_factor)  # p c 是一个特征   f是代表分为了多少段
        x = self.linear(x)
        return x


class StageModule(nn.Module):
    def __init__(self, in_channels, hidden_dimension, layers, downscaling_factor, num_heads,
                 head_dim, window_size, wa, prob, mask,
                 relative_pos_embedding, wa_dim=4096):
        super().__init__()
        assert layers % 2 == 0, 'Stage layers need to be divisible by 2 for regular and shifted block.'

        self.patch_partition = PatchMerging(in_channels=in_channels, out_channels=hidden_dimension,
                                            downscaling_factor=downscaling_factor)

        self.layers = nn.ModuleList([])
        for _ in range(layers // 2):
            self.layers.append(nn.ModuleList([
                SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=False, window_size=window_size, relative_pos_embedding=relative_pos_embedding,
                          wa_dim=wa_dim, wa=wa, prob=prob, mask=mask),
                SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=True, window_size=window_size, relative_pos_embedding=relative_pos_embedding,
                          wa_dim=wa_dim, wa=wa, prob=prob, mask=mask),
            ]))

    def forward(self, x):
        x = self.patch_partition(x)
        for regular_block, shifted_block in self.layers:
            x = regular_block(x)
            x = shifted_block(x)
        # return x.permute(0, 3, 1, 2)
        return x


class DA_Net(nn.Module):
    def __init__(self, hidden_dim, layers, heads, channels, wa, prob, mask,
                 t=300, down_dim=1024, num_classes=1000, head_dim=32, window_size=7,
                 downscaling_factors=(4, 2, 2, 2), relative_pos_embedding=True):
        '''

        :param hidden_dim: 隐藏层个数
        :param layers: 每一层的block循环次数
        :param heads: head个数
        :param channels: 输入通道
        :param wa: bool 是否启用window attention
        :param prob: bool  是否启动prob attention
        :param mask: bool 是否启用mask
        :param t: 输入的time stamp 长度
        :param down_dim: 嵌入层的降维长度
        :param num_classes: 输出类别
        :param head_dim: 每个head的维度
        :param window_size: 窗口大小
        :param downscaling_factors:每层下采样 倍数
        :param relative_pos_embedding: 是否使用相对位置信息
        '''
        super().__init__()
        self.downsample = nn.Linear(t, down_dim)
        # self.stage1 = StageModule(in_channels=channels, hidden_dimension=hidden_dim[0], layers=layers[0],
        #                           downscaling_factor=downscaling_factors[0], num_heads=heads[0], head_dim=head_dim,
        #                           window_size=window_size, relative_pos_embedding=relative_pos_embedding,
        #                           # embedding_dim=embedding_dim,
        #                           wa_dim = down_dim//downscaling_factors[0],wa=wa,prob=prob,mask=mask)
        # self.stage2 = StageModule(in_channels=hidden_dim[0], hidden_dimension=hidden_dim[1], layers=layers[1],
        #                           downscaling_factor=downscaling_factors[1], num_heads=heads[1], head_dim=head_dim, wa_dim = down_dim//downscaling_factors[0]//downscaling_factors[1],
        #                           window_size=window_size, relative_pos_embedding=relative_pos_embedding,wa=wa,prob=prob,mask=mask)
        # self.stage3 = StageModule(in_channels=hidden_dim[1], hidden_dimension=hidden_dim[2], layers=layers[2],
        #                           downscaling_factor=downscaling_factors[2], num_heads=heads[2], head_dim=head_dim,
        #                           window_size=window_size, relative_pos_embedding=relative_pos_embedding,wa_dim = down_dim//downscaling_factors[0]//downscaling_factors[1]//downscaling_factors[2],wa=wa,prob=prob,mask=mask)
        # self.stage4 = StageModule(in_channels=hidden_dim[2], hidden_dimension=hidden_dim[3], layers=layers[3],
        #                           downscaling_factor=downscaling_factors[3], num_heads=heads[3], head_dim=head_dim,
        #                           window_size=window_size, relative_pos_embedding=relative_pos_embedding,wa_dim = down_dim//downscaling_factors[0]//downscaling_factors[1]//downscaling_factors[2]//downscaling_factors[3],wa=wa,prob=prob,mask=mask)

        self.EncoderList = ModuleList()
        for i in range(len(hidden_dim)):
            layer = StageModule(
                in_channels=channels if i == 0 else hidden_dim[i - 1],
                hidden_dimension=hidden_dim[i],
                layers=layers[i],
                downscaling_factor=downscaling_factors[i],
                num_heads=heads[i],
                head_dim=head_dim,
                window_size=window_size,
                relative_pos_embedding=relative_pos_embedding,
                wa_dim=down_dim // reduce(lambda x, y: x * y, downscaling_factors[:i + 1]),
                wa=wa,
                prob=prob,
                mask=mask
            )
            self.EncoderList.append(layer)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(hidden_dim[-1]),
            nn.Linear(hidden_dim[-1], num_classes)
        )

    def forward(self, ts):
        # ts = ts.transpose(2,1)  #B,C,L  --> B,C,L
        ds = self.downsample(ts)  # B,C,L'
        x = ds.transpose(2, 1)

        for Encoder in self.EncoderList:
            # print('x before',x.shape)
            x = Encoder(x)
            # print('x after',x.shape)
        # x = self.stage1(x)
        # x = self.stage2(x)
        # encoder = self.stage3(x)
        # encoder = self.stage4(x)
        encoder = x
        output = encoder.mean(dim=[1])
        return ds, encoder, output, self.mlp_head(output)
