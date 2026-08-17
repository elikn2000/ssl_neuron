# Attention and Block adapted from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
# DINO adapted from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/dino.py

import copy
import torch
import numpy as np
import torch.nn as nn
from typing import Any
from ssl_neuron.PV_Space import PVFC, PVManifoldMLR,PV_layer_norm, PVManifold, PVFC_2, PVMLR_2, _pv_dist

class GraphAttention(nn.Module):
    """ Implements GraphAttention.

    Graph Attention interpolates global transformer attention
    (all nodes attend to all other nodes based on their
    dot product similarity) and message passing (nodes attend
    to their 1-order neighbour based on dot-product
    attention).

    Attributes:
        dim: Dimensionality of key, query and value vectors.
        num_heads: Number of parallel attention heads.
        bias: If set to `True`, use bias in input projection layers.
          Default is `False`.
        use_exp: If set to `True`, use the exponential of the predicted
          weights to trade-off global and local attention.
    """
    def __init__(self,
                 
                 dim: int,
                 num_heads: int = 8,
                 bias: bool = False,
                 use_exp: bool = True) -> nn.Module:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.scale = dim ** -0.5
        self.use_exp = use_exp
        self.qkv_projection = nn.Linear(dim, dim * num_heads * 3, bias=bias)
        self.proj = nn.Linear(dim * num_heads, dim)
        # Weigth to trade of local vs. global attention.
        self.predict_gamma = nn.Linear(dim, 2)
        # Initialize projection such that gamma is close to 1
        # in the beginning of training.
        self.predict_gamma.weight.data.uniform_(0.0, 0.01)

        
    @torch.jit.script
    def fused_mul_add(a, b, c, d):
        return (a * b) + (c * d)

    def forward(self, x, adj):
        
        B, N, C = x.shape # (batch x num_nodes x feat_dim) isn't it more like (batch x num_nodes x dim)?
        
        qkv = self.qkv_projection(x).view(B, N, 3, self.num_heads, self.dim).permute(0, 3, 1, 2, 4)
        query, key, value = qkv.unbind(dim=3) # (batch x num_heads x num_nodes x dim)
        attn = (query @ key.transpose(-2, -1)) * self.scale # (batch x num_heads x num_nodes x num_nodes)
        # Predict trade-off weight per node
        gamma = self.predict_gamma(x)[:, None].repeat(1, self.num_heads, 1, 1)
        if self.use_exp:
            
            # Parameterize gamma to always be positive
            gamma = torch.exp(gamma)

        adj = adj[:, None].repeat(1, self.num_heads, 1, 1)

        # Compute trade-off between local and global attention.
        attn = self.fused_mul_add(gamma[:, :, :, 0:1], attn, gamma[:, :, :, 1:2], adj)
        
        attn = attn.softmax(dim=-1)

        x = (attn @ value).transpose(1, 2).reshape(B, N, -1) # (batch_size x num_nodes x (num_heads * dim))
        return self.proj(x)
class HyperbolicGraphAttention(nn.Module):
    """ Implements GraphAttention.

    Graph Attention interpolates global transformer attention
    (all nodes attend to all other nodes based on their
    dot product similarity) and message passing (nodes attend
    to their 1-order neighbour based on dot-product
    attention).

    Attributes:
        dim: Dimensionality of key, query and value vectors.
        num_heads: Number of parallel attention heads.
        bias: If set to `True`, use bias in input projection layers.
          Default is `False`.
        use_exp: If set to `True`, use the exponential of the predicted
          weights to trade-off global and local attention.
    """
    def __init__(self,
                 
                 dim: int,
                 num_heads: int = 8,
                 bias: bool = False,
                 k: float = -1.0,
                 use_exp: bool = True,
                 hyperbolic_linear: str = 'gyro',
                 attention_scores: str = 'squaredist',
                 value_accredition: str = 'standard'
                 ) -> nn.Module:
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.scale = dim ** -0.5
        self.k = k
        self.manifold = PVManifold(k=k)
        self.use_exp = use_exp
        if hyperbolic_linear == 'gyro':
            self.qkv_projection = PVFC_2(k=self.k, in_features=dim, out_features=dim * num_heads * 3, use_bias=bias, act=None)
            self.proj = PVFC_2(k=self.k, in_features=dim * num_heads, out_features=dim, use_bias=bias, act=None)
        if hyperbolic_linear == 'planes':
            self.qkv_projection = PVFC(k=self.k, in_features=dim, out_features=dim * num_heads * 3, use_bias=bias, act=None)
            self.proj = PVFC(k=self.k, in_features=dim * num_heads, out_features=dim, use_bias=bias, act=None)
        self.attention_scores=attention_scores
        self.value_accredition=value_accredition
        # Weigth to trade of local vs. global attention.
        self.predict_gamma = nn.Linear(dim, 2)
        # Initialize projection such that gamma is close to 1
        # in the beginning of training.
        self.predict_gamma.weight.data.uniform_(0.0, 0.01)

        
    @torch.jit.script
    def fused_mul_add(a, b, c, d):
        return (a * b) + (c * d)

    def forward(self, x, adj):
        
        B, N, C = x.shape # (batch x num_nodes x feat_dim) isn't it more like (batch x num_nodes x dim)?
        
        qkv = self.qkv_projection(x).view(B, N, 3, self.num_heads, self.dim).permute(0, 3, 1, 2, 4)
        query, key, value = qkv.unbind(dim=3) # (batch x num_heads x num_nodes x dim)
        if self.attention_scores == "squaredist":
            attn = -self.manifold.dist(query, key)**2 * self.scale # (batch x num_heads x num_nodes x num_nodes)
        elif self.attention_scores == "Lorentzian_product":
            attn = -self.manifold.Lorentz_prod(query, key) * self.scale # (batch x num_heads x num_nodes x num_nodes)
        else:
            raise ValueError(f"Unsupported attention_scores '{self.attention_scores}'. Valid options are: ['squaredist', 'Lorentzian_product']")
        # Predict trade-off weight per node
        gamma = self.predict_gamma(x)[:, None].repeat(1, self.num_heads, 1, 1)
        if self.use_exp:
            
            # Parameterize gamma to always be positive
            gamma = torch.exp(gamma)

        adj = adj[:, None].repeat(1, self.num_heads, 1, 1)

        # Compute trade-off between local and global attention.
        attn = self.fused_mul_add(gamma[:, :, :, 0:1], attn, gamma[:, :, :, 1:2], adj)
        
        attn = attn.softmax(dim=-1)
        if self.value_accredition == "standard":
            x= (attn @ value).transpose(1, 2).reshape(B, N, -1) # (batch_size x num_nodes x (num_heads * dim))
        elif self.value_accredition == "Lorentzian_centroid":
            vals=attn @ value
            norms=torch.clamp(self.manifold.Lorentz_norm(vals), min=1e-12)
            x = self.manifold.s *  (vals/norms).transpose(1, 2).reshape(B, N, -1) # (batch_size x num_nodes x (num_heads * dim))
        else:
            raise ValueError(f"Unsupported valueaccredition '{self.value_accredition}'. Valid options are: ['standard', 'Lorentzian_centroid']")
        return self.proj(x)
    
class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> nn.Module:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)
class Hyp_MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, k: float, hyperbolic_linear: str) -> nn.Module:
        super().__init__()
        if hyperbolic_linear == 'gyro':
            self.net = nn.Sequential(
            PVFC_2(in_features=dim, out_features=hidden_dim, k=k, use_bias=True, act='gelu'),
            PVFC_2(in_features=hidden_dim, out_features=dim, k=k, use_bias=True, act='none'),
            )
        if hyperbolic_linear == 'planes':
            self.net = nn.Sequential(
            PVFC(in_features=dim, out_features=hidden_dim, k=k, use_bias=True, act='gelu'),
            PVFC(in_features=hidden_dim, out_features=dim, k=k, use_bias=True, act='none'),
            )
    def forward(self, x):
        return self.net(x)


class AttentionBlock(nn.Module):
    """ Implements an attention block.
    """
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 mlp_ratio: int = 4,
                 bias: bool = False,
                 use_exp: bool = True,
                 norm_layer: Any = nn.LayerNorm) -> nn.Module:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(dim=dim, hidden_dim=dim * mlp_ratio)
        self.attn = GraphAttention( dim=dim, num_heads=num_heads, bias=bias, use_exp=use_exp )
        

    def forward(self, x, a):
        x = self.norm1(x)
        x = x + self.attn(x, a)
        x = self.norm2(x)
        x = x + self.mlp(x)
        return x
class HyperbolicAttentionBlock(nn.Module):
    """ Implements an attention block.
    """
    def __init__(self,
                 k: float,
                 dim: int,
                 num_heads: int,
                 mlp_ratio: int = 4,
                 bias: bool = False,
                 use_exp: bool = True,
                 hyperbolic_linear: str = 'gyro',
                 attention_scores: str = 'squaredist',
                 value_accredition: str = 'standard',
                 rescon: str = 'weighted_geodesic_midpoint'
                 ) -> nn.Module:
        super().__init__()
        
        self.k=k
        self.manifold = PVManifold(k=k)
        self.rescon=rescon
        if self.rescon =='weighted_geodesic_midpoint':
            self.res_scale_att = nn.Parameter(torch.tensor(0.0))
            self.res_scale_mlp = nn.Parameter(torch.tensor(0.0))
            self.delta = nn.Parameter(torch.tensor(float(1.5)))
        elif self.rescon =='weighted_gyroaddition':
            self.res_scale_att = nn.Parameter(torch.tensor(1.0))
            self.res_scale_mlp = nn.Parameter(torch.tensor(1.0))
            self.delta = nn.Parameter(torch.tensor(float(1.0)))
        else: 
            raise ValueError(f"Unsupported residual connection method '{self.rescon}'. Valid options are: ['weighted_geodesic_midpoint', 'weighted_gyroaddition']")
        self.norm1 = PV_layer_norm(k=k, dimension=dim)
        self.norm2 = PV_layer_norm(k=k, dimension=dim)
        self.mlp = Hyp_MLP(dim=dim, hidden_dim=dim * mlp_ratio, k=k, hyperbolic_linear=hyperbolic_linear)
        self.attn = HyperbolicGraphAttention(k=k, dim=dim, num_heads=num_heads, bias=bias, use_exp=use_exp, hyperbolic_linear=hyperbolic_linear, attention_scores=attention_scores, value_accredition=value_accredition)
        

    def forward(self, x, a):
        if self.rescon== 'weighted_geodesic_midpoint':
            residual = x
            x = self.norm1(x)
            x =  self.manifold.Residual(residual, self.attn(x, a), alpha=torch.sigmoid(self.res_scale_att))
            x=self.manifold.gyro_scalar_mul( self.delta, x)
            residual = x
            x = self.norm2(x)
            x = self.manifold.Residual(residual , self.mlp(x), alpha=torch.sigmoid(self.res_scale_mlp))
            x=self.manifold.gyro_scalar_mul(self.delta, x)

        
        elif self.rescon == 'weighted_gyroaddition':
            residual= x
            x=self.norm1(x)
            x=self.manifold.gyro_add(residual,self.manifold.gyro_scalar_mul(self.res_scale_att, self.attn(x, a)) )
            x=self.manifold.gyro_scalar_mul(self.delta, x)
            residual=x
            x=self.norm2(x)

            x=self.manifold.gyro_add(residual, self.manifold.gyro_scalar_mul(self.res_scale_mlp,self.mlp(x)))
            x=self.manifold.gyro_scalar_mul(self.delta, x)
        else:
            raise ValueError(f"Unsupported residual connection method '{self.rescon}'. Valid options are: ['weighted_geodesic_midpoint', 'weighted_gyroaddition']")
        return x
    
    
class GraphTransformer(nn.Module):
    def __init__(self,
                 n_nodes: int = 200,
                 dim: int = 32,
                 hyp_depth: int = 0,
                 euc_depth: int = 5,
                 num_heads: int = 8,
                 mlp_ratio: int = 2,
                 feat_dim: int = 8,
                 num_classes: int = 1000,
                 pos_dim: int = 32,
                 proj_dim: int = 128,
                 num_proj_layers: int = 3,
                 use_exp: bool = True,
                 hyperbolic_Projection: bool = False,
                 hyperbolic_linear: str = 'planes',
                 attention_scores: str = 'squaredist',
                 rescon: str = 'weighted_geodesic_midpoint',
                 value_accredition: str = 'standard',
                 loc_embedding: int = 0,
                 k: float = -1.0,
                 loss_function: str = 'cross_entropy') -> nn.Module:
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.cls_pos_embedding = nn.Parameter(torch.randn(1, 1, dim))
        if hyperbolic_linear == 'gyro':
            self.hyperbolic_linear=PVFC_2
            self.hyperbolic_classifier=PVMLR_2
        if hyperbolic_linear == 'planes':
            self.hyperbolic_linear=PVFC
            self.hyperbolic_classifier=PVManifoldMLR
        self.k=k
        self.blocks = nn.Sequential(*[
            AttentionBlock( dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, use_exp=use_exp)
            for i in range(euc_depth)], *[HyperbolicAttentionBlock(k=self.k, dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, use_exp=use_exp, hyperbolic_linear=hyperbolic_linear, attention_scores=attention_scores, value_accredition=value_accredition, rescon=rescon)
            for i in range(hyp_depth)])

        self.to_pos_embedding = nn.Linear(pos_dim, dim)

        if hyp_depth>0:

            self.mlp_head = nn.Sequential(
                PV_layer_norm(k=self.k, dimension=dim),
                self.hyperbolic_linear(in_features=dim, out_features=dim, k=self.k, use_bias=True)
                )
        else:
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim)
            )
        if hyperbolic_Projection is True:
            projector_layers=[]
            if num_proj_layers==0:
                proj_dim=dim
            else:
                projector_layers.append(self.hyperbolic_linear(in_features=dim,  out_features=proj_dim, k=self.k, use_bias=True, act='gelu'))
                for _ in range(num_proj_layers-1):
                    projector_layers.append(self.hyperbolic_linear(in_features=proj_dim,  out_features=proj_dim, k=self.k, use_bias=True, act='gelu'))
            projector_layers.append(PV_layer_norm(k=self.k, dimension=proj_dim))
            
            if loss_function == 'cross_entropy':
                projector_layers.append(self.hyperbolic_classifier(in_features=proj_dim,  num_classes=num_classes, k=self.k))
            if loss_function == 'cross_entropy_contrastive':
                projector_layers.append(self.hyperbolic_classifier(in_features=proj_dim,  num_classes=num_classes, k=self.k))
            if loss_function == 'hyperbolic':
                projector_layers.append(self.hyperbolic_linear(in_features=proj_dim,  out_features=num_classes, k=self.k, use_bias=True, act='gelu'))
            
            self.projector=nn.Sequential(*projector_layers)
                
            
       
        else:
            projector_layers=[]
            if num_proj_layers==0:
                proj_dim=dim
            else:
                projector_layers.append(nn.Linear(dim, proj_dim))
                projector_layers.append(nn.GELU())
                for _ in range(num_proj_layers-1):
                    projector_layers.append(nn.Linear(proj_dim, proj_dim))
                    projector_layers.append(nn.GELU())
               
            projector_layers.append(nn.LayerNorm(proj_dim))
            projector_layers.append(nn.Linear(proj_dim, num_classes))
            self.projector=nn.Sequential(*projector_layers)

        self.to_node_embedding = nn.Sequential(
            nn.Linear(feat_dim, dim * 2),
            nn.ReLU(True),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, node_feat, adj, lapl, loc_embedding):
        B, N, _ = node_feat.shape

        # Compute initial node embedding.
        x = self.to_node_embedding(node_feat)

        # Compute positional encoding
        pos_embedding_token = self.to_pos_embedding(lapl)

        # Add "classification" token
        cls_pos_enc = self.cls_pos_embedding.repeat(B, 1, 1)
        pos_embedding = torch.cat((cls_pos_enc, pos_embedding_token), dim=1)

        cls_tokens = self.cls_token.repeat(B, 1, 1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add classification token entry to adjanceny matrix. 
        adj_cls = torch.zeros(B, N + 1, N + 1, device=node_feat.device)
        # TODO(test if useful)
        adj_cls[:, 0, 0] = 1.
        adj_cls[:, 1:, 1:] = adj

        x += pos_embedding

        for block in self.blocks:
            x = block(x, adj_cls)
        x = x[:, 0]
      
        x = self.mlp_head(x)
        
        if loc_embedding>0:
            x=self.projector[:loc_embedding](x)
            y=self.projector[loc_embedding:](x)
        else:
            y=self.projector(x)
        return x, y


class ExponentialMovingAverage():
    """ Exponential moving average.

    Attributes:
        decay: Moving average decay parameter in [0., 1.] (float).
    """
    def __init__(self, decay: float):
        super().__init__()
        self.decay = decay
        assert (decay > 0.) and (decay < 1.), 'Decay must be in [0., 1.]'

    def update_average(
        self,
        previous_state: torch.Tensor,
        update: torch.Tensor,
        decay: float = None,
    ):
        if previous_state is None:
            return update
        if decay is not None:
            return previous_state * decay + (1 - decay) * update
        else:
            return previous_state * self.decay + (1 - self.decay) * update


def update_moving_average(ema_updater, teacher_model, student_model, decay=None):
    for student_params, teacher_params in zip(student_model.parameters(), teacher_model.parameters()):
        teacher_weights, weight_update = teacher_params.data, student_params.data
        teacher_params.data = ema_updater.update_average(teacher_weights, weight_update, decay=decay)

class GraphDINO(nn.Module):
    def __init__(
        self,
        transformer: nn.Module,
        num_classes: int = 1000,
        student_temp: float = 0.9,
        teacher_temp: float = 0.06,
        moving_average_decay: float = 0.999,
        center_moving_average_decay: float = 0.9,
        loss_function: str = 'cross_entropy', 
        contrastive_loss_factor: float = 0.25
    ):
        super().__init__()
        self.student_encoder = transformer
        self.teacher_encoder = copy.deepcopy(self.student_encoder)
        # Weights of teacher model are updated using an exponential moving
        # average of the student model. Thus, disable gradient update.
        for p in self.teacher_encoder.parameters():
            p.requires_grad = False

        self.teacher_ema_updater = ExponentialMovingAverage(moving_average_decay)
        
        self.register_buffer('teacher_centers', torch.zeros(1, num_classes))
        self.register_buffer('previous_centers',  torch.zeros(1, num_classes))

        self.teacher_centering_ema_updater = ExponentialMovingAverage(center_moving_average_decay)

        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.loss_function = loss_function.lower()

        valid_loss_functions = {'euclidean', 'hyperbolic', 'cross_entropy', 'cross_entropy_contrastive'}
        if self.loss_function not in valid_loss_functions:
            raise ValueError(
                f"Unsupported loss_function '{loss_function}'. "
                f"Valid options are: {sorted(valid_loss_functions)}"
            )
        if self.loss_function == 'euclidean':
            self.loss_fn = self.euclidean_loss
        elif self.loss_function == 'hyperbolic':
            self.loss_fn = self.hyperbolic_loss
        elif self.loss_function == 'cross_entropy':
            self.loss_fn = self.cross_entropy
        elif self.loss_function == 'cross_entropy_contrastive':
            self.loss_fn = self.cross_entropy_contrastive
        self.contrastive_loss_factor=contrastive_loss_factor
        
    def cross_entropy_contrastive(self, teacher_logits, student_logits, eps = 1e-20):
        T = teacher_logits.detach() - self.teacher_centers
        S = student_logits

        student_probs = (S / self.student_temp).softmax(dim=-1)
        teacher_probs = (T / self.teacher_temp).softmax(dim=-1)
        batch_size=teacher_probs.shape[0]
        T = torch.nn.functional.normalize(T, p=2, dim=1)
        S = torch.nn.functional.normalize(S, p=2, dim=1)

        
        loss = -(teacher_probs * torch.log(student_probs + eps)).sum(dim=-1).mean()

        mask = torch.eye(T.size(0), device=T.device, dtype=torch.bool)
        teacher_scores = torch.cat([
        ((T @ T.T) / self.teacher_temp**2).masked_fill(mask, float("-inf")),
        ((T @ S.T) / (self.teacher_temp*self.student_temp))
        ], dim=1)
        
        teacher_den = torch.logsumexp(teacher_scores, dim=1)
        student_scores = torch.cat([
                ((S @ S.T) / self.student_temp**2).masked_fill(mask, float("-inf")),
                ((S @ T.T) / (self.teacher_temp*self.student_temp))
                ], dim=1)
        
        student_den = torch.logsumexp(student_scores, dim=1)
        positive = 2 * torch.diagonal(T @ S.T).sum()/(self.teacher_temp * self.student_temp)
        

        contrastive_loss = - ( positive - teacher_den.sum(dim=0) - student_den.sum(dim=0) )/batch_size**2
        loss += self.contrastive_loss_factor * contrastive_loss

        return loss
    def cross_entropy(self, teacher_logits, student_logits, eps = 1e-20):
        teacher_logits = teacher_logits.detach()
        student_probs = (student_logits / self.student_temp).softmax(dim = -1)
        teacher_probs = ((teacher_logits - self.teacher_centers) / self.teacher_temp).softmax(dim = -1)
        loss = - (teacher_probs * torch.log(student_probs + eps)).sum(dim = -1).mean()
        return loss
    def euclidean_loss(self, teacher_logits, student_logits):
        teacher_logits=teacher_logits.detach()
        loss = (teacher_logits - student_logits).pow(2).sum(dim = -1).mean()
        return loss
    def hyperbolic_loss(self, teacher_logits, student_logits, k = -1):
        # 1. Detach and avoid inplace issues
        teacher_logits = teacher_logits.detach()
        loss=_pv_dist(teacher_logits, student_logits, k=k, neg_k=-k, s=1.0 / np.sqrt(-k), tiny=1e-15).mean()
        # # 2. Secure the norms
        # # Ensure the value inside sqrt is strictly positive
        # t_norm_sq = torch.norm(teacher_logits, dim=-1)**2
        # s_norm_sq = torch.norm(student_logits, dim=-1)**2

        # t_sqrt = torch.sqrt(torch.clamp(t_norm_sq - 1/k, min=1e-7))
        # s_sqrt = torch.sqrt(torch.clamp(s_norm_sq - 1/k, min=1e-7))

        # # 3. Calculate the dot product
        # dot_prod = (teacher_logits * student_logits).sum(dim=-1)

        # # 4. Clamp the acosh input 
        # # Using a slightly larger min (1 + 1e-5) prevents the 1/sqrt(0) gradient problem
        # acosh_input = torch.clamp(K * (dot_prod - t_sqrt * s_sqrt), min=1 + 1e-5)

        # # 5. Final Loss
        # loss = (1 / np.sqrt(abs(K))) * torch.acosh(acosh_input).mean()
        # # teacher_logits=teacher_logits.detach()
        # # loss=1/np.sqrt(abs(K))*torch.acosh(torch.clamp(K*((teacher_logits * student_logits).sum(dim=-1)-torch.sqrt(torch.norm(teacher_logits, dim=-1)**2-1/K)*torch.sqrt(torch.norm(student_logits, dim=-1)**2-1/K)),min=1+eps)).mean()
        return loss
    def update_moving_average(self, decay=None):
        update_moving_average(self.teacher_ema_updater, self.teacher_encoder, self.student_encoder, decay=decay)

        new_teacher_centers = self.teacher_centering_ema_updater.update_average(self.teacher_centers, self.previous_centers)
        self.teacher_centers.copy_(new_teacher_centers)

    def forward(self, node_feat1, node_feat2, adj1, adj2, lapl1, lapl2, loc_embedding):
        batch_size = node_feat1.shape[0]

        # Concatenate the two views to compute embeddings as one batch.
        node_feat = torch.cat([node_feat1, node_feat2], dim=0)
        adj = torch.cat([adj1, adj2], dim=0)
        lapl = torch.cat([lapl1, lapl2], dim=0)

        _, student_proj = self.student_encoder(node_feat, adj, lapl, loc_embedding)
        student_proj1, student_proj2 = torch.split(student_proj, batch_size, dim=0)

        with torch.no_grad():
            teacher_logits, teacher_proj = self.teacher_encoder(node_feat, adj, lapl, loc_embedding)
            teacher_proj1, teacher_proj2 = torch.split(teacher_proj, batch_size, dim=0)
        
        teacher_logits_avg = teacher_proj.mean(dim = 0)
        teacher_logits_avgnorm=torch.norm(teacher_logits, dim=-1).mean(dim=0)
        self.previous_centers.copy_(teacher_logits_avg)
        max_val=torch.norm(teacher_logits, dim=-1).max()
        

        loss1 = self.loss_fn(teacher_proj1, student_proj2)
        loss2 = self.loss_fn(teacher_proj2, student_proj1)
        loss = (loss1 + loss2) / 2
        return loss, max_val, teacher_logits_avgnorm


def create_model(config):
    num_classes = config['model']['num_classes']

    # Create encoder.
    transformer = GraphTransformer(n_nodes=config['data']['n_nodes'],
                 dim=config['model']['dim'], 
                 hyp_depth=config['model']['hyp_depth'], 
                 euc_depth=config['model']['euc_depth'],
                 num_heads=config['model']['n_head'],
                 feat_dim=config['data']['feat_dim'],
                 pos_dim=config['model']['pos_dim'],
                 proj_dim=config['model']['proj_dim'],
                 num_classes=num_classes,
                 hyperbolic_Projection=config['model']['hyperbolic_Projection'],
                 hyperbolic_linear=config['model']['hyperbolic_linear'],
                 attention_scores=config['model']['attention_scores'],
                 value_accredition=config['model']['value_accredition'],
                 rescon=config['model']['rescon'],
                 num_proj_layers=config['model']['num_proj_layers'],
                 loc_embedding=config['testing']['loc_embedding'],
                 loss_function=config['model']['loss_function'],
                 k=config['model']['curvature'],
                 )

    # Create GraphDINO.
    model = GraphDINO(
        transformer,
        num_classes=num_classes,
        moving_average_decay=config['model']['move_avg'],
        center_moving_average_decay=config['model']['center_avg'],
        teacher_temp=config['model']['teacher_temp'],
        loss_function=config['model']['loss_function'],
        contrastive_loss_factor=config['model']['contrastive_loss_factor']
    )
    
    return model