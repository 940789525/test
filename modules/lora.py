import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from typing import Optional, Tuple

class LoraInjectedLinear(nn.Module):
    """
    A linear layer with LoRA injected.
    This class now intelligently adapts to the dtype of the original module.
    """
    def __init__(self, original_module: nn.Linear, r=4):
        super().__init__()
        in_features = original_module.in_features
        out_features = original_module.out_features

        if r > min(in_features, out_features):
            raise ValueError(f"LoRA rank {r} must be less or equal than {min(in_features, out_features)}")
        
        self.linear = original_module
        
        # 获取原始模块的dtype (e.g., torch.float16 or torch.float32)
        self.dtype = original_module.weight.dtype
        
        self.lora_down = nn.Linear(in_features, r, bias=False)
        self.lora_up = nn.Linear(r, out_features, bias=False)
        self.scale = 1.0
        
        # 将新创建的LoRA层转换为与原始模块相同的dtype
        self.lora_down.to(device=original_module.weight.device, dtype=self.dtype)
        self.lora_up.to(device=original_module.weight.device, dtype=self.dtype)
        
        nn.init.normal_(self.lora_down.weight, std=1 / r)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x):
        # The forward pass now works seamlessly as all dtypes match.
        return self.linear(x) + self.lora_up(self.lora_down(x)) * self.scale

class LoraInjectedMultiheadAttention(nn.Module):
    """
    A wrapper for nn.MultiheadAttention to inject LoRA into Q, K, V projections.
    This class now intelligently adapts to the dtype of the original module.
    """
    def __init__(self, mha_module: nn.MultiheadAttention, r=4):
        super().__init__()
        if r > mha_module.embed_dim:
            raise ValueError(f"LoRA rank {r} must be less or equal than embed_dim {mha_module.embed_dim}")

        self.mha = mha_module
        self.r = r
        self.scale = 1.0
        
        d_model = mha_module.embed_dim
        # 获取原始模块的dtype和device
        self.dtype = mha_module.in_proj_weight.dtype
        device = mha_module.in_proj_weight.device

        self.q_lora_down = nn.Linear(d_model, r, bias=False)
        self.q_lora_up = nn.Linear(r, d_model, bias=False)
        self.k_lora_down = nn.Linear(d_model, r, bias=False)
        self.k_lora_up = nn.Linear(r, d_model, bias=False)
        self.v_lora_down = nn.Linear(d_model, r, bias=False)
        self.v_lora_up = nn.Linear(r, d_model, bias=False)

        # 将新创建的LoRA层全部转换为与原始模块相同的dtype和device
        self.to(device=device, dtype=self.dtype)

        nn.init.normal_(self.q_lora_down.weight, std=1 / r)
        nn.init.zeros_(self.q_lora_up.weight)
        nn.init.normal_(self.k_lora_down.weight, std=1 / r)
        nn.init.zeros_(self.k_lora_up.weight)
        nn.init.normal_(self.v_lora_down.weight, std=1 / r)
        nn.init.zeros_(self.v_lora_up.weight)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None,
                need_weights: bool = True, attn_mask: Optional[torch.Tensor] = None,
                average_attn_weights: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        q_proj, k_proj, v_proj = F.linear(query, self.mha.in_proj_weight, self.mha.in_proj_bias).chunk(3, dim=-1)

        q = q_proj + self.q_lora_up(self.q_lora_down(query)) * self.scale
        k = k_proj + self.k_lora_up(self.k_lora_down(key)) * self.scale
        v = v_proj + self.v_lora_up(self.v_lora_down(value)) * self.scale

        attn_output, attn_output_weights = F.multi_head_attention_forward(
            query=q, key=k, value=v,
            embed_dim_to_check=self.mha.embed_dim, num_heads=self.mha.num_heads,
            in_proj_weight=torch.empty(0), in_proj_bias=None,
            bias_k=self.mha.bias_k, bias_v=self.mha.bias_v,
            add_zero_attn=self.mha.add_zero_attn, dropout_p=self.mha.dropout,
            out_proj_weight=self.mha.out_proj.weight, out_proj_bias=self.mha.out_proj.bias,
            training=self.training, key_padding_mask=key_padding_mask,
            need_weights=need_weights, attn_mask=attn_mask, average_attn_weights=average_attn_weights
        )
        return attn_output, attn_output_weights

def inject_lora(model: nn.Module, r: int = 8):
    """
    Traverses the model and replaces target layers with their LoRA-injected counterparts.
    Includes a fallback for older PyTorch versions that lack the get_submodule method.
    """
    def get_parent_module(model, path):
        parent_path = '.'.join(path.split('.')[:-1])
        if not parent_path: return model
        parent = model
        for part in parent_path.split('.'):
            parent = getattr(parent, part)
        return parent

    for name, module in model.named_modules():
        if isinstance(module, nn.MultiheadAttention):
            parent_module = get_parent_module(model, name)
            module_name = name.split('.')[-1]
            # 传递整个原始模块
            setattr(parent_module, module_name, LoraInjectedMultiheadAttention(module, r=r))
            print(f"Injected LoRA into MultiheadAttention: {name}")

        if re.search(r'vision_model.encoder.layers.\d+.mlp.fc(1|2)', name) or \
           re.search(r'text_model.encoder.layers.\d+.mlp.fc(1|2)', name):
            
            if isinstance(module, nn.Linear):
                parent_module = get_parent_module(model, name)
                module_name = name.split('.')[-1]
                # 传递整个原始模块
                setattr(parent_module, module_name, LoraInjectedLinear(module, r=r))
                print(f"Injected LoRA into Linear layer: {name}")

# extract_lora 函数保持不变
def extract_lora(model: nn.Module):
    for name, module in model.named_modules():
        parent_name = '.'.join(name.split('.')[:-1])
        if not parent_name:
            continue
        module_name = name.split('.')[-1]
        
        def get_parent_module(model, path):
            parent = model
            for part in path.split('.'):
                parent = getattr(parent, part)
            return parent
        
        parent_module = get_parent_module(model, parent_name)

        if isinstance(module, LoraInjectedLinear):
            in_features = module.linear.in_features
            out_features = module.linear.out_features
            bias = module.linear.bias is not None
            
            new_linear = nn.Linear(in_features, out_features, bias=bias)
            
            merged_weight = module.linear.weight.data + \
                            (module.lora_up.weight.data @ module.lora_down.weight.data) * module.scale
            new_linear.weight.data.copy_(merged_weight)
            if bias:
                new_linear.bias.data.copy_(module.linear.bias.data)
            
            setattr(parent_module, module_name, new_linear)
            print(f"Extracted and merged LoRA from Linear: {name}")

        elif isinstance(module, LoraInjectedMultiheadAttention):
            d_model = module.mha.embed_dim
            
            new_mha = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=module.mha.num_heads,
                dropout=module.mha.dropout,
                bias=module.mha.in_proj_bias is not None,
                batch_first=module.mha.batch_first,
            )
            
            q_delta = module.q_lora_up.weight.data @ module.q_lora_down.weight.data
            k_delta = module.k_lora_up.weight.data @ module.k_lora_down.weight.data
            v_delta = module.v_lora_up.weight.data @ module.v_lora_down.weight.data
            
            Wq, Wk, Wv = module.mha.in_proj_weight.data.chunk(3)
            
            Wq_new = Wq + (q_delta * module.scale).T
            Wk_new = Wk + (k_delta * module.scale).T
            Wv_new = Wv + (v_delta * module.scale).T
            
            new_mha.in_proj_weight.data.copy_(torch.cat([Wq_new, Wk_new, Wv_new]))
            
            if module.mha.in_proj_bias is not None:
                new_mha.in_proj_bias.data.copy_(module.mha.in_proj_bias.data)
            new_mha.out_proj.weight.data.copy_(module.mha.out_proj.weight.data)
            if module.mha.out_proj.bias is not None:
                new_mha.out_proj.bias.data.copy_(module.mha.out_proj.bias.data)
                
            setattr(parent_module, module_name, new_mha)
            print(f"Extracted and merged LoRA from MultiheadAttention: {name}")