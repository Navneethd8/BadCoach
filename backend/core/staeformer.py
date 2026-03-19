import torch
import torch.nn as nn
import torch.nn.functional as F

class STAE_Embedding(nn.Module):
    def __init__(self, num_nodes, num_steps, embed_dim):
        super(STAE_Embedding, self).__init__()
        self.spatial_emb = nn.Parameter(torch.zeros(1, num_nodes, embed_dim))
        self.temporal_emb = nn.Parameter(torch.zeros(num_steps, 1, embed_dim))
        
        nn.init.xavier_uniform_(self.spatial_emb)
        nn.init.xavier_uniform_(self.temporal_emb)

    def forward(self, x):
        # x: (B, T, N, D)
        return x + self.spatial_emb + self.temporal_emb

class STAEformerModel(nn.Module):
    def __init__(self, task_classes, num_nodes=34, num_steps=16, embed_dim=128, num_heads=4, num_layers=2, dropout=0.1):
        """
        STAEformer implementation for Badminton Stroke Recognition.
        Nodes: 33 pose joints + 1 global CNN feature = 34 nodes.
        Steps: 16 frames.
        """
        super(STAEformerModel, self).__init__()
        self.num_nodes = num_nodes
        self.num_steps = num_steps
        self.embed_dim = embed_dim

        # 1. Feature Projection
        # 33 joints (x, y, z)
        self.joint_proj = nn.Linear(3, embed_dim)
        # 1 CNN global feature (2048-dim)
        self.cnn_proj = nn.Linear(2048, embed_dim)

        # 2. Spatio-Temporal Adaptive Embedding
        self.stae_emb = STAE_Embedding(num_nodes, num_steps, embed_dim)

        # 3. Temporal Transformer Layers (pre-norm for stability)
        encoder_layer_temp = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer_temp, num_layers=num_layers)

        # 4. Spatial Transformer Layers (pre-norm for stability)
        encoder_layer_spat = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.spatial_transformer = nn.TransformerEncoder(encoder_layer_spat, num_layers=num_layers)

        # 5. Multi-Task Heads (input: 2*embed_dim from temporal avg+max pool)
        self.heads = nn.ModuleDict({
            task: nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, num_c)
            )
            for task, num_c in task_classes.items()
        })

    def forward(self, joint_seq, cnn_seq):
        """
        Args:
            joint_seq: (B, T, 33, 3)
            cnn_seq: (B, T, 2048)
        """
        B, T, N_j, D_j = joint_seq.shape
        _, _, D_c = cnn_seq.shape

        # 1. Project to common embed_dim
        e_joints = self.joint_proj(joint_seq) # (B, T, 33, D)
        e_cnn = self.cnn_proj(cnn_seq).unsqueeze(2) # (B, T, 1, D)
        
        # Combine into (B, T, 34, D)
        x = torch.cat([e_joints, e_cnn], dim=2)

        # 2. Add STAE
        x = self.stae_emb(x)

        # 3. Temporal Attention (B*N, T, D) with residual
        identity = x
        x_temp = x.permute(0, 2, 1, 3).reshape(B * self.num_nodes, T, self.embed_dim)
        x_temp = self.temporal_transformer(x_temp)
        x_temp = x_temp.reshape(B, self.num_nodes, T, self.embed_dim).permute(0, 2, 1, 3)
        x = identity + x_temp

        # 4. Spatial Attention (B*T, N, D) with residual
        identity = x
        x_spat = x.reshape(B * T, self.num_nodes, self.embed_dim)
        x_spat = self.spatial_transformer(x_spat)
        x_spat = x_spat.reshape(B, T, self.num_nodes, self.embed_dim)
        x = identity + x_spat

        # 5. Global Pooling: pool nodes first, then avg+max over time (same pattern as CNN+LSTM)
        x = torch.mean(x, dim=2)  # (B, T, D)
        avg_pool = torch.mean(x, dim=1)   # (B, D)
        max_pool, _ = torch.max(x, dim=1)  # (B, D)
        final_feature = torch.cat([avg_pool, max_pool], dim=1)  # (B, 2*D)

        # 6. Task Heads
        logits = {task: head(final_feature) for task, head in self.heads.items()}
        return logits

if __name__ == "__main__":
    task_classes = {"stroke_type": 9, "position": 10}
    model = STAEformerModel(task_classes)
    
    dummy_joints = torch.randn(2, 16, 33, 3)
    dummy_cnn = torch.randn(2, 16, 2048)
    
    out = model(dummy_joints, dummy_cnn)
    for k, v in out.items():
        print(f"{k}: {v.shape}")
