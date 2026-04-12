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
    def __init__(
        self,
        task_classes,
        num_joints=33,
        num_steps=16,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
        use_cnn=True,
        use_pose=True,
    ):
        """
        STAEformer for Badminton Stroke Recognition.

        When use_pose=True, use_cnn=True  -> 33 pose joints + 1 CNN node = 34 spatial nodes.
        When use_pose=True, use_cnn=False -> 33 pose joints only (pose-only ablation).
        When use_pose=False -> CNN-only: one spatial node per timestep (requires use_cnn=True).
        """
        super(STAEformerModel, self).__init__()
        if not use_pose and not use_cnn:
            raise ValueError("STAEformerModel: use_pose=False requires use_cnn=True (CNN-only graph).")
        self.use_cnn = use_cnn
        self.use_pose = use_pose
        self.num_joints = num_joints
        if not use_pose:
            self.num_nodes = 1
        elif use_cnn:
            self.num_nodes = num_joints + 1
        else:
            self.num_nodes = num_joints
        self.num_steps = num_steps
        self.embed_dim = embed_dim

        self.joint_proj = nn.Linear(3, embed_dim) if use_pose else None
        self.cnn_proj = nn.Linear(2048, embed_dim) if use_cnn else None

        self.stae_emb = STAE_Embedding(self.num_nodes, num_steps, embed_dim)

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

    def forward(self, joint_seq=None, cnn_seq=None):
        """
        Args:
            joint_seq: (B, T, 33, 3) — required when use_pose=True.
            cnn_seq:   (B, T, 2048) — required when use_cnn=True.
        """
        if self.use_pose:
            assert joint_seq is not None, "joint_seq required when use_pose=True"
            assert self.joint_proj is not None
            B, T = joint_seq.shape[:2]
            e_joints = self.joint_proj(joint_seq)  # (B, T, 33, D)
            if self.use_cnn:
                assert cnn_seq is not None, "cnn_seq required when use_cnn=True"
                assert self.cnn_proj is not None
                e_cnn = self.cnn_proj(cnn_seq).unsqueeze(2)  # (B, T, 1, D)
                x = torch.cat([e_joints, e_cnn], dim=2)  # (B, T, 34, D)
            else:
                x = e_joints
        else:
            assert cnn_seq is not None, "cnn_seq required when use_pose=False"
            assert self.cnn_proj is not None
            B, T = cnn_seq.shape[:2]
            x = self.cnn_proj(cnn_seq).unsqueeze(2)  # (B, T, 1, D)

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
    dummy_joints = torch.randn(2, 16, 33, 3)
    dummy_cnn = torch.randn(2, 16, 2048)

    print("--- use_cnn=True (33 joints + 1 CNN = 34 nodes) ---")
    model = STAEformerModel(task_classes, use_cnn=True)
    out = model(dummy_joints, dummy_cnn)
    for k, v in out.items():
        print(f"  {k}: {v.shape}")

    print("--- use_cnn=False (33 joints, pose-only) ---")
    model_pose = STAEformerModel(task_classes, use_cnn=False)
    out = model_pose(dummy_joints)
    for k, v in out.items():
        print(f"  {k}: {v.shape}")

    print("--- use_pose=False, use_cnn=True (CNN token only) ---")
    model_cnn = STAEformerModel(task_classes, use_cnn=True, use_pose=False)
    out = model_cnn(cnn_seq=dummy_cnn)
    for k, v in out.items():
        print(f"  {k}: {v.shape}")
