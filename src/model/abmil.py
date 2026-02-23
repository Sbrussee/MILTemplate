# -----------------------------
# ABMIL
# -----------------------------
class ABMIL(MIL):
    """
    Attention-based Multiple Instance Learning (Ilse et al.)
    Input:  h: (B, M, in_dim)
    Output: logits: (B, num_classes)
    """

    def __init__(
        self,
        in_dim: int = 1024,
        embed_dim: int = 512,
        num_fc_layers: int = 1,
        dropout: float = 0.25,
        attn_dim: int = 384,
        gate: bool = True,
        num_classes: int = 2,
    ):
        super().__init__(in_dim=in_dim, embed_dim=embed_dim, num_classes=num_classes)

        self.patch_embed = create_mlp(
            in_dim=in_dim,
            hid_dims=[embed_dim] * max(0, (num_fc_layers - 1)),
            dropout=dropout,
            out_dim=embed_dim,
            end_with_fc=False,
        )

        attn_cls = GlobalGatedAttention if gate else GlobalAttention
        self.global_attn = attn_cls(L=embed_dim, D=attn_dim, dropout=dropout, num_classes=1)

        if num_classes > 0:
            self.classifier = nn.Linear(embed_dim, num_classes)

        self.initialize_weights()

    def forward_attention(
        self,
        h: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        attn_only: bool = True,
    ):
        """
        Returns:
          if attn_only: A_base (B, K, M) (pre-softmax logits)
          else: (h_embed (B, M, embed_dim), A_base (B, K, M))
        """
        # h: (B,M,in_dim)
        h = self.patch_embed(h)  # (B,M,embed_dim)

        A = self.global_attn(h)  # (B,M,1)
        A = torch.transpose(A, -2, -1)  # (B,1,M)

        if attn_mask is not None:
            # attn_mask: (B,M) float/bool where 1=valid, 0=masked
            attn_mask = attn_mask.to(dtype=A.dtype)
            A = A + (1.0 - attn_mask).unsqueeze(1) * torch.finfo(A.dtype).min

        if attn_only:
            return A
        return h, A

    def forward_features(
        self,
        h: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        return_attention: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Returns:
          slide_feats: (B, embed_dim)
          log_dict: {'attention': A_base or None}
        """
        h, A_base = self.forward_attention(h, attn_mask=attn_mask, attn_only=False)  # h:(B,M,E), A:(B,1,M)
        A = F.softmax(A_base, dim=-1)  # (B,1,M)

        slide_feats = torch.bmm(A, h).squeeze(1)  # (B,E)

        log_dict: Dict[str, Any] = {"attention": A_base if return_attention else None}
        return slide_feats, log_dict

    def forward_head(self, h: torch.Tensor) -> torch.Tensor:
        return self.classifier(h)

    def forward(
        self,
        h: torch.Tensor,
        loss_fn: Optional[nn.Module] = None,
        label: Optional[torch.LongTensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        return_slide_feats: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        slide_feats, log_dict = self.forward_features(h, attn_mask=attn_mask, return_attention=return_attention)
        logits = self.forward_head(slide_feats)

        cls_loss = MIL.compute_loss(loss_fn, logits, label)
        results = {"logits": logits, "loss": cls_loss}

        log_dict["loss"] = float(cls_loss.item()) if cls_loss is not None else -1.0
        if return_slide_feats:
            log_dict["slide_feats"] = slide_feats

        return results, log_dict