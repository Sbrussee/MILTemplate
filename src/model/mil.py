# -----------------------------
# MIL base class
# -----------------------------
class MIL(ABC, nn.Module):
    """
    Abstract base class for MIL models.
    """

    def __init__(self, in_dim: int, embed_dim: int, num_classes: int):
        super().__init__()
        self.in_dim = int(in_dim)
        self.embed_dim = int(embed_dim)
        self.num_classes = int(num_classes)

    @abstractmethod
    def forward_attention(self, h: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, attn_only: bool = True):
        raise NotImplementedError

    @abstractmethod
    def forward_features(
        self, h: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, return_attention: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def forward_head(self, h: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        h: torch.Tensor,
        loss_fn: Optional[nn.Module] = None,
        label: Optional[torch.LongTensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        return_slide_feats: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        raise NotImplementedError

    @staticmethod
    def compute_loss(loss_fn: Optional[nn.Module], logits: Optional[torch.Tensor], label: Optional[torch.LongTensor]):
        if loss_fn is None or logits is None or label is None:
            return None
        return loss_fn(logits, label)

    def initialize_weights(self):
        """
        Kaiming for Linear, Xavier for Conv2d, sensible defaults for norms.
        """
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.LayerNorm):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.BatchNorm1d):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)

    def initialize_classifier(self, num_classes: Optional[int] = None):
        if num_classes is None:
            num_classes = self.num_classes
        self.classifier = nn.Linear(self.embed_dim, int(num_classes))
        nn.init.kaiming_uniform_(self.classifier.weight, nonlinearity="relu")
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)