class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        # Stelle sicher, dass alpha den richtigen Typ hat
        if alpha is not None:
            self.alpha = alpha.float() if isinstance(alpha, torch.Tensor) else torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Stelle sicher, dass die Eingaben den richtigen Typ haben
        inputs = inputs.float()  # Wandle in float32 um
        targets = targets.long()  # Wandle in long um
        
        B, C, H, W = inputs.size()
        
        # Reshape inputs and targets
        inputs = inputs.permute(0, 2, 3, 1).contiguous().view(-1, C)
        targets = targets.view(-1)
        
        # Stelle sicher, dass alpha zu float konvertiert wird und auf demselben Gerät liegt
        weight = None
        if self.alpha is not None:
            weight = self.alpha.to(inputs.device).float()
        
        # Berechne Cross-Entropy-Loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=weight)
        
        # Berechne Wahrscheinlichkeiten
        probs = F.softmax(inputs, dim=1)
        probs_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Wende Focal-Gewichtung an
        focal_weight = (1 - probs_t) ** self.gamma
        focal_loss = focal_weight * ce_loss
        
        # Wende reduction an
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss


# Combined loss function for training
def combined_loss_function(outputs, pre_masks, post_masks):
    pre_outputs = outputs[:, :2]  # First 2 channels
    post_outputs = outputs[:, 2:]  # Remaining channels
    
    # Calculate focal losses
    pre_loss = focal_loss_pre(pre_outputs, pre_masks)
    post_loss = focal_loss_post(post_outputs, post_masks)
    
    # Combine losses (you can adjust weights)
    total_loss = pre_loss + post_loss
    return total_loss