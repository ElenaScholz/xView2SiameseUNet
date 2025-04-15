class EarlyStopping:
    def __init__(self, patience=5, delta=0.01, verbose=False, checkpoint_dir=None, 
                 experiment_group=None, experiment_id=None):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.checkpoint_dir = checkpoint_dir
        self.experiment_group = experiment_group
        self.experiment_id = experiment_id
        self.best_loss = None
        self.no_improvement_count = 0
        self.stop_training = False
        self.best_checkpoint_path = None
        
    def check_early_stop(self, val_loss, epoch, model, optimizer=None, scheduler=None, writer=None):
        # Initialize best_loss if it's the first check
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, epoch, model, optimizer, scheduler, writer)
        # If validation loss improved by at least delta
        elif val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.no_improvement_count = 0
            self.save_checkpoint(val_loss, epoch, model, optimizer, scheduler, writer)
            if self.verbose:
                print(f'Validation loss decreased to {val_loss:.4f}. Saving checkpoint...')
        # If validation loss did not improve
        else:
            self.no_improvement_count += 1
            if self.verbose:
                print(f'No improvement in validation loss for {self.no_improvement_count} epochs.')
            if self.no_improvement_count >= self.patience:
                self.stop_training = True
                if self.verbose:
                    print(f'Early stopping triggered after {self.patience} epochs without improvement.')
        
        return self.stop_training
    
    def save_checkpoint(self, val_loss, epoch, model, optimizer=None, scheduler=None, writer=None):
        """Save complete checkpoint when validation loss decreases."""
        if self.checkpoint_dir:
            checkpoint_path = self.checkpoint_dir / f'{self.experiment_group}_{self.experiment_id}_epoch_{epoch+1}.pth'
            
            try:
                checkpoint = {
                    'epoch': epoch+1,
                    'model_state_dict': model.state_dict(),
                    'loss': val_loss,
                }
                
                # Add optimizer and scheduler states if available
                if optimizer is not None:
                    checkpoint['optimizer_state_dict'] = optimizer.state_dict()
                if scheduler is not None:
                    checkpoint['scheduler_state_dict'] = scheduler.state_dict()
                
                torch.save(checkpoint, checkpoint_path)
                
                self.best_checkpoint_path = checkpoint_path
                
                if writer is not None:
                    best_checkpoint_path_str = f"val_loss:{val_loss:.4f}@{checkpoint_path}"
                    writer.add_text("Best Checkpoint Path", best_checkpoint_path_str, epoch)
                
                if self.verbose:
                    print(f'Successfully saved checkpoint at {checkpoint_path}')
                    
            except Exception as e:
                print(f"Error saving checkpoint: {e}")