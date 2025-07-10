import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from model.SelfAttention_Family import AttentionLayer


plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args, model):
    warmup_epochs = 0
    lr = args.learning_rate
    if epoch <= warmup_epochs:
        # Linear warm-up: gradually increase learning rate
        lr = args.learning_rate * (epoch / warmup_epochs)
    else:
        # Apply different learning rate schedules
        if args.lradj == 'type1':
            lr = args.learning_rate * (0.5 ** ((epoch - warmup_epochs) // 1))
        elif args.lradj == 'type7':
            lr = args.learning_rate * (0.7 ** ((epoch - warmup_epochs) // 1))
        elif args.lradj == 'type6':
            lr = args.learning_rate * (0.6 ** ((epoch - warmup_epochs) // 1))
        elif args.lradj == 'type2':
            # Specific learning rate adjustment for certain epochs
            lr_schedule = {
                2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
                10: 5e-7, 15: 1e-7, 20: 5e-8
            }
            if epoch in lr_schedule:
                lr = lr_schedule[epoch]
    # else:
    #     if args.lradj == 'type1':
    #         lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - warmup_epochs) // 1))}
    #     if args.lradj == 'type7':
    #         lr_adjust = {epoch: args.learning_rate * (0.7 ** ((epoch - warmup_epochs) // 1))}
    #     if args.lradj == 'type6':
    #         lr_adjust = {epoch: args.learning_rate * (0.6 ** ((epoch - warmup_epochs) // 1))}
    #     elif args.lradj == 'type2':
    #         lr_adjust = {
    #             2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
    #             10: 5e-7, 15: 1e-7, 20: 5e-8
    #         }
    if hasattr(model, 'prompt'):
        prompt_lr = lr * 0.1 
        for name, param in model.named_parameters():
            if 'prompt' in name and param.requires_grad:  # Check if `prompt` is trainable
                for param_group in optimizer.param_groups:
                    # if param in param_group['params']:
                    for param in param_group['params']:
                        if any(param is p for p in param_group['params']):
                            param_group['lr'] = prompt_lr
                print(f"Adjusting learning rate for 'prompt' layer to {prompt_lr}")
  
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print('Updating learning rate to {}'.format(lr))

def load_state_dict_with_prefix(model, checkpoint_path, device):
        state_dict = torch.load(checkpoint_path, map_location=device)

        is_model_wrapped = isinstance(model, torch.nn.DataParallel)
        has_module_prefix = any(k.startswith("module.") for k in state_dict.keys())

        if is_model_wrapped and not has_module_prefix:
            state_dict = {"module." + k: v for k, v in state_dict.items()}
        elif not is_model_wrapped and has_module_prefix:
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded trainable parameters from {checkpoint_path}")
        if missing_keys:
            print(f" Missing keys (not found in checkpoint): {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys (not used by model): {unexpected_keys}")


    
def freeze_attention_modules(model):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    for name, module in model.patch_encoder.named_modules():
        if 'attention' in name and isinstance(module, nn.Module):
            for param in module.parameters():
                param.requires_grad = False

    for name, module in model.patch_fusion.named_modules():
        if isinstance(module, nn.MultiheadAttention):  
            for param in module.parameters():
                param.requires_grad = False
    print("All relevant attention layers frozen.")

class EarlyStopping:
    def __init__(self, patience=1, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        if isinstance(model, torch.nn.DataParallel):
            model = model.module  
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if np.all(gt[i] == 1) and np.all(pred[i] == 1) and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if np.all(gt[j] == 0):
                    break
                else:
                    if np.all(pred[j] == 0):
                        pred[j] = 1
            for j in range(i, len(gt)):
                if np.all(gt[j] == 0):
                    break
                else:
                    if np.all(pred[j] == 0):
                        pred[j] = 1
        elif np.all(gt[i] == 0):
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred

def save_model(model, optimizer, epoch, loss, checkpoint_path):
    """
    Save the model state_dict, prompt layer, optimizer, and training info.
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'prompt': model.prompt,  # 保存 prompt 层
    }, checkpoint_path)
    print(f"Model and prompt saved at {checkpoint_path}")

def load_model(model, optimizer, checkpoint_path):
    """
    Load the model state_dict, prompt layer, optimizer, and training info.
    """
    checkpoint = torch.load(checkpoint_path)
    if isinstance(model, nn.DataParallel):
        model = model.module
    # state_dict = checkpoint['model_state_dict']
    # new_state_dict = {}
    # for k, v in state_dict.items():
    #     name = k[7:] if k.startswith('module.') else k  # Remove 'module.' prefix
    #     new_state_dict[name] = v
    # model.load_state_dict(new_state_dict)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.prompt = checkpoint['prompt']  # Load the prompt layer
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Model and prompt loaded from {checkpoint_path}")
    return model, optimizer, epoch, loss


    
def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)
