from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import *
from model.embed import *
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score,roc_auc_score
from utils.metrics_label import *
from utils.metrics_score import VUS_PR, VUS_ROC
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import copy
from thop import profile
warnings.filterwarnings('ignore')


class Exp_Anomaly_Detection(Exp_Basic):
    def __init__(self, args):
        super(Exp_Anomaly_Detection, self).__init__(args)
        self.args = args
        self.model = None
        
    def _build_model(self, dataset_config):
        enc_in = dataset_config['enc_in']
        c_out = dataset_config['c_out']
        self.model = self.model_dict[self.args.model].Model(
            self.args,
            enc_in=enc_in,
            c_out=c_out
        ).float().to(self.device)

        if self.args.continue_training == 1:
            setting = self.args.train_path
            print('loading model..')
            print('continue training..')
            print(setting)
            state_dict = torch.load(setting)
            
            self.model= state_dict
        if self.args.use_multi_gpu and self.args.use_gpu:
            self.model = nn.DataParallel(self.model, device_ids=self.args.device_ids)

        return self.model

    def _get_data(self, flag, dataset_type):
        data_set, data_loader = data_provider(self.args, flag, dataset_type)
        return data_set,data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
        
    def fine_tune_prompt(self, model, train_loader, num_epochs=10, learning_rate=1e-5):
        if isinstance(model, torch.nn.DataParallel):
            actual_model = model.module
        else:
            actual_model = model
        if actual_model.previous_prompt is not None:
            print("Using previous prompt as starting point")          
            previous_prompt = actual_model.previous_prompt.detach().clone().float().to(self.device)
            previous_prompt = torch.clamp(previous_prompt, -10.0, 10.0)
            actual_model.vae_prompt.data = previous_prompt
        else:
            batch_x, _ = next(iter(train_loader))
            batch_x = batch_x.float().to(self.device)     
            current_prompt = actual_model.generate_vae_prompt(batch_x)
            current_prompt = current_prompt.float().to(self.device)
            current_prompt = torch.clamp(current_prompt, -10.0, 10.0)
            actual_model.vae_prompt = nn.Parameter(current_prompt.detach().clone(), requires_grad=True)
            
        optimizer = optim.Adam([actual_model.vae_prompt], lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)     
        criterion = nn.MSELoss()
        actual_model.train()  
  
        initial_model_params = {}
        for name, param in actual_model.named_parameters():
            if param.requires_grad and name != "vae_prompt":
                initial_model_params[name] = param.detach().clone()
        
        prompt_mean = torch.mean(actual_model.vae_prompt).item()
        prompt_std = torch.std(actual_model.vae_prompt).item()
       
        best_loss = float('inf')
        best_prompt = None
        
        for epoch in range(num_epochs):
            running_loss = 0.0
            batch_count = 0
            valid_batch_count = 0
            
            for i, (batch_x, batch_y) in enumerate(train_loader):
                batch_x = batch_x.float().to(self.device)
                optimizer.zero_grad()
                outputs = actual_model(batch_x, None, None, None)
                f_dim = -1 if self.args.features=='MS' else 0
                outputs = outputs[:,:,f_dim:]
                loss = criterion(outputs, batch_x)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(actual_model.parameters(), max_norm=1.0)
                with torch.no_grad():
                    for name, param in actual_model.named_parameters():
                        if param.requires_grad and name != "vae_prompt" and param.grad is not None:
                            delta = param.grad.clone()
                            delta = torch.clamp(delta, -1.0, 1.0) 
                            prompt_value = torch.mean(actual_model.vae_prompt).item()
                            prompt_scale = torch.sigmoid(torch.tensor(prompt_value)).item() * 0.01
                            if name in initial_model_params:
                                param.data = initial_model_params[name] + prompt_scale * delta
                optimizer.step()
                running_loss += loss.item()
                batch_count += 1
                valid_batch_count += 1
                
            avg_loss = running_loss / batch_count if batch_count > 0 else float('inf')
            print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.6f}, Valid batches: {valid_batch_count}/{batch_count}")
            scheduler.step(avg_loss)
            if avg_loss < best_loss and not (np.isnan(avg_loss) or np.isinf(avg_loss)):
                best_loss = avg_loss
                best_prompt = actual_model.vae_prompt.detach().clone()
            if hasattr(actual_model, 'vae_prompt') and actual_model.vae_prompt is not None:
                prompt_save_path = os.path.join(self.args.checkpoints, f"prompt_epoch_{epoch+1}.pth")
                prompt_to_save = actual_model.vae_prompt.detach().clone()
                if torch.isnan(prompt_to_save).any() or torch.isinf(prompt_to_save).any():
                    print("Warning: NaN or Inf detected in prompt before saving. Using best prompt instead.")
                    prompt_to_save = best_prompt if best_prompt is not None else torch.zeros_like(prompt_to_save) + 0.01
                prompt_cpu = prompt_to_save.cpu()
                torch.save(prompt_cpu, prompt_save_path)
        
        if best_prompt is not None:
            actual_model.vae_prompt.data = best_prompt.clone() 
        actual_model.save_prompt(actual_model.vae_prompt)
        
    def train_on_multiple_datasets(self, dataset_types):
        all_test_loaders = []
        all_train_loaders = []

        dataset_configs = {
            'SMD': {'enc_in': 38, 'c_out': 38},'SMAP': {'enc_in': 25, 'c_out': 25},'MSL': {'enc_in': 55, 'c_out': 55},
            'PSM': {'enc_in': 25, 'c_out': 25},'SWAT': {'enc_in': 51, 'c_out': 51},'UCR': {'enc_in': 1, 'c_out': 1},
        }
        self.model = self._build_model(dataset_configs[dataset_types[0]])

        for idx, dataset_type in enumerate(dataset_types):
            print(f'>>>>>>>>>>>>>>>> Start training on {dataset_type} dataset >>>>>>>>>>>>>>>>>>>>')
            train_data, train_loader = self._get_data(flag='train', dataset_type=dataset_type)
            vali_data, vali_loader = self._get_data(flag='val', dataset_type=dataset_type)
            test_data, test_loader = self._get_data(flag='test', dataset_type=dataset_type)
            all_train_loaders.append(train_loader)
            all_test_loaders.append(test_loader)
        
            is_first_dataset = (idx == 0)
            dataset_save_dir = os.path.join(self.args.checkpoints, f"model_{dataset_type}")
            if not os.path.exists(dataset_save_dir):
                os.makedirs(dataset_save_dir)
                
            if is_first_dataset:
                print("Training on the first dataset...")
                self.fine_tune_prompt(self.model, train_loader, num_epochs=2, learning_rate=1e-4)
            else:
                prev_prompt_path = os.path.join(self.args.checkpoints, f"model_{dataset_types[idx-1]}", 'final_prompt.pth')
                if os.path.exists(prev_prompt_path):
                    saved_prompt = torch.load(prev_prompt_path, map_location=self.device)
                
                    if isinstance(self.model, torch.nn.DataParallel):
                        self.model.module.previous_prompt = saved_prompt
                    else:
                        self.model.previous_prompt = saved_prompt
                    print(f"Loaded prompt from previous dataset {dataset_types[idx-1]}, shape: {saved_prompt.shape}")
                else:
                    print(f"Warning: No prompt found for previous dataset {dataset_types[idx-1]}. Proceeding with fresh initialization.")

                self.fine_tune_prompt(self.model, train_loader,  num_epochs=1,  learning_rate=1e-4)

            model_optim = self._select_optimizer()
            criterion = self._select_criterion()
            path = os.path.join(self.args.checkpoints, f"model_{dataset_type}")
            if not os.path.exists(path):
                os.makedirs(path)
            self.train_single_dataset(train_loader, vali_loader, test_loader, model_optim, criterion, path, dataset_type)
        
            if idx == 0:
                freeze_attention_modules(self.model)
            final_prompt_path = os.path.join(path, 'final_prompt.pth')
            
            if isinstance(self.model, torch.nn.DataParallel):
                current_prompt = self.model.module.vae_prompt.detach().clone()
            else:
                current_prompt = self.model.vae_prompt.detach().clone()
                
            torch.save(current_prompt, final_prompt_path)
            print(f"Saved final prompt for dataset {dataset_type}")
      
        for idx, dataset_type in enumerate(dataset_types):
            print(f'>>>>>>>>>>>>>>>> Start testing on {dataset_type} dataset >>>>>>>>>>>>>>>>>>>>')
            train_loader = all_train_loaders[dataset_types.index(dataset_type)]
            test_loader = all_test_loaders[dataset_types.index(dataset_type)]
            best_model_path = os.path.join(self.args.checkpoints, f"model_{dataset_type}", 'checkpoint.pth')
            load_state_dict_with_prefix(self.model, best_model_path, self.device)
            self.test(train_loader, test_loader, dataset_type, best_model_path)

    def train_single_dataset(self, train_loader, vali_loader, test_loader, model_optim, criterion, path, dataset_type):
        print("======================TRAIN MODE======================")
        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        if isinstance(self.model, torch.nn.DataParallel):
            self.model = self.model.module
        prompt_save_dir = os.path.join(path, 'prompts')
        if not os.path.exists(prompt_save_dir):
            os.makedirs(prompt_save_dir)
            
        for epoch in range(self.args.train_epochs):
            self.model.train()
            epoch_loss = []
            epoch_time=time.time()
            for i, (batch_x, batch_y) in enumerate(train_loader):
            
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                outputs = self.model(batch_x, None, None, None)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                criterion_loss = criterion(outputs, batch_x)
                epoch_loss.append(criterion_loss.item())
                criterion_loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.mean(epoch_loss)
            vali_loss = self.vali(vali_loader, criterion)
            print(f"Epoch {epoch+1}, Train Loss: {train_loss:.6f}, Vali Loss: {vali_loss:.6f}")
            if hasattr(self.model, 'vae_prompt') and self.model.vae_prompt is not None:
                prompt_save_path = os.path.join(prompt_save_dir, f'vae_prompt-epoch-{epoch+1}.pth')
                torch.save(self.model.vae_prompt.detach().cpu(), prompt_save_path)
                self.model.save_prompt(self.model.vae_prompt)
         
            if not np.isinf(vali_loss) and not np.isnan(vali_loss):
                early_stopping(vali_loss, self.model, path)
                if early_stopping.early_stop:
                    print("Early stopping triggered")
                    break
            else:
                print("Skipping early stopping check due to invalid validation loss.")

            adjust_learning_rate(model_optim, epoch + 1, self.args, self.model)
        if isinstance(self.model, torch.nn.DataParallel):
            final_prompt = self.model.module.vae_prompt.detach().clone()
        else:
            final_prompt = self.model.vae_prompt.detach().clone()
            
        final_prompt_path = os.path.join(path, 'final_prompt.pth')
        torch.save(final_prompt, final_prompt_path)
        print(f"Saved final prompt, shape: {final_prompt.shape}")
        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))

    def vali(self, vali_loader, criterion):
        # print("======================VALI MODE======================")
        total_loss = []
        self.model.eval()
        if hasattr(self.model, 'vae_prompt') and self.model.vae_prompt is not None:
            if self.model.vae_prompt.device != self.device:
                self.model.vae_prompt.data = self.model.vae_prompt.data.to(self.device)
                print(f"Moved vae_prompt to {self.device} for validation")    
        with torch.no_grad(): 
            for i, (batch_x, _) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                outputs = self.model(batch_x, None, None, None)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                pred = outputs.detach().cpu()
                true = batch_x.detach().cpu() 
                loss = criterion(pred, true)
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        
        return total_loss

    def remove_module_prefix(self, state_dict):
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('module.'):
                new_state_dict[key[7:]] = value  
            else:
                new_state_dict[key] = value
        return new_state_dict
    
    def test(self, train_loader, test_loader, dataset_type, best_model_path):
        # print("======================TEST MODE======================")
        test_start_time = time.time()
        torch.cuda.empty_cache()

        attens_energy = []
        folder_path = './test_results/' 
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        self.model.eval()
        self.anomaly_criterion = nn.MSELoss(reduce=False)

        # (1) stastic on the train set
        print("Stastic on the train set...")
        train_energy=[]
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(train_loader):
                batch_x = batch_x.float().to(self.device)
                outputs = self.model(batch_x, None, None, None)
                score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
                score = score.detach().cpu().numpy()
                attens_energy.append(score)
        
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) find the threshold
        print("Find the threshold...")
        test_energy = []
        test_labels = []
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                outputs = self.model(batch_x, None, None, None)
                score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
                score = score.detach().cpu().numpy()
                test_energy.append(score)
                test_labels.append(batch_y)

        test_energy = np.concatenate(test_energy, axis=0).reshape(-1)
        test_energy = np.array(test_energy)

        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        threshold = np.percentile(combined_energy, 100 - self.args.anomaly_ratio)

        print(f"Threshold: {threshold}")

        # (3) evaluation on the test set
        print("Evaluation on the test set...")
        pred = (test_energy > threshold).astype(int)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_labels = np.array(test_labels)
        gt = test_labels.astype(int)
        print("pred:   ", pred.shape)
        print("gt:     ", gt.shape)

        # (4) detection adjustment
        gt_adjusted,pred_adjusted=adjustment(gt,pred)   
        gt_adjusted=np.array(gt_adjusted)
        pred_adjusted=np.array(pred_adjusted)
        print("pred_adjusted: ", pred_adjusted.shape)
        print("gt_adjusted:   ", gt_adjusted.shape)

        accuracy = accuracy_score(gt_adjusted, pred_adjusted)
        precision, recall, f_score, support = precision_recall_fscore_support(gt_adjusted, pred_adjusted, average='binary')
        roc_auc = roc_auc_score(gt_adjusted, pred_adjusted)
        affiliation_f_score = affiliation_f(gt_adjusted, pred_adjusted)
        vus_roc = VUS_ROC(gt_adjusted, pred_adjusted)
        vus_pr = VUS_PR(gt_adjusted, pred_adjusted)

        print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F-score: {f_score:.4f}, Affiliation F-score: {affiliation_f_score:.4f}, VUS_PR: {vus_pr:.4f}, VUS_ROC: {vus_roc:.4f}, ROC-AUC: {roc_auc:.4f}")
        print(f"Total test time: {time.time() - test_start_time}")
        torch.cuda.empty_cache()
        return