from tqdm.auto import tqdm
import torch
import wandb
import numpy as np
import os
import time
import gc
import einops as ein
import torch.nn.functional as F
from transformers import trainer

class BaselineTrainer():
    """
    훈련과정입니다.
    """
    def __init__(self, model, criterion, metric, optimizer, device, save_dir,
                 train_dataloader, valid_dataloader=None, lr_scheduler=None, epochs=1):
        self.model = model
        self.criterion = criterion
        self.metric = metric
        self.optimizer = optimizer
        self.device = device
        self.save_dir = save_dir
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.lr_scheduler = lr_scheduler
        self.epochs = epochs

        self.best_model_epoch, self.val_loss_values = [], []
        self.is_token_type_ids = False
        # check = True
        # for model_name in ['roberta', 'distilbert', 'albert', 'camembert', 'flaubert']:
        #     if model_name in model.model.name_or_path:
        #         check = False
        # if check and 'bert' in model.model.name_or_path:
        #     self.is_token_type_ids = True

    def train(self):
        """
        train_epoch를 돌고 valid_epoch로 평가합니다.
        """
        for epoch in range(self.epochs):
            standard_time = time.time()
            self._train_epoch(epoch)
            self._valid_epoch(epoch)
            wandb.log({'epoch' : epoch, 'runtime(Min)' : (time.time() - standard_time) / 60})
        self.select_best_model()
        torch.cuda.empty_cache()
        del self.model, self.train_dataloader, self.valid_dataloader
        gc.collect()
    
    def _train_epoch(self, epoch):
        gc.collect()
        self.model.train()
        epoch_loss = 0
        steps = 0
        pbar = tqdm(self.train_dataloader)
        for i, batch in enumerate(pbar):
            self.optimizer.zero_grad()
            steps += 1

            if self.is_token_type_ids: # BERT 모델일 경우 token_type_ids를 넣어줘야 합니다.
                inputs = {
                    'input_ids': batch['input_ids'].to(self.device),
                    'attention_mask': batch['attention_mask'].to(self.device),
                    'token_type_ids' : batch['token_type_ids'].to(self.device)
                }
            else:
                inputs = {
                    'input_ids': batch['input_ids'].to(self.device),
                    'attention_mask': batch['attention_mask'].to(self.device),
                }

            start_logits, end_logits = self.model(**inputs)

            start_positions = batch['start_positions'].to(self.device)
            end_positions = batch['end_positions'].to(self.device)

            # seq_len 길이만큼 boundary를 설정하여 seq_len 밖으로 벗어날 경우 벗어난 값을 최소값인 0(cls 토큰)으로 설정해줌
            start_positions = start_positions.clamp(0, start_logits.size(1))
            end_positions = end_positions.clamp(0, end_logits.size(1))

            # 각 start, end의 loss 평균
            loss = (self.criterion(start_logits, start_positions) + self.criterion(end_logits, end_positions)) / 2
                
            loss.backward()
            epoch_loss += loss.detach().cpu().numpy().item()
            
            self.optimizer.step()
            
            pbar.set_postfix({
                'loss' : epoch_loss / steps,
                'lr' : self.optimizer.param_groups[0]['lr'],
            })
            wandb.log({'train_loss':epoch_loss/steps})
        pbar.close()

    def _valid_epoch(self, epoch):
        val_loss = 0
        val_steps = 0
        start_logits_all, end_logits_all = [],[]
        len_val_dataset = self.valid_dataloader.dataset.num_rows
        with torch.no_grad():
            self.model.eval()
            for valid_batch in tqdm(self.valid_dataloader):
                val_steps += 1
                if self.is_token_type_ids: # BERT 모델일 경우 token_type_ids를 넣어줘야 합니다.
                    inputs = {
                        'input_ids': valid_batch['input_ids'].to(self.device),
                        'attention_mask': valid_batch['attention_mask'].to(self.device),
                        'token_type_ids' : valid_batch['token_type_ids'].to(self.device)
                    }
                else:
                    inputs = {
                            'input_ids': valid_batch['input_ids'].to(self.device),
                            'attention_mask': valid_batch['attention_mask'].to(self.device),
                        }

                start_logits, end_logits = self.model(**inputs)

                start_positions = valid_batch['start_positions'].to(self.device)
                end_positions = valid_batch['end_positions'].to(self.device)

                # seq_len 길이만큼 boundary를 설정하여 seq_len 밖으로 벗어날 경우 벗어난 값을 최소값인 0(cls 토큰)으로 설정해줌
                start_positions = start_positions.clamp(0, start_logits.size(1))
                end_positions = end_positions.clamp(0, end_logits.size(1))

                loss = (self.criterion(start_logits, start_positions) + self.criterion(end_logits, end_positions)) / 2
                val_loss += loss.detach().cpu().numpy().item()

                start_logits_all.append(start_logits.detach().cpu().numpy())
                end_logits_all.append(end_logits.detach().cpu().numpy())

            val_loss /= val_steps

            start_logits_all = np.concatenate(start_logits_all)[:len_val_dataset]
            end_logits_all = np.concatenate(end_logits_all)[:len_val_dataset]
            metrics = self.metric.compute_EM_f1(start_logits_all, end_logits_all, epoch)
            
            print(f"Epoch [{epoch+1}/{self.epochs}] Val_loss : {val_loss}")
            print(f"Epoch [{epoch+1}/{self.epochs}] Extact Match :", metrics['exact_match'])
            print(f"Epoch [{epoch+1}/{self.epochs}] F1_score :", metrics['f1'])
            wandb.log({'epoch' : epoch+1, 'val_loss' : val_loss})
            wandb.log({'epoch' : epoch+1, 'Exact_Match' : metrics['exact_match']})
            wandb.log({'epoch' : epoch+1, 'f1_score' : metrics['f1']})

            if epoch < 9:
                epoch = '0' + str(epoch+1)
            else:
                epoch = epoch + 1
            torch.save(self.model.state_dict(), f'save/{self.save_dir}/epoch:{epoch}_model.pt')
            print('save checkpoint!')

        self.best_model_epoch.append(f'save/{self.save_dir}/epoch:{epoch}_model.pt')
        self.val_loss_values.append(val_loss)

    def select_best_model(self):
        best_model = self.best_model_epoch[np.array(self.val_loss_values).argmin()]
        os.rename(best_model, best_model.split('.pt')[0] + '_best.pt')
