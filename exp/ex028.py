# ========================================
# library
# ========================================
from cgitb import enable
import numpy as np
import pandas as pd
import os
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
import torch
import random
from torch.cuda.amp import autocast, GradScaler
import time
from transformers import AutoModel, AutoConfig, get_cosine_schedule_with_warmup, AutoTokenizer
from transformers.models.deberta.modeling_deberta import ContextPooler
from sklearn.metrics import mean_squared_error
import torch.utils.checkpoint
import logging
from contextlib import contextmanager
import sys
from prettytable import PrettyTable
from tqdm import tqdm

# ==================
# Constant
# ==================
ex = "028"
TRAIN_PATH = "../input/fb3/train_folds.csv"
if not os.path.exists(f"../output/ex/ex{ex}"):
    os.makedirs(f"../output/ex/ex{ex}")
    os.makedirs(f"../output/ex/ex{ex}/ex{ex}_model")

MODEL_PATH_BASE = f"../output/ex/ex{ex}/ex{ex}_model/ex{ex}"
OOF_SAVE_PATH = f"../output/ex/ex{ex}/ex{ex}_oof.npy"
LOGGER_PATH = f"../output/ex/ex{ex}/ex{ex}.txt"
CONFIG_SAVE_PATH = f"../output/ex/ex{ex}/ex{ex}_model/ex{ex}_config.pth"
TOKENIZER_SAVE_PATH = f"../output/ex/ex{ex}/ex{ex}_tokenizer/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============
# Settings
# ===============
SEED = 42
num_workers = 4
BATCH_SIZE = 8
n_epochs = 4
es_patience = 10
max_len = 512
weight_decay = 0.01
backbone_lr = 1e-5
classifier_lr = 1e-5
warmup_ratio = 0
folds = 4
gradient_accumulation_steps = 1

MODEL_PATH = 'microsoft/deberta-v3-large'
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
# tokenizer.add_special_tokens({"additional_special_tokens": ["\n"]})
tokenizer.save_pretrained(TOKENIZER_SAVE_PATH)

# ===============
# Functions
# ===============


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class FeedbackDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len, targets=None):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.target = targets

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        inputs = self.tokenizer.encode_plus(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            add_special_tokens=True
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]
        if self.target is not None:
            return {
                "input_ids": ids,
                "attention_mask": mask,
                "token_type_ids": token_type_ids,
                "target": self.target[item]
            }
        else:
            return {
                "input_ids": ids,
                "attention_mask": mask,
                "token_type_ids": token_type_ids,
            }


class Collate:
    def __init__(self, tokenizer, isTrain=True):
        self.tokenizer = tokenizer
        self.isTrain = isTrain

    def __call__(self, batch):
        output = dict()
        output["input_ids"] = [sample["input_ids"] for sample in batch]
        output["attention_mask"] = [sample["attention_mask"]
                                    for sample in batch]
        if self.isTrain:
            output["target"] = [sample["target"] for sample in batch]

        # calculate max token length of this batch
        batch_max = max([len(ids) for ids in output["input_ids"]])

        # add padding
        if self.tokenizer.padding_side == "right":
            output["input_ids"] = [
                s + (batch_max - len(s)) * [self.tokenizer.pad_token_id] for s in output["input_ids"]]
            output["attention_mask"] = [
                s + (batch_max - len(s)) * [0] for s in output["attention_mask"]]
        else:
            output["input_ids"] = [
                (batch_max - len(s)) * [self.tokenizer.pad_token_id] + s for s in output["input_ids"]]
            output["attention_mask"] = [
                (batch_max - len(s)) * [0] + s for s in output["attention_mask"]]

        # convert to tensors
        output["input_ids"] = torch.tensor(
            np.array(output["input_ids"]), dtype=torch.long)
        output["attention_mask"] = torch.tensor(
            np.array(output["attention_mask"]), dtype=torch.long)
        if self.isTrain:
            output["target"] = torch.tensor(
                np.array(output["target"]), dtype=torch.float32)

        return output['input_ids'], output['attention_mask'], output['target']


collate_fn_fast = Collate(tokenizer, isTrain=True)


class FeedbackModel(nn.Module):
    def __init__(self, config_path=None, pretrained=False):
        super(FeedbackModel, self).__init__()
        if config_path is None:
            self.config = AutoConfig.from_pretrained(MODEL_PATH, output_hidden_states=True)
            self.config.hidden_dropout = 0.
            self.config.hidden_dropout_prob = 0.
            self.config.attention_dropout = 0.
            self.config.attention_probs_dropout_prob = 0.
            LOGGER.info(self.config)
        else:
            self.config = torch.load(config_path)
        if pretrained:
            self.deberta = AutoModel.from_pretrained(MODEL_PATH, config=self.config)
        else:
            self.deberta = AutoModel.from_config(self.config)

        self.fc = nn.Linear(self.config.hidden_size, 6)
        self._init_weights(self.fc)
        self.deberta.gradient_checkpointing_enable()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, ids, mask, token_type_ids=None):
        emb = self.deberta(ids, attention_mask=mask, token_type_ids=token_type_ids)[
            'last_hidden_state'][:, 0, :]
        output = self.fc(emb)
        return output


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss


class MCRMSELoss(nn.Module):
    def __init__(self, num_scored=6):
        super().__init__()
        self.rmse = RMSELoss()
        self.num_scored = num_scored

    def forward(self, yhat, y):
        score = 0
        for i in range(self.num_scored):
            score += self.rmse(yhat[:, i], y[:, i]) / self.num_scored

        return score


class AWP:
    def __init__(
        self,
        model,
        optimizer,
        adv_param="weight",
        adv_lr=1,
        adv_eps=0.2,
        start_epoch=0,
        adv_step=1,
        scaler=None
    ):
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.start_epoch = start_epoch
        self.adv_step = adv_step
        self.backup = {}
        self.backup_eps = {}
        self.scaler = scaler

    def attack_backward(self, ids, mask, labels, epoch):
        if (self.adv_lr == 0) or (epoch < self.start_epoch):
            return None

        self._save()
        for i in range(self.adv_step):
            self._attack_step()
            with autocast(enabled=True):
                output = model(ids, mask)
                adv_loss = criterion(output, labels)
            self.optimizer.zero_grad()
            self.scaler.scale(adv_loss).backward() 
            
        self._restore()  

    def _attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]), self.backup_eps[name][1]
                    )
    def _save(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def _restore(self,):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}


def MCRMSE(y_trues, y_preds):
    scores = []
    idxes = y_trues.shape[1]
    for i in range(idxes):
        y_true = y_trues[:, i]
        y_pred = y_preds[:, i]
        score = mean_squared_error(y_true, y_pred, squared=False)  # RMSE
        scores.append(score)
    mcrmse_score = np.mean(scores)
    return mcrmse_score, scores
    

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def setup_logger(out_file=None, stderr=True, stderr_level=logging.INFO, file_level=logging.DEBUG):
    LOGGER.handlers = []
    LOGGER.setLevel(min(stderr_level, file_level))

    if stderr:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(FORMATTER)
        handler.setLevel(stderr_level)
        LOGGER.addHandler(handler)

    if out_file is not None:
        handler = logging.FileHandler(out_file)
        handler.setFormatter(FORMATTER)
        handler.setLevel(file_level)
        LOGGER.addHandler(handler)

    LOGGER.info("logger set up")
    return LOGGER


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    LOGGER.info(f'[{name}] done in {time.time() - t0:.0f} s')


LOGGER = logging.getLogger()
FORMATTER = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
setup_logger(out_file=LOGGER_PATH)

# ================================
# Main
# ================================
data_dict = pd.read_csv(TRAIN_PATH)
y_cols = ['cohesion', 'syntax', 'vocabulary',
          'phraseology', 'grammar', 'conventions']
y = data_dict[y_cols]
fold_array = data_dict['fold'].values

# ================================
# train
# ================================
with timer('deberta-v3-large'):
    set_seed(SEED)
    oof = np.zeros([len(data_dict), 6])
    for fold in range(folds):
        x_train = data_dict.iloc[fold_array != fold].reset_index(drop=True)
        y_train = y.iloc[fold_array != fold].reset_index(drop=True)

        x_val = data_dict.iloc[fold_array == fold].reset_index(drop=True)
        y_val = y.iloc[fold_array == fold].reset_index(drop=True)

        train_datagen = FeedbackDataset(
            x_train['full_text'].values, tokenizer, max_len, y_train.values.reshape(-1, 6))

        train_generator = DataLoader(
            dataset=train_datagen,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn_fast,
            pin_memory=True,
            drop_last=True
        )

        valid_datagen = FeedbackDataset(
            x_val['full_text'].values, tokenizer, max_len, y_val.values.reshape(-1, 6))
        valid_generator = DataLoader(
            dataset=valid_datagen,
            batch_size=BATCH_SIZE*2,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn_fast,
            pin_memory=True,
            drop_last=False
        )

        model = FeedbackModel(pretrained=True)
        torch.save(model.config, CONFIG_SAVE_PATH)
        model.to(device)

        num_train_steps = int(len(train_generator)*n_epochs)
        num_warmup_steps = int(warmup_ratio*num_train_steps)

        param_optimizer = list(model.deberta.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'lr': backbone_lr, 'weight_decay': weight_decay},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'lr': backbone_lr, 'weight_decay': 0.0},
            {'params': [p for n, p in model.named_parameters() if "deberta" not in n],
             'lr': classifier_lr, 'weight_decay': 0.0}
        ]

        optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            lr=backbone_lr,
            betas=(0.9, 0.999),
            weight_decay=weight_decay,
            eps=1e-6
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_train_steps,
            num_cycles=0.5
        )

        criterion = nn.SmoothL1Loss(reduction='mean')
        best_val = None
        scaler = GradScaler(enabled=True)
        # awp = AWP(
        #     model,
        #     optimizer,
        #     adv_lr=1e-2,
        #     adv_eps=1e-3,
        #     start_epoch=2,
        #     scaler=scaler
        # )
        for epoch in range(n_epochs):
            with timer(f'model_fold:{epoch}'):
                model.train()
                train_losses = AverageMeter()
                tk0 = tqdm(train_generator, total=len(train_generator))
                for step, (batch_input_ids, batch_attention_mask, batch_target) in enumerate(tk0):
                    batch_input_ids = batch_input_ids.to(device)
                    batch_attention_mask = batch_attention_mask.to(device)
                    batch_target = torch.from_numpy(
                        np.array(batch_target)).float().to(device)

                    with autocast(enabled=True):
                        logits = model(batch_input_ids, batch_attention_mask)
                        loss = criterion(logits, batch_target)

                    train_losses.update(loss.item(), logits.size(0))
                    scaler.scale(loss).backward()
                    # awp.attack_backward(batch_input_ids, batch_attention_mask, batch_target, epoch)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1000)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()

                    tk0.set_postfix(loss=train_losses.avg)

                val_losses = AverageMeter()
                model.eval()
                preds = np.ndarray([0, 6])
                tk1 = tqdm(valid_generator, total=len(valid_generator))
                for i, (batch_input_ids, batch_attention_mask, batch_target) in enumerate(tk1):
                    batch_input_ids = batch_input_ids.to(device)
                    batch_attention_mask = batch_attention_mask.to(
                        device)
                    batch_target = torch.from_numpy(
                        np.array(batch_target)).float().to(device)

                    with torch.no_grad():
                        logits = model(batch_input_ids,
                                        batch_attention_mask)
                        loss = criterion(batch_target, logits)
                    
                    val_losses.update(loss.item(), logits.size(0))

                    preds = np.concatenate(
                        [preds, logits.detach().cpu().numpy()], axis=0
                    )
                
                avg_loss = val_losses.avg
                avg_score, scores = MCRMSE(y_val.values, preds)
                train_loss = train_losses.avg
                score_table = PrettyTable(y_cols)
                score_table.add_row(scores)
                LOGGER.info(
                    f'Fold: {fold} | Epoch: {epoch} | Train Loss: {train_loss} | Val Loss: {avg_loss} | Val Score: {avg_score}'
                )
                LOGGER.info(score_table)

                # ===================
                # early stop
                # ===================
                if not best_val:
                    best_val = avg_score
                    oof[fold_array == fold] = preds.reshape(-1, 6)
                    torch.save(model.state_dict(), MODEL_PATH_BASE +
                                f"_{fold}.pth")  # Saving the model
                    continue

                if avg_score <= best_val:
                    best_val = avg_score
                    oof[fold_array == fold] = preds.reshape(-1, 6)
                    torch.save(model.state_dict(), MODEL_PATH_BASE +
                                f"_{fold}.pth")  # Saving current best model

val_score, scores = MCRMSE(y.values.reshape(-1, 6), oof)
LOGGER.info(f'oof_score:{val_score} | target scores: {scores}')
np.save(OOF_SAVE_PATH, oof)
