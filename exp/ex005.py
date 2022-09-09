# ========================================
# library
# ========================================
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
from transformers import AutoModel, AutoConfig, get_linear_schedule_with_warmup, AutoTokenizer
from transformers.models.deberta.modeling_deberta import ContextPooler
from sklearn.metrics import mean_squared_error
import torch.utils.checkpoint
import logging
from contextlib import contextmanager
import sys

# ==================
# Constant
# ==================
ex = "005"
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
SEED = 0
num_workers = 4
BATCH_SIZE = 4
n_epochs = 5
es_patience = 10
max_len = 512
weight_decay = 0.1
lr = 5e-6
warmup_ratio = 0.1
eval_step = 100
folds = 5

MODEL_PATH = 'microsoft/deberta-v3-large'
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
tokenizer.add_special_tokens({"additional_special_tokens": ["\n"]})
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
        inputs = self.tokenizer(
            text,
            max_length=self.max_len,
            padding=False,
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True
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
    def __init__(self):
        super(FeedbackModel, self).__init__()
        self.config = AutoConfig.from_pretrained(MODEL_PATH)
        self.backbone = AutoModel.from_config(self.config)
        self.backbone.resize_token_embeddings(len(tokenizer))

        self.pooler = ContextPooler(self.config)
        self.out = nn.Linear(self.pooler.output_dim, 6)

        self.backbone.gradient_checkpointing_enable()

    def forward(self, ids, mask, token_type_ids=None):
        emb = self.backbone(ids, attention_mask=mask, token_type_ids=token_type_ids)[
            'last_hidden_state']
        output = self.pooler(emb)
        output = self.out(output)
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
            with torch.cuda.amp.autocast():
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


def calc_loss(y_true, y_pred):
    colwise_mse = torch.mean(torch.square(y_pred - y_true), dim=0)
    loss = torch.mean(torch.sqrt(colwise_mse), dim=0)
    return loss


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
        y_train = y.iloc[fold_array != fold]
        x_val = data_dict.iloc[fold_array == fold].reset_index(drop=True)
        y_val = y.iloc[fold_array != fold]

        train_datagen = FeedbackDataset(
            x_train['full_text'].values, tokenizer, max_len, y_train.values.reshape(-1,6))

        train_generator = DataLoader(
            dataset=train_datagen,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn_fast,
            pin_memory=True
        )

        valid_datagen = FeedbackDataset(
            x_val['full_text'].values, tokenizer, max_len, y_val.values.reshape(-1,6))
        valid_generator = DataLoader(
            dataset=valid_datagen,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn_fast,
            pin_memory=True
        )

        model = FeedbackModel()
        torch.save(model.config, CONFIG_SAVE_PATH)
        model.to(device)

        num_train_steps = int(len(train_generator)*n_epochs)
        num_warmup_steps = int(warmup_ratio*num_train_steps)

        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            lr=lr,
            betas=(0.9, 0.98),
            weight_decay=weight_decay,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_train_steps
        )

        criterion = MCRMSELoss()
        best_val = None
        scaler = GradScaler()
        awp = AWP(
            model,
            optimizer,
            adv_lr=1e-2,
            adv_eps=1e-3,
            start_epoch=1,
            scaler=scaler
        )
        for epoch in range(n_epochs):
            with timer(f'model_fold:{epoch}'):
                train_losses = AverageMeter()
                
                val_scores = AverageMeter()

                model.train()

                for step, (batch_input_ids, batch_attention_mask, batch_target) in enumerate(train_generator):
                    batch_input_ids = batch_input_ids.to(device)
                    batch_attention_mask = batch_attention_mask.to(device)
                    batch_target = torch.from_numpy(
                        np.array(batch_target)).float().to(device)

                    optimizer.zero_grad()
                    with autocast():
                        logits = model(batch_input_ids, batch_attention_mask)
                        loss = criterion(logits, batch_target)

                    train_losses.update(loss.item(), logits.size(0))
                    scaler.scale(loss).backward()
                    awp.attack_backward(batch_input_ids, batch_attention_mask, batch_target, epoch)
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    
                    if step % eval_step == 0:
                        val_losses = AverageMeter()
                        model.eval()
                        preds = np.ndarray([0, 6])
                        for i, (batch_input_ids, batch_attention_mask, batch_target) in enumerate(valid_generator):
                            batch_input_ids = batch_input_ids.to(device)
                            batch_attention_mask = batch_attention_mask.to(device)
                            batch_target = torch.from_numpy(
                                np.array(batch_target)).float().to(device)

                            with torch.no_grad():
                                logits = model(batch_input_ids, batch_attention_mask)
                                
                            loss = calc_loss(batch_target, logits)    
                            val_losses.update(loss.item(), logits.size(0))

                            logits = logits.to('cpu').numpy()
                            preds = np.concatenate(
                                [preds, logits], axis=0
                            )

                        avg_loss = val_losses.avg
                        LOGGER.info(
                            f'Fold: {fold} | Epoch: {epoch} | Step: {step} | Loss: {avg_loss}'
                        )

                        # ===================
                        # early stop
                        # ===================
                        if not best_val:
                            best_val = val_losses.avg
                            oof[fold_array == fold] = preds.reshape(-1, 6)
                            torch.save(model.state_dict(), MODEL_PATH_BASE +
                                    f"_{fold}.pth")  # Saving the model
                            continue

                        if val_losses.avg <= best_val:
                            best_val = val_losses.avg
                            oof[fold_array == fold] = preds.reshape(-1, 6)
                            torch.save(model.state_dict(), MODEL_PATH_BASE +
                                    f"_{fold}.pth")  # Saving current best model

val_score = calc_loss(oof, y.values.reshape(-1))
LOGGER.info(f'oof_score:{val_score}')
np.save(OOF_SAVE_PATH, oof)
