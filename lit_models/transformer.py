import faiss
import os
import pickle
import json
import random
import torch
import torch.nn as nn
import numpy as np
from functools import partial
from .base import BaseLitModel
from .util import f1_eval, compute_f1, acc, f1_score
from transformers.optimization import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup
from functools import reduce
from copy import deepcopy


def multilabel_categorical_crossentropy(y_pred, y_true):
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return (neg_loss + pos_loss).mean()

        
def decode(tokenizer, output_ids):
    return [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in output_ids]


class BertLitModel(BaseLitModel):
    """
    use AutoModelForMaskedLM, and select the output by another layer in the lit model
    """
    def __init__(self, model, args, tokenizer):
        super().__init__(model, args)
        self.tokenizer = tokenizer
        
        with open(f"{args.data_dir}/rel2id.json","r") as file:
            rel2id = json.load(file)
        
        Na_num = 0
        self.id2rel = {}
        for k, v in rel2id.items():
            if k == "NA" or k == "no_relation" or k == "Other":
                Na_num = v
                break
        self.rel2id = rel2id
        for k, v in rel2id.items():
            self.id2rel[int(v)] = k
        num_relation = len(rel2id)
        # init loss function
        self.loss_fn = multilabel_categorical_crossentropy if "dialogue" in args.data_dir else nn.CrossEntropyLoss()
        # ignore the no_relation class to compute the f1 score
        self.eval_fn = f1_eval if "dialogue" in args.data_dir else partial(f1_score, rel_num=num_relation, na_num=Na_num)
        self.best_f1 = 0
        self.t_lambda = args.t_lambda
        
        self.label_st_id = tokenizer("[class1]", add_special_tokens=False)['input_ids'][0]
        self.relation_tag = tokenizer("-", add_special_tokens=False)['input_ids'][0]
        self.relation_tokens = []
        self.subject_word = []
        self.object_word = []
        self.final_word = []
        self.other_subject_word = None
        self.other_object_word = None
        self.tokenizer = tokenizer

        self._init_label_word()

    def _init_label_word(self, ):
        args = self.args
        # ./dataset/dataset_name
        dataset_name = args.data_type #args.data_dir.split("/")[1]
        model_name_or_path = args.model_name_or_path.split("/")[-1]
        label_path = f"./dataset/{model_name_or_path}_{dataset_name}.pt"
        # [num_labels, num_tokens], ignore the unanswerable
        if "dialogue" in args.data_dir:
            label_word_idx = torch.load(label_path)[:-1]
        else:
            label_word_idx = torch.load(label_path)
        self.label_word_idx = label_word_idx
        num_labels = len(label_word_idx)
        print(num_labels)
        print(len(self.tokenizer))
        for a in range(1, num_labels+1):
            if args.MVRE:
                for j in range(args.multi_viewer_num):
                    self.tokenizer.add_tokens(f"[class{a}_{j}]", special_tokens=True)
            self.tokenizer.add_tokens(f"[relation{a}]", special_tokens=True)
        self.tokenizer.add_tokens("[other_subject_word]", special_tokens=True)
        self.tokenizer.add_tokens("[other_word]", special_tokens=True)

        print(len(self.tokenizer))
        self.model.resize_token_embeddings(len(self.tokenizer))
        print(self.model.config.vocab_size) 
        self.predict = nn.ModuleList([nn.Linear(self.model.config.hidden_size, 1).to(self.device) for i in range(args.multi_viewer_num)])
        print(self.tokenizer.mask_token_id)
        with torch.no_grad():
            word_embeddings = self.model.get_input_embeddings()
            if args.MVRE:
                continous_label_word = [[self.tokenizer(f"[class{i}_{j}]", add_special_tokens=False)['input_ids'] for j in range(args.multi_viewer_num)] for i in range(1, num_labels+1)] #[a[0] for a in self.tokenizer([f"[class{i}_0]" for i in range(1, num_labels+1)], add_special_tokens=False)['input_ids']]
            else:
                continous_label_word = [a[0] for a in self.tokenizer([f"[class{i}]" for i in range(1, num_labels+1)], add_special_tokens=False)['input_ids']]

            if self.args.init_answer_words:
                if self.args.init_answer_words_by_one_token:
                    for i, idx in enumerate(label_word_idx):
                        word_embeddings.weight[continous_label_word[i]] = word_embeddings.weight[idx][-1]
                else:
                    #import pdb;pdb.set_trace()
                    #"/home/fch/multi-viewer-prompt/dataset/semeval/k-shot/train4.txt"
                    #"/home/fch/multi-viewer-prompt/dataset/tacred/k-shot/train.txt"
                    #"/home/fch/multi-viewer-prompt/dataset/tacrev/k-shot/train.txt"
                    sample_file = "./template/semeval.txt" #"/home/fanchenghao/multi-viewer-prompt/dataset/tacred/k-shot/train3.txt" #"/home/fanchenghao/multi-viewer-prompt/dataset/semeval/k-shot/train4.txt" #os.path.join(self.args.data_dir, "train.txt") #"/home/fanchenghao/multi-viewer-prompt/dataset/semeval/k-shot/train3.txt"
                    if self.args.data_type == "tacrev":
                         sample_file = "./template/tacrev.txt"
                    elif self.args.data_type == "tacred":
                        sample_file = "./template/tacred.txt"
                    data = {}
                    with open(sample_file, 'r') as f:
                        for i in f:
                            data_i = json.loads(i)
                            str_i = " ".join(data_i["token"]) + "".join(data_i["h"]["name"]) + " " + "<mask> " * args.multi_viewer_num + "".join(data_i["t"]["name"])
                            if str(self.rel2id[data_i["relation"]]) not in data:
                                data[str(self.rel2id[data_i["relation"]])] = []
                            data[str(self.rel2id[data_i["relation"]])].append(str_i)
                    from transformers import pipeline
                    unmasker = pipeline('fill-mask', model='roberta-large')
                    for i, idx in enumerate(label_word_idx):
                        multi_w = np.random.uniform(low=0,high=1,size=(args.multi_viewer_num, len(idx)))
                        if args.MVRE:
                            if str(i) not in data:
                                str_test = None
                            else:
                                str_test = data[str(i)][0]
                                output = unmasker(str_test)
                            
                            for j in range(args.multi_viewer_num):
                                if not self.args.pipeline_init:
                                    word_embeddings.weight[continous_label_word[i][j]] = torch.mean(word_embeddings.weight[idx], dim=0)
                                    continue
                                if str_test is None:
                                    token_list = idx
                                    word_embeddings.weight[continous_label_word[i][j]] = torch.mean(word_embeddings.weight[idx], dim=0)
                                else:
                                    if args.multi_viewer_num == 1:
                                        token_str = output[0]["token_str"]
                                    else:
                                        token_str = output[j][0]["token_str"]
                                    token_list = [self.tokenizer.encode(token_str)[1]]
                                    if self.args.rm_SI:
                                        word_embeddings.weight[continous_label_word[i][j]] = torch.mean(word_embeddings.weight[token_list], dim=0)
                                    else:
                                        word_embeddings.weight[continous_label_word[i][j]] = torch.mean(word_embeddings.weight[torch.tensor(token_list + idx.numpy().tolist())], dim=0)

                        else:
                            word_embeddings.weight[continous_label_word[i]] = torch.mean(word_embeddings.weight[idx], dim=0)
            
            if self.args.init_type_words:
                so_word = [a[0] for a in self.tokenizer(["[obj]","[sub]"], add_special_tokens=False)['input_ids']]
                meaning_word = [a[0] for a in self.tokenizer(["person","organization", "location", "date", "country"], add_special_tokens=False)['input_ids']]
            
                for i, idx in enumerate(so_word):
                    word_embeddings.weight[so_word[i]] = torch.mean(word_embeddings.weight[meaning_word], dim=0)
            assert torch.equal(self.model.get_input_embeddings().weight, word_embeddings.weight)
            assert torch.equal(self.model.get_input_embeddings().weight, self.model.get_output_embeddings().weight)
        if args.MVRE:
            continous_label_word = []
            self.relation_tokens = [[self.tokenizer(f"[class{i}_{j}]", add_special_tokens=False)['input_ids'] for j in range(args.multi_viewer_num)] for i in range(1, num_labels+1)]
            count = 0
            for i, idx in enumerate(label_word_idx):
                if self.id2rel[i].endswith("(e2,e1)"):
                    continous_label_word.append([self.relation_tokens[i][j] for j in range(args.multi_viewer_num)])
                else:
                    continous_label_word.append([self.relation_tokens[i][j] for j in range(args.multi_viewer_num)])
           
        self.word2label = continous_label_word 
                
    def forward(self, x):
        return self.model(x)

    def fusion_sum_mask(self, input_ids, hidden_state):
        mask_bs_idx, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
        subject_mask_idx = mask_idx[[i for i in range(0, len(mask_idx), 3)]]
        relation_mask_idx = mask_idx[[i for i in range(1, len(mask_idx), 3)]]
        object_mask_idx = mask_idx[[i for i in range(2, len(mask_idx), 3)]]
        bs = input_ids.shape[0]
        subject_hidden_state = hidden_state[torch.arange(bs), subject_mask_idx] # [bs, 1, model_dim]
        relation_hidden_state = hidden_state[torch.arange(bs), relation_mask_idx]
        object_hidden_state = hidden_state[torch.arange(bs), object_mask_idx]
        sum_mask_hidden_state = subject_hidden_state + relation_hidden_state + object_hidden_state
        logits = self.model.lm_head(sum_mask_hidden_state)
        return logits[:, self.final_word]

    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        input_ids, attention_mask, labels, so = batch
        result = self.model(input_ids, attention_mask, return_dict=True, output_hidden_states=True)
        logits = result.logits

        if self.args.MVRE:
            logits1, _ = self.pvp_multi_viewer(logits, input_ids, hidden_state=result.hidden_states[-1])
            loss = self.loss_fn(logits1, labels)

            if self.args.use_contrastive:   
                contrastive_loss = self.contrastive_loss(beta=self.args.contrastive_beta)
                loss = loss + self.args.contrastive_ratio * contrastive_loss
        else:
            logits = self.model.lm_head(result.hidden_states[-1])
            logits, _ = self.pvp(logits, input_ids, labels)
            loss = self.loss_fn(logits, labels)
        self.log("Train/loss", loss)
        return loss
    
    def get_loss(self, logits, input_ids, labels):
        _, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
        bs = input_ids.shape[0]
        mask_output = logits[torch.arange(bs), mask_idx]
        
        loss = self.loss_fn(mask_output, labels)
        return loss


    def validation_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        if self.current_epoch <=0:#not in [10, 20, 30, 39]:
            return {}
        input_ids, attention_mask, labels, _ = batch
        result = self.model(input_ids, attention_mask, return_dict=True, output_hidden_states=True)
        logits = result.logits

        if self.args.MVRE:
            if self.args.merge_to_one:
                logits_list, _ = self.pvp_one(logits, input_ids, hidden_state=result.hidden_states[-1])
                logits_merge = torch.cat(logits_list, dim=1)
                logits1 = torch.sum(logits_merge, dim=1)#torch.max(logits_merge, dim=1)[0]
            else:
                logits1, _ = self.pvp_multi_viewer(logits, input_ids, hidden_state=result.hidden_states[-1])
            #import pdb;pdb.set_trace()
            
        else:
            logits1, _ = self.pvp(logits, input_ids, hidden_state=result.hidden_states[-1])
            logits2, _ = self.pvp(logits, input_ids, hidden_state=result.hidden_states[-1])
            logi1ts3, _ = self.pvp(logits, input_ids, hidden_state=result.hidden_states[-3])
        logits = logits1# + contrastive_loss #0.6*logits1 + 0.3*logits2 + 0.1*logits3
        #logits2 = self.fusion_sum_mask(input_ids, result.hidden_states[-3])
        loss1 = self.loss_fn(logits, labels)
        #loss2 = self.loss_fn(logits2, labels)
        loss = loss1# + loss2
        self.log("Eval/loss", loss)
        return {"eval_logits": logits.detach().cpu().numpy(), "eval_labels": labels.detach().cpu().numpy(), "inputs": input_ids.detach().cpu().numpy()}
    
    def validation_epoch_end(self, outputs) -> None:
        logits = np.concatenate([o["eval_logits"] for o in outputs])
        labels = np.concatenate([o["eval_labels"] for o in outputs])
        inputs = np.concatenate([o["inputs"] for o in outputs])
        
        f1 = self.eval_fn(logits, labels)['f1']
        label_dict = {}
        for i, x in enumerate(self.label_word_idx):
            label_dict[str(i)] = self.tokenizer.decode(x)
        like_dict = {}
        
        print(f1)
        self.log("Eval/f1", f1)
        if f1 > self.best_f1:
            self.best_f1 = f1
        self.log("Eval/best_f1", self.best_f1, prog_bar=True, on_epoch=True)

    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        input_ids, attention_mask, labels, _ = batch
        result = self.model(input_ids, attention_mask, return_dict=True, output_hidden_states=True)
        logits = result.logits

        if self.args.MVRE:
            if self.args.merge_to_one:
                logits_list, _ = self.pvp_one(logits, input_ids, hidden_state=result.hidden_states[-1])
                logits_merge = torch.cat(logits_list, dim=1)
                logits1 = torch.max(logits_merge, dim=1)[0]
            else:
                logits1, _ = self.pvp_multi_viewer(logits, input_ids, hidden_state=result.hidden_states[-1])
        else:
            logits1, _ = self.pvp(logits, input_ids, hidden_state=result.hidden_states[-1])
        logits = logits1
        return {"test_logits": logits.detach().cpu().numpy(), "test_labels": labels.detach().cpu().numpy()}

    def test_epoch_end(self, outputs) -> None:
        logits = np.concatenate([o["test_logits"] for o in outputs])
        labels = np.concatenate([o["test_labels"] for o in outputs])

        f1 = self.eval_fn(logits, labels)['f1']
        self.log("Test/f1", f1)
        
    @staticmethod
    def add_to_argparse(parser):
        BaseLitModel.add_to_argparse(parser)
        parser.add_argument("--t_lambda", type=float, default=0.01, help="")
        parser.add_argument("--t_gamma", type=float, default=0.3, help="")
        return parser
    
    def compute_multi_mask(self, mask_output, id=0):
        word2label_fir = [i[id][0] for i in self.word2label]
        final_output = torch.zeros_like(mask_output[:, word2label_fir])
        for i in range(len(self.word2label)):
            continue_mul_list = []
            for j in range(len(self.word2label[i][id])):
                continue_mul_list.append(mask_output[:, self.word2label[i][id][j]])
            final_output[:, i] = reduce(lambda x, y: x+y, continue_mul_list)
        final_output = torch.softmax(final_output, dim=-1)
        return final_output

    def pvp_one(self, logits, input_ids, labels=None, hidden_state=None, attention_score=None):
        mask_bs_idx, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
        bs = input_ids.shape[0]
        mask_output = logits[torch.arange(bs), mask_idx]
        multi_viewer_final_output = [self.compute_multi_mask(mask_output, id=j).view(bs, 1, -1) for j in range(self.args.multi_viewer_num)]
        return multi_viewer_final_output, None
    
    def pvp_multi_viewer(self, logits, input_ids, labels=None, hidden_state=None, attention_score=None):
        mask_bs_idx, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
        
        if self.args.MVRE:
            multi_viewer_mask_idx = [mask_idx[[i for i in range(j, len(mask_idx), self.args.multi_viewer_num)]] for j in range(self.args.multi_viewer_num)]
            bs = input_ids.shape[0]
            multi_viewer_mask_output = [logits[torch.arange(bs), multi_viewer_mask_idx[j]] for j in range(self.args.multi_viewer_num)]
            multi_viewer_final_output = [self.compute_multi_mask(multi_viewer_mask_output[j], id=j) for j in range(self.args.multi_viewer_num)]
            if hidden_state is not None:
                multi_viewer_hidden_state = [hidden_state[torch.arange(bs), multi_viewer_mask_idx[j]] for j in range(self.args.multi_viewer_num)]
                multi_viewer_w = [nn.Sigmoid()(self.predict[j](multi_viewer_hidden_state[j])).view(-1) for j in range(self.args.multi_viewer_num)]
                final_output_list = []
                for i in range(self.args.multi_viewer_num):
                   final_output_list.append(torch.mm(torch.diag_embed(multi_viewer_w[i]), multi_viewer_final_output[i]))
                final_output = reduce(lambda x, y: x+y, final_output_list)
            return final_output, None
        else:
            bs = input_ids.shape[0]
            mask_output = logits[torch.arange(bs), mask_idx]
            assert mask_idx.shape[0] == bs, "only one mask in sequence!"
            final_output = mask_output[:,self.word2label]
        return final_output, None

    def pvp(self, logits, input_ids, labels=None, hidden_state=None, attention_score=None):
        mask_bs_idx, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
        bs = input_ids.shape[0]
        mask_output = logits[torch.arange(bs), mask_idx]
        assert mask_idx.shape[0] == bs, "only one mask in sequence!"
        final_output = mask_output[:,self.word2label]
        return final_output, None
    
    def contrastive_loss(self, beta=0.1):
        #import pdb;pdb.set_trace()
        class_label_emb_all = [[] for j in range(self.args.multi_viewer_num)]
        like_loss = 0
        unlike_loss = 0
        label_num = len(self.word2label)
        for i in self.word2label:
            class_label_emb = []
            for j in range(self.args.multi_viewer_num):
                class_label_emb.append(self.model.get_output_embeddings().weight[i[j][0]])
                class_label_emb_all[j].append(class_label_emb[-1].unsqueeze(0))
            
            for j in range(self.args.multi_viewer_num):
                for k in range(j+1, self.args.multi_viewer_num):
                    like_loss += 1 - torch.nn.functional.cosine_similarity(class_label_emb[j].reshape(1, -1), class_label_emb[k].reshape(1, -1))
        for j in range(self.args.multi_viewer_num):
            class_label_emb_all[j] = torch.cat(class_label_emb_all[j], dim=0)
            similarity_loss = 1 - torch.cosine_similarity(class_label_emb_all[j].unsqueeze(1), class_label_emb_all[j].unsqueeze(0), dim=-1)
            unlike_loss += similarity_loss.sum()
        if self.args.multi_viewer_num == 1:
            return - beta * unlike_loss / self.args.multi_viewer_num / label_num / (label_num + 1) / 2
        else:
            return like_loss / label_num / self.args.multi_viewer_num / (self.args.multi_viewer_num - 1) / 2 - beta * unlike_loss / self.args.multi_viewer_num / label_num / (label_num + 1) / 2
        

    def configure_optimizers(self):
        no_decay_param = ["bias", "LayerNorm.weight"]

        if not self.args.two_steps: 
            parameters = self.model.named_parameters()
        else:
            # model.bert.embeddings.weight
            parameters = [next(self.model.named_parameters())]
        # only optimize the embedding parameters
        optimizer_group_parameters = [
            {"params": [p for n, p in parameters if not any(nd in n for nd in no_decay_param)], "weight_decay": self.args.weight_decay},
            {"params": [p for n, p in parameters if any(nd in n for nd in no_decay_param)], "weight_decay": 0}
        ]

        
        optimizer = self.optimizer_class(optimizer_group_parameters, lr=self.lr, eps=1e-8)
        print(self.num_training_steps)
        print(self.one_epoch_step)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.one_epoch_step*1, num_training_steps=self.num_training_steps)
        #scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=self.one_epoch_step*2, num_training_steps=self.num_training_steps)
        return {
            "optimizer": optimizer, 
            "lr_scheduler":{
                'scheduler': scheduler,
                'interval': 'step',  # or 'epoch'
                'frequency': 1,
            }
        }