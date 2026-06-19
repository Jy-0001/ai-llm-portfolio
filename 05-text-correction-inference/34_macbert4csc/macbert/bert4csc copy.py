import torch
import torch.nn as nn
from transformers import BertForMaskedLM
from base_model import FocalLoss

class Bert4Csc(torch.nn.Module):
    def __init__(self, args, tokenizer):
        super(Bert4Csc, self).__init__()
        self.args = args
        
        self.bert = BertForMaskedLM.from_pretrained(args.bert_dir)

        self.detection = nn.Linear(self.bert.config.hidden_size, 1)

        self.sigmoid = nn.Sigmoid()

        self.tokenizer = tokenizer

        self.w = args.hyper_params

    def batch_encode(self, batch_data):
        max_len = max([len(x) for x in batch_data]) + 2

        input_ids, token_type_ids, attentionO_mask = [], [], []

        encoded_inputs = self.tokenizer.batch_encode_plus(
            batch_data, 
            max_length=max_len, 
            padding='max_length', 
            truncation=True, 
            return_tensors='pt', 
            return_token_type_ids=True, 
            return_attention_mask=True
        )

        return {'input_ids':encoded_inputs['input_ids'], 'attention_mask':encoded_inputs['attention_mask'], 'token_type_ids':encoded_inputs['token_type_ids']}
    
    def forward(self, texts, cor_labels=None, det_labels=None, device=None):
        if cor_labels is not None:
            text_labels = self.batch_encode(cor_labels)['input_ids']

            text_labels[text_labels == 0] = -100

            text_labels = text_labels.to(device)

        else:
            text_labels = None

        encoded_text = self.batch_encode(texts)

        for key in encoded_text.keys():
            encoded_text[key] = encoded_text[key].to(device)

        bert_outputs = self.bert(**encoded_text, 
                                labels=text_labels, 
                                return_dict=True, 
                                output_hidden_states=True)
        
        prob = self.detection(bert_outputs.hidden_states[-1])

        if text_labels is None:
            outputs = (prob, bert_outputs.logits)
        else:
            det_labels = det_labels.to(device)
            det_loss_fct = FocalLoss(num_labels=None, activation_type='sigmoid').cuda()

            active_loss = encoded_text['attention_mask'].view(-1, prob.shape[1]) == 1

            active_probs = prob.view(-1, prob.shape[1])[active_loss]

            active_labels = det_labels[active_loss]

            det_loss = det_loss_fct(active_probs, active_labels.float())

            loss = self.w * bert_outputs.loss + (1 - self.w) * det_loss

            outputs = (loss, self.sigmoid(prob).squeeze(-1), bert_outputs.logits)

        return outputs
    
    def predict(self, texts, device):
        inputs = self.batch_encode(texts)

        with torch.no_grad():
            outputs = self.forward(texts, device=device)

            y_hat = torch.argmax(outputs[1], dim=-1)

            expand_text_lens = torch.sum(inputs['attention_mask'], dim=-1)

        res = []

        for t_len, _y_hat in zip(expand_text_lens, y_hat):
            t_len = t_len.long()

            res.append(self.tokenizer.decode(_y_hat[1: t_len]).replace(' ', ''))

        return res
