import torch.nn as nn
from transformers import AutoModel


class ClassificationHead(nn.Module):
    def __init__(self, hidden_size, num_labels, dropout_rate=0.1) -> None:
        super().__init__()

        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size, num_labels)
    
    def forward(self, pooled_output):
        return self.classifier(self.dropout(pooled_output))

class StyleEncoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.bert = AutoModel.from_pretrained(config.bert_path)

        self.pitch_clf = ClassificationHead(config.bert_hidden_size, config.pitch_n_labels)
        self.speed_clf = ClassificationHead(config.bert_hidden_size, config.speed_n_labels)
        self.energy_clf = ClassificationHead(config.bert_hidden_size, config.energy_n_labels)
        self.emotion_clf = ClassificationHead(config.bert_hidden_size, config.emotion_n_labels)
        self.style_embed_proj = nn.Linear(config.bert_hidden_size, config.style_dim)

    def forward(self, input_ids, token_type_ids, attention_mask):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        ) # return a dict having ['last_hidden_state', 'pooler_output']

        pooled_output = outputs["pooler_output"]

        pitch_outputs = self.pitch_clf(pooled_output)
        speed_outputs = self.speed_clf(pooled_output)
        energy_outputs = self.energy_clf(pooled_output)
        emotion_outputs = self.emotion_clf(pooled_output)

        res = {
            "pooled_output":pooled_output,
            "pitch_outputs":pitch_outputs,
            "speed_outputs":speed_outputs,
            "energy_outputs":energy_outputs,
            "emotion_outputs":emotion_outputs,
        }

        return res