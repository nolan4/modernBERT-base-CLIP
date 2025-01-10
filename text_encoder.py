from transformers import AutoTokenizer, ModernBertModel
import torch
import torch.nn as nn
import torch.optim as optim
import pdb

class modernBERT(nn.Module):
    def __init__(self, model_name="answerdotai/ModernBERT-base"):
        super(modernBERT, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = ModernBertModel.from_pretrained(model_name)

    def forward(self, inputs):
        # inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        outputs = self.bert(**inputs)

        return outputs.last_hidden_state # logits

# Example training loop
if __name__ == "__main__":
    model = modernBERT("answerdotai/ModernBERT-base")

    texts = ["Potato's no name for a dog"]
    text_inputs = {"input_ids": model.tokenizer(texts)}
    output = model(text_inputs)

    print(output[0].shape)
