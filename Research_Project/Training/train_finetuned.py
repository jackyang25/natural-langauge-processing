#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#pip install transformers datasets


# In[3]:


from datasets import load_dataset
from transformers import GPT2Tokenizer
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from transformers import GPTNeoForCausalLM, Trainer, TrainingArguments

model = GPTNeoForCausalLM.from_pretrained("Pretrained_GPTNeo/")
tokenizer = GPT2Tokenizer.from_pretrained("Pretrained_GPTNeo/")
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    output = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=35, return_tensors="pt")
    output["labels"] = output["input_ids"].clone()
    return output  

dataset = load_dataset("text", data_files={"train": "Datasets/merged_dataset.txt"})
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text'])


# In[ ]:


import torch
if torch.cuda.is_available():
    model.to('cuda')


# In[6]:



training_args = TrainingArguments(
    output_dir="output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,
    learning_rate=5e-6,
    fp16=True,
    weight_decay=0.01,
    logging_dir="logs",
    logging_steps=10,
    save_steps=1000,
    save_total_limit=1,
)


# In[4]:



# In[5]:


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
)
trainer.train()


# In[ ]:


model.save_pretrained("FineTuned_Models/New/")
tokenizer.save_pretrained("FineTuned_Models/New/")

