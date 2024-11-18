# I. Light Weight Fine-Tuning
1. Parameter-Efficient Fine-Tuning (PEFT) definitions:
The light weight fine-tuning is the common methods to adapt the pretrained model to a specific tasks with small additional training parameters.
This method can help to save the computional resources and training time. Because only a small subset of model parameter are adjusted, instead of the entire model.
This method is suitable to adapt the pretrained model with limited resource but still can take advantages with the powerful pretrained model.

2. Project flow
- Load a pre-trained model and evaluate its performance.
- Perform parameter-efficient fine-tuning using the pre-trained model.
- Perform inference using the fine-tuned model and compare its performance to the original model.

# II. Loading and Evaluating the Foundation Model
- Model: GPT-2ForSequenceClassification (gpt2) is using for transformer for the spam analysis. The gpt2 can be used for various taks of NLP since it can capture the patterns and contexts in sentences.
- Evaluation approach: Using the Trainer evaluation. The solution is to compare the accuracy of the model with no parameters change (all weights are freezed) and fine tune some parameters. By this comparison, we can select the better model for the text classification.
- Fine-tuning dataset: The dataset using is collected from Hugging Face to check if the message is spam or not spam. Here is the link to the dataset: https://huggingface.co/datasets/ucirvine/sms_spam
- The classification accuracy of the model is 85%. This is quite good result when we do not need to change anything from the original model. Let's see if we can improve the accuracy with fine tuning model

![image](https://github.com/user-attachments/assets/61fe86b6-f659-4766-8eed-bfe6369ba05e)


# III. Performing Parameter-Efficient Fine-Tuning
Create a PEFT model from loaded model LoRA, run a training loop, and save the PEFT model weights. Here is the code:
```
model_ft = AutoModelForSequenceClassification.from_pretrained(
    "gpt2",
    num_labels = 2,
    id2label = {0: "not spam", 1: "spam"},
    label2id = {"not spam": 0, "spam": 1}
    )

model_ft.config.pad_token_id = tokenizer.pad_token_id

# Create a PEFT Config for LoRA
config = LoraConfig(r = 8, 
                    lora_alpha = 32,
                    target_modules = ['c_attn', 'c_proj'],
                    lora_dropout = 0.1,
                    bias = "none",
                    task_type=TaskType.SEQ_CLS
                )

peft_model = get_peft_model(model_ft, config)

# Create the Trainer to loop and get the best model
trainer_ft = Trainer(
                model = peft_model, 
                args = TrainingArguments(
                    output_dir = "./lora_model_output",
                    learning_rate = 2e-5,
                    per_device_train_batch_size = 32,
                    per_device_eval_batch_size = 32,
                    num_train_epochs = 3,
                    weight_decay = 0.01,
                    evaluation_strategy = "epoch",
                    save_strategy = "epoch",
                    load_best_model_at_end = True,
                    logging_dir='./logs',   
    ),
                train_dataset = tokenized_dataset["train"],
                eval_dataset = tokenized_dataset["test"],
                tokenizer = tokenizer,
                data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True, max_length=512),
                compute_metrics = compute_metrics,
)

# Evaluate
evaluation_results_peft = trainer_ft.evaluate()
print("Evaluation Results:", evaluation_results_peft)
```

![image](https://github.com/user-attachments/assets/ccef017d-a823-4b4e-936c-5393d73de6c5)


The model accuracy is 91%, which is better than original model (84%)


# IV. Performing Inference with a PEFT Model
Load the saved PEFT model weights and evaluate the performance of the trained PEFT model. Be sure to compare the results to the results from prior to fine-tuning.
Randomly pick some records from testing dataset to check the predictions
![image](https://github.com/user-attachments/assets/3c0af1bb-e779-4f95-8dda-f74d3d1128f0)

Most of the predictions are matched with the labels. For the final records, the prediction is wrong (it is actually a spam). Overall the model can recognize quite good between spam and not spam messages.

# V. Conclusion
- Using the fine tuning can help to train the new LLM mode with small number of parameters but still efficient. The weights are freezed so we can focus on the changed parameters and can save the resource for training.
- The fine tuning model have better accuracy comparing to the original model (91% vs 84%)
