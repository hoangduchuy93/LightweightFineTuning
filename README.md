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
- The model accuracy without changing any parameters:
![image](https://github.com/user-attachments/assets/61fe86b6-f659-4766-8eed-bfe6369ba05e)
- The classification accuracy of the model is 85%. This is quite good result when we do not need to change anything from the original model. Let's see if we can improve the accuracy with fine tuning model
