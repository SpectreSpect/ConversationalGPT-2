# ConversationalGPT-2

![Static Badge](https://img.shields.io/badge/Python-%237F52FF?style=for-the-badge&logo=Python&logoColor=white)
![Static Badge](https://img.shields.io/badge/PyTorch-%23FE7B7B?style=for-the-badge&logo=PyTorch&logoColor=white)
![Static Badge](https://img.shields.io/badge/PyTorchXLA-%234DA651?style=for-the-badge&logo=PyG&logoColor=white)
![Static Badge](https://img.shields.io/badge/CometML-%234A2296?style=for-the-badge&logo=bitcomet&logoColor=white)
![Static Badge](https://img.shields.io/badge/TPU-%2325A162?style=for-the-badge&logo=turbo&logoColor=white)
![Static Badge](https://img.shields.io/badge/Git-%23EA330E?style=for-the-badge&logo=git&logoColor=white)

The language model, **ConversationalGPT-2**, was built from scratch using [PyTorch](https://pytorch.org/) and follows a transformer architecture similar to modern language models.

Initially, the model was trained on the [fineweb dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) to predict the most likely continuation of text. It was then fine-tuned on the [DailyDialog dataset](https://huggingface.co/datasets/li2017dailydialog/daily_dialog) to enhance its ability to engage in natural, coherent conversations.

With over 100 million parameters, training was conducted on the [Kaggle platform](https://www.kaggle.com/):
- [ConvGPT2-Finetune Notebook](https://www.kaggle.com/code/spectrespect/convgpt2-finetune)

The trained model is also available on Kaggle and can be found here:
- [ConversationalGPT-2 Model](https://www.kaggle.com/models/spectrespect/convgpt2)

## Running the Model  
To run the project, follow these steps:

1. Download the repository to your local machine.
2. Run the Python script `chat_interface.py`.
   - If the necessary models are not already present, they will automatically be downloaded.

# Preview
This section provides a demonstration of how the model responds in conversations. You can view a sample dialogue in action via the GIF below:

<div align="center">
  <img src="https://github.com/user-attachments/assets/f218d7cc-9018-4592-89fb-61a5f65e7538" alt="model preview">
</div>
</br>
The dialogues presented here illustrate different aspects of the model's conversational abilities. Each interaction was generated by the model, and you can reproduce them by downloading the project and asking similar questions. <br>

### Dialogue 1: General Conversational Skills
This dialogue highlights the model's ability to engage in casual, everyday conversations.
| Dialog 1 |
| ----------------------------------------------------------------------------------------- | 
|You: Hello. What's your name?<br> |
|AI:   My name is Kate .<br> | 
|You: Nice to meet you, Kate! Where are you from?<br> |
|AI:   I'm from Texas . <br> |
|You: Oh, so you're from the US, right?<br> |
|AI:   Yes , I am . <br> |
|You: Sounds great. What's your favorite fruit?<br> |
|AI:   I love apples . <br> |
|You: Apples are yummy.<br> |
|AI:   I love apples . <br> |
|You: Have you tried pineapples?<br> |
|AI:   Yes , I have . <br> |
|You: What color are apples?<br> |
|AI:   I don't know . <br> |
|You: What color is the sky? <br> |
|AI:   It's blue .<br> |

### Dialogue 2: Memory and Fact Recall
This dialogue demonstrates the model's ability to remember and recall information over multiple interactions.
| Dialog 2 |
| ----------------------------------------------------------------------------------------- | 
|You: Hello, how are you?   |
|AI:   I'm fine . I'm just worried about my friends . |
|You: Remember, the marble is blue.|
|AI:   Are you sure ? |
|You: Yes|
|AI:   I'm sure . I'm just worried about my friends . I'm just worried about my friends . |
|You: What color is the marble?|
|AI:   It's blue . |
|You: Correct! Now, remember, the car is white.|
|AI:   But I'm not sure . |
|You: How was your day?|
|AI:   I was fine . |
|You: What's your name?|
|AI:   It's Mal . |
|You: What color is the car?|
|AI:   It's white .|

### Dialogue 3: Understanding the World
This dialogue illustrates the model’s grasp of basic facts about the world, such as object speeds, languages spoken in different countries, and what scientists do.
| Dialog 3 |
| ----------------------------------------------------------------------------------------- | 
|You: What is faster: a car or a human? |
|AI:   A car is faster than a human . |
|You: What do scientists do?|
|AI:   They study the environment . |
|You: What language is spoken in the US?|
|AI:   We speak English . We study it .|
