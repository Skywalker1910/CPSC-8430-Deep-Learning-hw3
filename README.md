# README: BERT-Base for Question Answering on Spoken-SQuAD
# CPSC-8430-Deep-Learning-hw3
CPSCE 8430 Deep Learning HomeWork 3 

## Overview
This project involves implementing a **BERT-Base** model for **Question Answering (QA)** on the **Spoken-SQuAD** dataset. The aim was to leverage the power of BERT-Base to moderately improve the **F1 Score** and **Word Error Rate (WER)** over smaller models, while balancing training efficiency. This README will guide you through setting up the environment, training the model, and evaluating its performance.

## Dataset
**Spoken-SQuAD** is a speech-transcribed version of the original SQuAD dataset, containing paragraphs, questions, and corresponding answers. The dataset includes inherent transcription errors, adding an additional layer of complexity to the QA task.

## Prerequisites
- **Python 3.8+**
- **PyTorch** (version 1.7+)
- **Transformers library** from Hugging Face
- **CUDA** (optional, for GPU acceleration)
- **Requirements**: Install the required packages using the command below:

```bash
pip install -r requirements.txt
```

**Requirements.txt** should contain:
- torch
- transformers
- tqdm
- numpy
- datasets

## Data Preprocessing
The **Spoken-SQuAD** dataset was preprocessed using **BertTokenizerFast**. Key preprocessing steps included:
- Tokenizing the context and questions.
- Setting the **max sequence length** to **384** tokens and **document stride** to **128** for overlapping contexts.
- Generating offset mappings to link character positions to token positions.
- Filtering out examples where the answers were truncated.

## Model Details
We used **BERT-Base** (`bert-base-uncased`), which has **12 encoder layers** and **110 million parameters**. This model provides a good balance between complexity and feasibility, offering significantly more power than smaller BERT variants while being manageable for training.

### Customization
- **Fine-Tuning**: The last **4 encoder layers** of the model were unfrozen for fine-tuning to adapt BERT to the Spoken-SQuAD QA task.
- **QA Head**: A custom question-answering head was added to predict the start and end tokens of the answer spans.

## Training Setup
### Hyperparameters
The model was trained using the following parameters:
- **Batch Size**: 12 (effective batch size 24 using gradient accumulation).
- **Learning Rate**: **3e-5**, with a **linear decay schedule** and **warm-up**.
- **Epochs**: 4, with **early stopping** if validation loss didn’t improve for 2 consecutive checks.
- **Optimizer**: `AdamW` with a **weight decay** of **0.01** to help generalization.

### Techniques Used
- **AMP (Automatic Mixed Precision)**: To reduce memory usage and speed up training by using lower precision where possible.
- **Gradient Checkpointing**: To save GPU memory by recomputing activations during the backward pass.
- **Gradient Accumulation**: Gradients were accumulated over 2 steps to simulate a larger batch size without exhausting memory.

## Training Instructions
1. **Prepare Dataset**: Ensure the Spoken-SQuAD dataset is downloaded and properly formatted.
2. **Run Training Script**: Use the following command to train the model:
   
   ```bash
   python train.py --dataset_path <path_to_spoken_squad> --epochs 4 --batch_size 12 --learning_rate 3e-5
   ```
3. **Training Monitoring**: Training progress can be monitored using the **tqdm** progress bar integrated into the script.

## Evaluation
The model was evaluated on a validation set using the following metrics:
- **Word Error Rate (WER)**: Measures the effect of transcription errors on answer accuracy.
- **F1 Score**: Evaluates the overlap between predicted and true answers.
- **Exact Match (EM)**: Checks if the predicted answers match exactly with the reference answers.

### Evaluation Results
- **WER**: **1.85**
- **F1 Score**: **63.5%**
- **EM**: **40.1%**

The results showed moderate improvements over smaller models, indicating that BERT-Base was able to better manage the complexity of spoken language QA.

## Recommendations for Future Work
- **Unfreeze More Layers**: Consider unfreezing additional layers for more extensive fine-tuning.
- **Use a Larger Model**: If computational resources are available, using **BERT-Large** could yield further performance gains.
- **Train on Full Dataset**: Training on the entire Spoken-SQuAD dataset could improve model generalization.
- **More Epochs**: Training for **6-8 epochs** may allow the model to learn more intricate patterns.

## Conclusion
This implementation demonstrates the power of **BERT-Base** for QA on the Spoken-SQuAD dataset, achieving moderate performance improvements in **F1 Score** and **WER**. Leveraging techniques like AMP, gradient accumulation, and selective fine-tuning allowed the training to be both effective and manageable. Future enhancements could include unfreezing more layers or switching to a larger model.

## Folder Structure
```
.
├── data
│   └── spoken_squad.json  # Preprocessed dataset
├── models
│   └── bert_base_model.pt # Trained model weights
├── train.py               # Training script
├── evaluate.py            # Evaluation script
├── requirements.txt       # Dependencies
├── README.md              # Project documentation
└── utils
    └── preprocessing.py   # Preprocessing functions
```

## License
This project is licensed under the MIT License.

## Contact
For questions or contributions, please contact **[Your Name]** at **[your.email@example.com]**.


