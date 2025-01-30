# 🧠 DistilBERT Emotion Classifier  

## 🚀 Overview  
This project uses **DistilBERT** to extract sentence embeddings from the **Hugging Face "emotion" dataset** and trains a **Logistic Regression classifier** to predict emotions.  

## 📌 Features  
✅ Loads and preprocesses the **emotion dataset**  
✅ Tokenizes text using **DistilBERT tokenizer**  
✅ Extracts **hidden state embeddings** from DistilBERT  
✅ Trains a **Logistic Regression model** for classification  
✅ Evaluates model performance on the validation set  

## 📂 Dataset  
- The dataset comes from Hugging Face: [`emotion`](https://huggingface.co/datasets/emotion)  
- Contains **text samples** labeled with emotions like **joy, sadness, anger, etc.**  

## 🛠️ Installation & Requirements  

### **1️⃣ Install Dependencies**  

🎯 Project Workflow
1️⃣ Load the Emotion Dataset
2️⃣ Tokenize the text using DistilBERT
3️⃣ Extract hidden states (sentence embeddings) from DistilBERT
4️⃣ Store embeddings in the dataset
5️⃣ Train a Logistic Regression model on embeddings
6️⃣ Evaluate the model on validation data

🚀 Model Performance
Model used: DistilBERT + Logistic Regression
Evaluation metric: Accuracy on validation set
📊 Expected Results
With proper tuning, you can achieve ~90% accuracy on emotion classification.
Possible improvements:
Fine-tuning DistilBERT instead of using static embeddings.
Trying other classifiers like SVM or Neural Networks.
🤖 Future Enhancements
🔹 Fine-tune DistilBERT on the emotion dataset
🔹 Experiment with different ML classifiers
🔹 Deploy as an API using FastAPI/Flask

