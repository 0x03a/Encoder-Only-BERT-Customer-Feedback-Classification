


# 🎭 BERT Sentiment Analysis Dashboard  

An interactive **Streamlit web app** for real-time **sentiment analysis** using a fine-tuned **BERT** model.  
This dashboard allows users to predict, visualize, and evaluate sentiment (Positive/Negative) from customer feedback in real time or batch mode.  

---

## 🚀 Features  

✅ **Real-Time Sentiment Prediction**  
Analyze single text inputs with live BERT-based predictions and probability charts.  

📊 **Model Evaluation**  
Upload validation results to view metrics like Accuracy, F1-score, Confusion Matrix, and Classification Report.  

📁 **Batch Prediction**  
Upload a CSV file containing customer feedback and process hundreds of texts instantly with downloadable results.  

💾 **Offline Model Loading**  
Loads your fine-tuned BERT model and tokenizer directly from local files (no Hugging Face API dependency).  

🎨 **Interactive Visualizations**  
Powered by **Plotly** and **Streamlit**, offering a clean and responsive interface.  

---

## 🧠 Tech Stack  

| Component | Description |
|------------|-------------|
| **Language** | Python 3 |
| **Framework** | Streamlit |
| **Model** | BERT (`bert-base-uncased`) |
| **Libraries** | `transformers`, `torch`, `pandas`, `scikit-learn`, `plotly`, `numpy` |
| **Visualization** | Plotly charts + Streamlit UI |

---

## ⚙️ Installation  

### 1️⃣ Clone the repository  
```bash
git clone https://github.com/<your-username>/bert-sentiment-dashboard.git
cd bert-sentiment-dashboard
````

### 4️⃣ Place your trained model in this path

```
/home/<username>/Downloads/sentiment/bert_sentiment_model
```

It should contain:

```
config.json
pytorch_model.bin or model.safetensors
tokenizer_config.json
vocab.txt
```

---

## 🧩 Usage

### ▶️ Run the Streamlit app

```bash
streamlit run app.py
```

### 🖥️ Open in browser

```
http://localhost:8501
```

---

## 🧭 Project Structure

```
📂 bert-sentiment-dashboard
│
├── app.py                         # Main Streamlit app
├── requirements.txt               # Dependencies
├── /model/                        # Folder containing trained BERT model files
│   ├── config.json
│   ├── vocab.txt
│   ├── tokenizer_config.json
│   └── model.safetensors
└── README.md                      # Project documentation
```

---

## 📸 Screenshots

| Page                     | Description                                          |
| ------------------------ | ---------------------------------------------------- |
| **🔮 Predict Sentiment** | Real-time input with confidence meter and emojis     |
| **📊 Model Evaluation**  | Metrics, Confusion Matrix, and Classification Report |
| **📁 Batch Prediction**  | Bulk analysis with downloadable CSV output           |

> *(Add screenshots here for better presentation!)*

---

## 📈 Example CSV Formats

**For Model Evaluation:**

```csv
true_label,predicted_label
0,0
1,1
0,1
```

**For Batch Prediction:**

```csv
customer_feedback
"This product is amazing!"
"Terrible quality, not worth it."
"Good value for the price."
```

---

## 🧾 Sample Output

| Text                   | Predicted Sentiment | Confidence |
| ---------------------- | ------------------- | ---------- |
| I love this product!   | Positive            | 97.6%      |
| Worst experience ever. | Negative            | 94.3%      |

---

## ❤️ Acknowledgements

* [Hugging Face Transformers](https://huggingface.co/transformers/)
* [PyTorch](https://pytorch.org/)
* [Streamlit](https://streamlit.io/)
* [Plotly](https://plotly.com/python/)

---

## 📜 License

This project is licensed under the **MIT License**.
Feel free to use, modify, and distribute it.

---
