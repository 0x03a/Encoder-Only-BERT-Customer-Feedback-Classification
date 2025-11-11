import streamlit as st
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import plotly.graph_objects as go
import plotly.express as px

# Page config
st.set_page_config(
    page_title="BERT Sentiment Analyzer",
    page_icon="🎭",
    layout="wide"
)

# Title
st.title("🎭 BERT Sentiment Analysis Dashboard")
st.markdown("---")

# Load model and tokenizer (cached)
@st.cache_resource
def load_model():
    try:
        model_path = '/home/inshal/Downloads/sentiment/Encoder-Only-BERT-Customer-Feedback-Classification/bert_sentiment_model'
        tokenizer = BertTokenizerFast.from_pretrained(model_path)
        model = BertForSequenceClassification.from_pretrained(model_path)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        return model, tokenizer, device
    except:
        st.error("❌ Model not found! Please train the model first.")
        return None, None, None

model, tokenizer, device = load_model()

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    st.markdown("---")
    
    page = st.radio(
        "Select Page:",
        ["🔮 Predict Sentiment"]
    )
    
    st.markdown("---")
    st.markdown("### 📌 Model Info")
    st.info(f"""
    **Model:** BERT-base-uncased  
    **Classes:** Positive, Negative  
    **Device:** {device if device else 'N/A'}
    """)

# Prediction function
def predict_sentiment(text, model, tokenizer, device):
    if not text.strip():
        return None, None, None
    
    encoding = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(logits, dim=1).item()
    
    sentiment = 'Positive' if pred == 1 else 'Negative'
    confidence = probs[0][pred].item()
    prob_neg = probs[0][0].item()
    prob_pos = probs[0][1].item()
    
    return sentiment, confidence, (prob_neg, prob_pos)

# PAGE 1: Predict Sentiment
if page == "🔮 Predict Sentiment":
    st.header("🔮 Real-Time Sentiment Prediction")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Enter Text")
        user_input = st.text_area(
            "Type or paste customer feedback:",
            height=150,
            placeholder="e.g., This product exceeded my expectations!"
        )
        
        predict_btn = st.button("🚀 Analyze Sentiment", type="primary")
    
 
    
    
    if predict_btn and user_input and model:
        with st.spinner("Analyzing..."):
            sentiment, confidence, probs = predict_sentiment(user_input, model, tokenizer, device)
            
            if sentiment:
                st.markdown("---")
                st.subheader("📊 Results")
                
                # Result display
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Sentiment", sentiment)
                
                with col2:
                    st.metric("Confidence", f"{confidence*100:.2f}%")
                
                with col3:
                    emoji = "😊" if sentiment == "Positive" else "😞"
                    st.metric("Status", emoji)
                
                # Probability chart
                st.subheader("📈 Probability Distribution")
                fig = go.Figure(data=[
                    go.Bar(
                        x=['Negative', 'Positive'],
                        y=[probs[0]*100, probs[1]*100],
                        marker_color=['#FF6B6B', '#4ECDC4'],
                        text=[f"{probs[0]*100:.1f}%", f"{probs[1]*100:.1f}%"],
                        textposition='auto'
                    )
                ])
                fig.update_layout(
                    yaxis_title="Probability (%)",
                    showlegend=False,
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

# # PAGE 2: Model Evaluation
# elif page == "📊 Model Evaluation":
#     st.header("📊 Model Performance Metrics")
    
#     if model:
#         st.info("⚠️ Load your validation data to see evaluation metrics")
        
#         # File uploader for validation data
#         uploaded_file = st.file_uploader("Upload validation predictions CSV", type=['csv'])
        
#         if uploaded_file:
#             df_eval = pd.read_csv(uploaded_file)
            
#             if 'true_label' in df_eval.columns and 'predicted_label' in df_eval.columns:
#                 true_labels = df_eval['true_label'].values
#                 pred_labels = df_eval['predicted_label'].values
                
#                 # Calculate metrics
#                 accuracy = accuracy_score(true_labels, pred_labels)
#                 f1 = f1_score(true_labels, pred_labels, average='weighted')
#                 cm = confusion_matrix(true_labels, pred_labels)
                
#                 # Display metrics
#                 col1, col2, col3 = st.columns(3)
                
#                 with col1:
#                     st.metric("🎯 Accuracy", f"{accuracy*100:.2f}%")
                
#                 with col2:
#                     st.metric("📈 F1-Score", f"{f1:.4f}")
                
#                 with col3:
#                     st.metric("📝 Samples", len(true_labels))
                
#                 st.markdown("---")
                
#                 # Confusion Matrix
#                 col1, col2 = st.columns(2)
                
#                 with col1:
#                     st.subheader("Confusion Matrix")
#                     fig_cm = px.imshow(
#                         cm,
#                         labels=dict(x="Predicted", y="Actual", color="Count"),
#                         x=['Negative', 'Positive'],
#                         y=['Negative', 'Positive'],
#                         text_auto=True,
#                         color_continuous_scale='Blues'
#                     )
#                     st.plotly_chart(fig_cm, use_container_width=True)
                
#                 with col2:
#                     st.subheader("Classification Report")
#                     report = classification_report(
#                         true_labels, 
#                         pred_labels, 
#                         target_names=['Negative', 'Positive'],
#                         output_dict=True
#                     )
#                     report_df = pd.DataFrame(report).transpose()
#                     st.dataframe(report_df.style.highlight_max(axis=0))
#             else:
#                 st.error("CSV must contain 'true_label' and 'predicted_label' columns")
#         else:
#             # Show mock metrics as example
#             st.markdown("### 📋 Sample Metrics Format")
#             st.code("""
# Expected CSV format:
# true_label,predicted_label
# 0,0
# 1,1
# 0,1
# ...
#             """)
#     else:
#         st.error("Model not loaded!")

# # PAGE 3: Batch Prediction
# elif page == "📁 Batch Prediction":
#     st.header("📁 Batch Sentiment Prediction")
    
#     if model:
#         st.markdown("Upload a CSV file with customer feedback for batch processing")
        
#         uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
        
#         if uploaded_file:
#             df_batch = pd.read_csv(uploaded_file)
            
#             st.subheader("📄 Uploaded Data Preview")
#             st.dataframe(df_batch.head(10))
            
#             # Select text column
#             text_column = st.selectbox("Select text column:", df_batch.columns)
            
#             if st.button("🚀 Process Batch", type="primary"):
#                 with st.spinner(f"Processing {len(df_batch)} rows..."):
#                     sentiments = []
#                     confidences = []
                    
#                     progress_bar = st.progress(0)
                    
#                     for idx, text in enumerate(df_batch[text_column]):
#                         sentiment, confidence, _ = predict_sentiment(
#                             str(text), model, tokenizer, device
#                         )
#                         sentiments.append(sentiment if sentiment else 'Unknown')
#                         confidences.append(confidence if confidence else 0.0)
                        
#                         progress_bar.progress((idx + 1) / len(df_batch))
                    
#                     df_batch['Predicted_Sentiment'] = sentiments
#                     df_batch['Confidence'] = [f"{c*100:.2f}%" for c in confidences]
                    
#                     st.success("✅ Batch processing complete!")
                    
#                     # Results
#                     st.subheader("📊 Results")
#                     st.dataframe(df_batch)
                    
#                     # Statistics
#                     col1, col2, col3 = st.columns(3)
                    
#                     with col1:
#                         pos_count = (df_batch['Predicted_Sentiment'] == 'Positive').sum()
#                         st.metric("Positive", pos_count)
                    
#                     with col2:
#                         neg_count = (df_batch['Predicted_Sentiment'] == 'Negative').sum()
#                         st.metric("Negative", neg_count)
                    
#                     with col3:
#                         avg_conf = np.mean([float(c.strip('%'))/100 for c in df_batch['Confidence']])
#                         st.metric("Avg Confidence", f"{avg_conf*100:.2f}%")
                    
#                     # Download button
#                     csv = df_batch.to_csv(index=False)
#                     st.download_button(
#                         label="📥 Download Results CSV",
#                         data=csv,
#                         file_name="sentiment_predictions.csv",
#                         mime="text/csv"
#                     )
                    
#                     # Visualization
#                     st.subheader("📈 Sentiment Distribution")
#                     sentiment_counts = df_batch['Predicted_Sentiment'].value_counts()
#                     fig = px.pie(
#                         values=sentiment_counts.values,
#                         names=sentiment_counts.index,
#                         color=sentiment_counts.index,
#                         color_discrete_map={'Positive': '#4ECDC4', 'Negative': '#FF6B6B'}
#                     )
#                     st.plotly_chart(fig, use_container_width=True)
#     else:
#         st.error("Model not loaded!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with ❤️ using BERT & Streamlit</p>
</div>
""", unsafe_allow_html=True)