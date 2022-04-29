import streamlit as st
from onnxruntime import (GraphOptimizationLevel, InferenceSession, SessionOptions)
from scipy.special import softmax
import numpy as np
from transformers import AutoTokenizer
import gdown

url = "https://drive.google.com/uc?id=1XRien4A0Lg5Lv7wlfFyQ-pb7Sh7NQSiN"
output = "model.quant.onnx"
gdown.download(url, output, quiet=False)

def create_model_for_provider(model_path, provider="CPUExecutionProvider"): 
    options = SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    session = InferenceSession(str(model_path), options, providers=[provider])
    session.disable_fallback()
    return session

class OnnxPipeline:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
    def __call__(self, query):
        model_inputs = self.tokenizer(query, return_tensors="pt")
        inputs_onnx = {k: v.cpu().detach().numpy() 
                       for k, v in model_inputs.items()}
        logits = self.model.run(None, inputs_onnx)[0][0, :]
        probs = softmax(logits)
        pred_idx = np.argmax(probs).item()
        sentiments = {0: "POSITIVE", 1: "NEUTRAL", 2: "NEGATIVE"}
        return sentiments[pred_idx], probs[pred_idx]

model_output = "model.quant.onnx"
model_ckpt = "afbudiman/indobert-classification"
onnx_quantized_model = create_model_for_provider(model_output)
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
pipe = OnnxPipeline(onnx_quantized_model, tokenizer)


def main():
    text = st.text_area(label='Put your review here')

    if st.button('Predict'):
        predict = pipe(text)
        st.write(f"It's {predict[0]} sentiment with {predict[1]} probability")
        

if __name__ == "__main__":
    st.set_page_config(page_title="Sentiment Analysis")
    st.title("Sentiment Analysis")
    main()
