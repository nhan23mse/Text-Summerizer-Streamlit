import asyncio
import streamlit as st 
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Fix lỗi asyncio (RuntimeError: no running event loop)
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Hiển thị tiêu đề trang
st.title("📜 AI Text Summarizer")

# Load mô hình và tokenizer (dùng 't5-small')
@st.cache_resource()
def load_model():
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small", return_dict=True)
    return tokenizer, model

tokenizer, model = load_model()

# Ô nhập văn bản
text_input = st.text_area("📝 Nhập văn bản cần tóm tắt:", height=200)

if st.button("✨ Tóm tắt ngay"):
    if text_input.strip():
        with st.spinner("🔄 Đang xử lý..."):
            # Mã hóa văn bản
            tokenized_text = tokenizer.encode_plus(
                text_input, return_attention_mask=True, return_tensors="pt"
            )

            # Sinh văn bản tóm tắt
            summary_ids = model.generate(
                input_ids=tokenized_text["input_ids"],
                attention_mask=tokenized_text["attention_mask"],
                max_length=150,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True
            )

            # Giải mã kết quả
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        # Hiển thị kết quả
        st.subheader("📌 Văn bản đã tóm tắt:")
        st.success(summary)
    else:
        st.warning("⚠️ Vui lòng nhập văn bản trước khi tóm tắt!")

# Thêm chú thích
st.markdown("---")
st.caption("🚀 Được hỗ trợ bởi [Hugging Face Transformers](https://huggingface.co/).")
