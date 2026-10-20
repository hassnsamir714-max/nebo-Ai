import streamlit as st
import requests

st.title("نبو (Nebo) - أول AI عراقي")
st.caption("صنع بواسطة المبرمج حسن سمير")

# نسحب المفتاح من إعدادات Streamlit بأمان
HF_TOKEN = st.secrets["HF_TOKEN"]
API_URL = "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-1.5B-Instruct"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("اكتب رسالتك هنا..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        payload = {
            "inputs": f"<|im_start|>system\nأنت مساعد اسمك نبو، صنعك حسن سمير. تكلم عراقي.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
            "parameters": {"max_new_tokens": 200, "temperature": 0.7}
        }
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
            reply = response.json()[0]['generated_text'].split("<|im_end|>")[-1].strip()
        except:
            reply = "عذراً يا حسن، السيرفر مشغول شوي. جرب مرة ثانية."
            
        st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
