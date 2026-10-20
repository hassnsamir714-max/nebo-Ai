import streamlit as st
from transformers import pipeline

@st.cache_resource
def load_model():
    return pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct")

pipe = load_model()

st.title("نبو (Nebo) - أول AI عراقي")
st.caption("صنع بواسطة المبرمج حسن سمير")

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
        messages = [{"role": "system", "content": "أنت مساعد ذكي ومهذب اسمك (نبو)، صنعك المبرمج حسن سمير. تتكلم بلهجة عراقية واضحة."}]
        for msg in st.session_state.messages:
            messages.append({"role": msg["role"], "content": msg["content"]})
        
        response = pipe(messages, max_new_tokens=150, do_sample=True, temperature=0.7)
        reply = response[0]['generated_text'][-1]['content']
        
        st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
