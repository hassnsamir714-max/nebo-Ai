"""Arabic chat interface for the locally trained Nebo character model."""

from pathlib import Path

import streamlit as st
import torch

from model import TinyGPT

ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "nebo_model.pt"

st.set_page_config(page_title="Nebo AI", page_icon="☁️", layout="centered")
st.title("☁️ نبو — Nebo AI")
st.caption("نموذج لغوي تجريبي صغير مُدرّب محلياً من الصفر باستخدام PyTorch.")
st.info("هذا نموذج تعليمي حرفي صغير، وليس نموذجاً ضخماً مثل ChatGPT. جودة الإجابات تعتمد على بيانات التدريب.")


@st.cache_resource
def load_model(path: str, modified: float):
    del modified
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    config = checkpoint["config"]
    model = TinyGPT(**config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    vocabulary = checkpoint["vocabulary"]
    return model, {character: index for index, character in enumerate(vocabulary)}, vocabulary, device


if not CHECKPOINT.exists():
    st.warning("لم يتم تدريب النموذج بعد. ثبّت المتطلبات ثم شغّل `python train.py`.")
    st.stop()

model, stoi, vocabulary, device = load_model(str(CHECKPOINT), CHECKPOINT.stat().st_mtime)
itоs = {index: character for index, character in enumerate(vocabulary)}

if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("اكتب رسالة لنبو...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        known = [stoi[character] for character in prompt if character in stoi]
        if not known:
            reply = "النموذج لم يتعلم أحرف هذه الرسالة. أضف نصوصاً مناسبة إلى بيانات التدريب ثم أعد التدريب."
        else:
            with st.spinner("نبو يفكر..."):
                context = torch.tensor([known], dtype=torch.long, device=device)
                with torch.no_grad():
                    generated = model.generate(context, max_new_tokens=220, temperature=0.8)[0].tolist()
                reply = "".join(itоs[index] for index in generated[len(known):]).strip()
                if not reply:
                    reply = "لم أتمكن من توليد نص؛ جرّب سؤالاً آخر."
        st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
