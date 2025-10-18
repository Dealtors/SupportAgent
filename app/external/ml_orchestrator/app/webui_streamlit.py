import streamlit as st
from app.orchestrator import process_ticket

st.set_page_config(page_title="Ticket Orchestrator", page_icon="🧠", layout="centered")

st.title("🧠 Ticket Orchestrator (Qwen2-7B)")
st.caption("Мини-демо без серверной магии, зато честно.")

q = st.text_area("Введите тикет или обращение", height=150, placeholder="Ошибка деплоя Docker...")

if st.button("Сформировать ответ"):
    if not q.strip():
        st.warning("Введите текст запроса.")
    else:
        with st.spinner("Думаю..."):
            label, conf, answer = process_ticket(q)
        st.subheader("Класс")
        st.write(f"{label}  (p={conf:.2f})")
        st.subheader("Ответ")
        st.write(answer)
