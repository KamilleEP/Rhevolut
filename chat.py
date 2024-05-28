import streamlit as st
import requests


ENDPOINT_URL = "https://qhkp4403vl.execute-api.us-east-1.amazonaws.com"

st.title("💬 Chatbot")

models = {
    "claude-v3-sonnet": "anthropic.claude-3-sonnet-20240229-v1:0",
    "claude-v3-haiku": "anthropic.claude-3-haiku-20240307-v1:0",
    "claude-v2.1": "anthropic.claude-v2:1",
    "claude-v2": "anthropic.claude-v2",
    "claude-instant": "anthropic.claude-instant-v1",
}

model_id_options = list(models.keys())

selected_model_id = st.selectbox("Modelo", model_id_options)
show_citations = st.checkbox("Mostrar citações")

chosen_model_id = models[selected_model_id]

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Como posso ajudar você?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    request_data = {
        "question": prompt,
        "modelId": chosen_model_id,
    }

    try:
        response = requests.post(ENDPOINT_URL + "/chat", json=request_data)
        if response.status_code == 200:
            response_data = response.json()
            response_text = response_data["response"]

            st.session_state.messages.append(
                {"role": "assistant", "content": response_text}
            )
            st.chat_message("assistant").write(response_text)

            if show_citations:
                if "citation" in response_data:
                    for citation_block in response_data["citation"]:
                        citation_text = citation_block["content"]["text"]
                        document_uri = citation_block["location"]["s3Location"]["uri"]
                        document_name = citation_block["metadata"]["name"]

                        st.markdown(f"**Citation from document:** {document_name}")
                        st.markdown(f"**Document URI:** {document_uri}")
                        st.markdown(f"**Content:**\n{citation_text}")
                        st.markdown("---")
        else:
            st.error("Falhou em obter uma resposta da api.")
    except Exception as e:
        st.error(f"Ocorreu um erro ao tentar enviar o pedido.")
