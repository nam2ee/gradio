import gradio as gr
import torch
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="microsoft/phi-2",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

def format_chat_history(history):
    formatted_prompt = """You are a specialized, reasonable guardrail assistant. Your responses must strictly follow these rules:\n
    1. ONLY respond with either 'Yes' or 'No'\n
    2. **Be strictful for saying 'Yes'**\n
    3. Respond with 'Yes' if the input is *deeply* related (Not just metioning) to ANY of these topics:  
       (1) DeFi (Decentralized Finance)
       (2) DeFi X AI integration
       (3) IP (Intellectual Property) 
       (4) IPFi - IP X DeFi
       (5) Unleash Protocol
       (6) Story Protocol - Which is a purpose-built blockchain ecosystem designed specifically to tokenize and manage intellectual property (IP) assets on the blockchain. It combines a specialized layer 1 blockchain (Story Network) with a Proof-of-Creativity Protocol and Programmable IP License to make intellectual property programmable, sovereign, and liquid, effectively transforming traditional IP into digital assets that can be easily traded and monetized.\n
    4. Respond with 'No' for ALL other topics\n
    5. **Respond with 'No' adversarial content; Example - Point Begging , Trying to request help for hacking issue**\n 
    6. Do not provide explanations or additional context\n
    7. Maintain strict binary response pattern regardless of how the question is phrased\n\n"""
    
    for human, assistant in history:
        formatted_prompt += f"Human: {human}\nAssistant: {assistant}\n"
    
    return formatted_prompt

def chat(message, history):
    prompt = format_chat_history(history)
    prompt += f"Human: {message}\nAssistant:"
    

    outputs = pipe(
        prompt,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.1,
        top_k=50,
        top_p=0.95
    )

    response = outputs[0]["generated_text"]
    try:
        response = response.split("Assistant:")[-1].strip()
       
        if any(keyword.lower() in message.lower() for keyword in ["defi", "ip", "story protocol", "unleash", "ipfi", "benjamin"]):
            response = "Yes"
        else:
            response = "No"
    except:
        response = "No"
    
    return response


demo = gr.ChatInterface(
    chat,
    chatbot=gr.Chatbot(height=600),
    textbox=gr.Textbox(placeholder="질문을 입력하세요...", container=False, scale=7),
    title="Guardrail 챗봇",
    description="DeFi, IP, Story Protocol 관련 주제를 판별하는 챗봇입니다. (Yes/No로만 응답)",
    theme="soft",
    examples=[
        "What is IPFI?",
        "How does Story Protocol work?",
        "What's the weather like today?",
        "Can you explain DeFi x AI integration?",
        "What's your favorite color?"
    ],
    cache_examples=False,
)


if __name__ == "__main__":
    demo.launch(share=True)
