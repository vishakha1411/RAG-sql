import gradio as gr
from chatbot.chatbot_backend import ChatBot
from utils.ui_settings import UISettings


def format_chatbot_output(chatbot, input_text):
    """
    Formats the chatbot output into Gradio's Chatbot-compatible format.
    """
    user_message = {"role": "user", "content": input_text}
    chatbot.append(user_message)
    return chatbot


# Define the Gradio Interface
with gr.Blocks() as demo:
    with gr.Tabs():
        with gr.TabItem("AgentGraph"):
            # First Row: Chatbot display
            chatbot = gr.Chatbot(
                [], elem_id="chatbot", height=500, type="messages"
            )

            # Second Row: Text input
            with gr.Row():
                input_txt = gr.Textbox(
                    lines=3,
                    placeholder="Enter text and press Enter",
                    container=True,
                )

            # Third Row: Submit and Clear Buttons
            with gr.Row():
                submit_btn = gr.Button("Submit")
                clear_btn = gr.ClearButton([input_txt, chatbot])

            # Processing logic
            input_txt.submit(
                fn=ChatBot.respond,
                inputs=[chatbot, input_txt],
                outputs=[input_txt, chatbot],
                queue=False,
            )

            submit_btn.click(
                fn=ChatBot.respond,
                inputs=[chatbot, input_txt],
                outputs=[input_txt, chatbot],
                queue=False,
            )

if __name__ == "__main__":
    demo.launch(share=True)
