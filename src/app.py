import gradio as gr
from pyprojroot import here
from chatbot.chatbot_backend import ChatBot
from utils.ui_settings import UISettings


with gr.Blocks() as demo:
    # Each browser session has its own conversation, which the clear button restarts.
    thread_id = gr.State()

    with gr.Tabs():
        with gr.TabItem("RAG-SQL-AGENT"):

            with gr.Row() as row_one:
                chatbot = gr.Chatbot(
                    [],
                    elem_id="chatbot",
                    bubble_full_width=False,
                    height=500,
                    avatar_images=(str(here("images/Avatar.png")), None),

                )
                chatbot.like(UISettings.feedback, None, None)

            with gr.Row():
                input_txt = gr.Textbox(
                    lines=3,
                    scale=8,
                    placeholder="Enter text and press Shift+Enter or Send",
                    container=False,
                    interactive=True,
                    autofocus=True,
                )

            with gr.Row() as row_two:
                text_submit_btn = gr.Button(value="Send ➤")
                clear_button = gr.ClearButton([input_txt, chatbot])
                clear_button.click(lambda: None, None, [thread_id], queue=False)

            txt_msg = input_txt.submit(fn=ChatBot.respond,
                                       inputs=[chatbot, input_txt, thread_id],
                                       outputs=[input_txt,
                                                chatbot,
                                                thread_id],
                                       queue=False).then(lambda: gr.Textbox(interactive=True),
                                                         None, [input_txt], queue=False)

            txt_msg = text_submit_btn.click(fn=ChatBot.respond,
                                            inputs=[chatbot, input_txt, thread_id],
                                            outputs=[input_txt,
                                                     chatbot,
                                                     thread_id],
                                            queue=False).then(lambda: gr.Textbox(interactive=True),
                                                              None, [input_txt], queue=False)


if __name__ == "__main__":
    demo.launch()
