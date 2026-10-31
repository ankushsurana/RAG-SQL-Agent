import gradio as gr


class UISettings:
    """
    Utility class for managing UI settings.
    """
    @staticmethod
    def feedback(data: gr.LikeData):
        """
        Process user feedback on the generated response.
        """
        if data.liked:
            print(f"You upvoted this response: {data.value}")
        else:
            print(f"You downvoted this response: {data.value}")
