import os
import pandas as pd
from typing import List
from datetime import datetime, date


class Memory:
    """
    A class for handling the storage of chatbot conversation history by writing chat logs to a CSV file.
    """
    @staticmethod
    def write_chat_history_to_file(gradio_chatbot: List,  thread_id: str, folder_path: str) -> None:
        """
        Writes the most recent chatbot interaction (user query and response) to a CSV file. The log includes
        the thread ID and timestamp of the interaction. The file for each day is saved with the current date as the filename.
        """
        tmp_list = list(gradio_chatbot[-1]) 

        today_str = date.today().strftime('%Y-%m-%d')
        tmp_list.insert(0, thread_id) 

        current_time_str = datetime.now().strftime('%H:%M:%S')
        tmp_list.insert(1, current_time_str)  

        # File path for today's CSV file
        file_path = os.path.join(folder_path, f'{today_str}.csv')

        new_df = pd.DataFrame([tmp_list], columns=[
                              "thread_id", "timestamp", "user_query", "response"])

        if os.path.exists(file_path):
            new_df.to_csv(file_path, mode='a', header=False, index=False)
        else:
            new_df.to_csv(file_path, mode='w', header=True, index=False)
