from open_ai_completion import OpenAICompletion

class MemoryConstructor:

    # Pre-defined prompts for OpenAI's GPT model
    SYSTEM_MESSAGE = """ 
        You are an expert at extracting the information from a text and construct human memories based on that."
    """

    USER_MESSAGE_TEMPLATE = """
        Let's think step by step.
        1. Consider the text: {}.
        2. Generate a title that captures the usage of the provided text, labeled as 'title'
        3. Generate a human-like memory that could be used to retrieve the most relevant inforation later, labeled as 'memory'.
        4. Return a JSON object in the following format:'title':'memory'.
    """

    def __init__(self, model:int, open_ai_key:str):
        """
        Initialize the QuestionGenerator.
        """
        self.openAIcompletion = OpenAICompletion(model, open_ai_key)

    def generate(self, text: str) -> dict:
        """
        Generate a set of closed-ended questions based on the provided text.

        Args:
            text (str): The reference content used to generate questions.

        Returns:
            dict: A dictionary of generated questions with keys indicating the question order and values being the questions themselves.
        """
        user_message = self.USER_MESSAGE_TEMPLATE.format(text)
        message = [
            {'role': 'system', 'content': self.SYSTEM_MESSAGE}, 
            {'role': 'user', 'content': user_message}
        ]

        openai_response = self.openAIcompletion.get_completion_from_messages(message)
        openai_response_json = self.openAIcompletion.extract_json_from_response(openai_response)

        return openai_response_json
