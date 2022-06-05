from chatterbot.logic import LogicAdapter
from chatterbot.conversation import Statement
from chatterbot import ChatBot


class MyLogicAdapter(LogicAdapter):
	def __init__(self, chatbot: ChatBot, **kwargs):
		super().__init__(chatbot, **kwargs)

	def can_process(self, statement:Statement):
		print(f"can process: {statement.text}")
		return True

	def process(self, input_statement:Statement, additional_response_selection_parameters):
		import random

		# Randomly select a confidence between 0 and 1
		confidence = random.uniform(0, 1)

		# For this example, we will just return the input as output
		selected_statement = input_statement
		selected_statement.confidence = confidence

		print(f"input {input_statement.text}\tconfidence {confidence}")

		return selected_statement