from chatterbot.logic import LogicAdapter, TimeLogicAdapter
from chatterbot.conversation import Statement
from chatterbot import ChatBot



class TestLogicAdapter(LogicAdapter):
	def __init__(self, chatbot: ChatBot, **kwargs):
		self.timeAdapter = TimeLogicAdapter(chatbot, **kwargs)
		super().__init__(chatbot, **kwargs)

	def can_process(self, statement:Statement):
		print(f"> can process {statement.text}? {self.timeAdapter.can_process(statement)}")
		return self.timeAdapter.can_process(statement)

	def process(self, input_statement:Statement, additional_response_selection_parameters)->Statement:
		print("*"*40)
		print(f">            response to {input_statement.in_response_to}, conversation: {input_statement.conversation}")
		print(f">            {input_statement.text}, confidence: {additional_response_selection_parameters}")
		res = self.timeAdapter.process(input_statement, additional_response_selection_parameters)
		print(f"> process -> response to {res.in_response_to}, conversation: {res.conversation}")
		print(f">            {res.text}, confidence: {res.confidence}")
		print("*"*40)
		return res