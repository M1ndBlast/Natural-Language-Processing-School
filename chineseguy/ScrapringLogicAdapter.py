from chatterbot.logic import LogicAdapter
from chatterbot.conversation import Statement
from chatterbot import ChatBot
from Scrapper import web_query


class ScrapringLogicAdapter(LogicAdapter):
	def __init__(self, chatbot: ChatBot, **kwargs):
		super().__init__(chatbot, **kwargs)

	def can_process(self, statement:Statement):
		"""
		@TODO Determinar como saber si puede procesar
		"""
		return len(statement.text)>=10

	def process(self, input_statement:Statement, additional_response_selection_parameters)->Statement:
		res = web_query(input_statement.text)
		res_statement = Statement(res, in_response_to=input_statement)
		res_statement.confidence = 0.1 if res=="No lo sé" else 0.7
		"""
		print("*"*40)
		print(f"> process -> response to {res_statement.in_response_to}, conversation: {res_statement.conversation}")
		print(f">            {res_statement.text}, confidence: {res_statement.confidence}")
		print("*"*40)
		"""
		return res_statement