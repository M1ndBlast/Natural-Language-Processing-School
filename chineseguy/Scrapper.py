import re, urllib.parse
from typing import List

from playwright.sync_api import sync_playwright, ElementHandle, Page


def query_selector(pointer: list, jsHandle:ElementHandle, selector: str)->bool:
	element = jsHandle.query_selector(selector)
	if element:
		pointer[0] = element
		return True
	return False

def transleDeepl(sentence:str):
	res = ""
	query_url = "https://www.deepl.com/zh/translator#es/zh/"+ urllib.parse.quote(sentence.encode('utf8'))
	try:
		with sync_playwright() as playwright:
			with playwright.firefox.launch(headless=True, slow_mo=50) as browser:
				with browser.new_page() as page:
					page.goto(query_url, wait_until='networkidle')
					translation = page.text_content("#target-dummydiv")
					if translation!=None:
						res = translation
	except Exception as e:
		print(f"I caught a exception {e.with_traceback()}")
	finally:
		res = re.sub(r"\n|\s", "", res)
		if res=="":
			res = "*Sin Traduccón Disponible*"
		return res

def web_query(query: str) -> str:
	res = "No lo sé"
	query_url = 'https://www.google.com/search?q=' + urllib.parse.quote(query.encode('utf8'))
	#print(query_url)

	try:
		with sync_playwright() as playwright:
			with playwright.firefox.launch(headless=True, slow_mo=50) as browser:
				with browser.new_page() as page:
					page.goto(query_url, wait_until='networkidle')
					container = page.wait_for_selector('#rcnt [data-hveid^="CAIQ"]', state='attached')
					if container:
						res_field:List[ElementHandle] = [None]
						if query_selector(res_field, container, '.kp-header'):			# Card
							#print("1")
							res = res_field[0].inner_text()
						elif query_selector(res_field, container, '.vk_bk'):			# Quick Responses
							#print("2")
							res = res_field[0].inner_text()

							fields = container.query_selector_all('.vk_bk > span:nth-child(1)')
							if len(fields)>1:
								res += " ".join([ 
										field.text_content() if i>0 
										else "" 
									for i, field in enumerate(fields)
								])
						elif query_selector(res_field, container, '#cwos'):				# Calculator
							#print("3")
							res = res_field[0].inner_text()
						elif query_selector(res_field, container, '[role="heading"]'):	# Multiple responses
							#print("4")
							fields = res_field[0].query_selector_all(":scope > div")
							
							res = " ".join([ 
									field.inner_text() if i+1<len(fields) 
									else "" 
								for i, field in enumerate(fields)
							])
						else:
							print("\rDon't found option")

	except Exception as e:
		print(f"I caught a exception {e.with_traceback()}")
	finally:
		res = re.sub(r'\n', ' ', res)
		res = re.sub(r' \(.+\)|\:', '', res)
		if res=="":
			print("\r___________")
			res = "No lo sé"
		return res

if __name__=="__main__":
	while True:
		print("Chinese Guy: "+web_query(input("Tú:\t     ")))
