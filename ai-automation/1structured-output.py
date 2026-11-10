from ollama import chat
from pydantic import BaseModel
import json

countries = input("Give me some countries: ")

class Country(BaseModel):
  name: str
  short_description: str
  absolute_location: str
  economy_type: str
  states_provinces: list[str]
  territories: list[str]

class CountryList(BaseModel):
  countries: list[Country]

response = chat(
  model='llama3.2:1b',
  messages=[{'role': 'user', 'content': f"Tell me about {countries}. For each country, give me a short description, an absolute location, the economy type, all states or provinces, and owned territories."}],
  format=CountryList.model_json_schema(),
)

countrylist = CountryList.model_validate_json(response.message.content)
print(countrylist)
for country in countrylist.countries:
  print(country.name)