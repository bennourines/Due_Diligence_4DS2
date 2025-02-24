from langchain.prompts import PromptTemplate

# Create a custom prompt template that escapes the problematic sections.
custom_template = """
Your question is: {question}

Here is the JSON schema:
{{
    "{{\"$defs\"}}": {{
        "foo": "bar"
    }},
    "{{\"properties\"}}": {{
        "example": "value"
    }}
}}
"""

prompt_template = PromptTemplate(
    template=custom_template,
    input_variables=["question"]
)

# Then, if SmartScraperGraph accepts a prompt template instead of a simple string, pass it:
smart_scraper_graph = SmartScraperGraph(
    prompt=prompt_template,  # instead of the simple string prompt
    source="https://coinmarketcap.com/",
    schema=Coins,
    config=graph_config
)
