from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from typing import List
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# Use environment variable for API key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

model = ChatOpenAI(model="gpt-3.5-turbo")

class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="User's input for recipe generation")

class Recipe(BaseModel):
    recipe_name: str = Field(..., description="Name of the recipe")
    ingredients: List[str] = Field(..., description="List of ingredients with quantities and preparation")
    instructions: List[str] = Field(..., description="Step-by-step instructions for preparing the recipe")
    servings: int = Field(..., description="Number of servings")
    nutrition: dict = Field(..., description="Nutritional information of the recipe")

class GenerateResponse(BaseModel):
    recipe: Recipe

# Create a PydanticOutputParser
output_parser = PydanticOutputParser(pydantic_object=Recipe)

# Create the prompt template
prompt = PromptTemplate(
    template="Generate a recipe based on the user's input.\n{format_instructions}\nUser Input: {user_input}",
    input_variables=["user_input"],
    partial_variables={"format_instructions": output_parser.get_format_instructions()},
)

@app.post("/generate", response_model=GenerateResponse)
async def generate_recipe(request: GenerateRequest):
    try:
        # Combine the prompt template with the model and output parser
        chain = prompt | model | output_parser

        # Generate the recipe
        recipe = chain.invoke({"user_input": request.prompt})

        return GenerateResponse(recipe=recipe)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recipe: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)