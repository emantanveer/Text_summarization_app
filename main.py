from transformers import GPT2LMHeadModel, GPT2Tokenizer
from fastapi import FastAPI, Request
import tensorflow as tf
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch

# Define FastAPI app
app = FastAPI()

# Allow CORS (Cross-Origin Resource Sharing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins, can be restricted later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up Jinja2 templates to render HTML
templates = Jinja2Templates(directory="templates")

# Class for incoming text data
class TextData(BaseModel):
    text: str

# Load the GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Set the pad token ID to the EOS token
tokenizer.pad_token = tokenizer.eos_token  # Set pad token to EOS token
model = GPT2LMHeadModel.from_pretrained(model_name, pad_token_id=tokenizer.pad_token_id)

# Function to generate a summary
def generate_summary(text):
    # Using a more specific prompt to guide the summarization
    prompt = f"Summarize the following text as if writing a headline for a news article:\n\n{text}\n\nSummary:"

    # Tokenize the input text and create attention mask
    inputs = tokenizer.encode_plus(
        prompt,
        return_tensors="pt",
        max_length=300,
        truncation=True,
        padding="max_length",  # Ensure padding is applied
        return_attention_mask=True  # Return attention mask
    )

    # Extract input IDs and attention mask
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Generate the summary
    summary_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,  # Pass the attention mask
        max_new_tokens=50,       
        min_length=10,       
        num_beams=5,         
        early_stopping=True   
    )

    # Decode the generated summary tokens
    sum = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    # Strip away the input prompt from the generated output to extract only the summary
    if "Summary:" in sum:
        summary = sum.split("Summary:")[-1].strip()  # Keep only the part after 'Summary:'
    else:
        summary = sum  # If no 'Summary:' is present, return the entire output

    return summary

# Example paragraph to summarize
paragraph = """
Artificial Intelligence (AI) is a rapidly advancing field that aims to create machines capable of performing tasks that typically require human intelligence. 
These tasks include problem-solving, understanding natural language, and recognizing patterns. 
As AI technology continues to evolve, it has the potential to revolutionize various industries, from healthcare to finance.
"""

# Generate and print the summary
summary = generate_summary(paragraph)
print("Summary:", summary)

# Serve the HTML page at the root URL
@app.get("/", response_class=HTMLResponse)
async def get_html_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# POST endpoint for text summarization
@app.post("/summarize/")
async def summarize_text(text_data: TextData):
    input_text = text_data.text
    summary = generate_summary(input_text)
    
    # Return the summary as a JSON response
    return JSONResponse(content={"summary": summary})

# Run the app with Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)



