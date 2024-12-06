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




# from fastapi import FastAPI, Request
# from fastapi.responses import HTMLResponse
# from fastapi.templating import Jinja2Templates
# from transformers import GPT2LMHeadModel, GPT2Tokenizer
# from pydantic import BaseModel
# from fastapi.responses import JSONResponse

# from fastapi.responses import Response

# from fastapi.middleware.cors import CORSMiddleware
# import torch

# # Define FastAPI app
# app = FastAPI()

# # Allow CORS (Cross-Origin Resource Sharing)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Allow all origins, can be restricted later
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Load GPT-2 model and tokenizer
# model_name = "gpt2"
# tokenizer = GPT2Tokenizer.from_pretrained(model_name)
# model = GPT2LMHeadModel.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id)

# # Set up Jinja2 templates to render HTML
# templates = Jinja2Templates(directory="templates")

# # Class for incoming text data
# class TextData(BaseModel):
#     text: str

# # Function to generate summary using GPT-2
# def generate_summary(inp):
    
#     # Tokenize the input text
#     inputs = tokenizer.encode(inp, return_tensors="pt")
#     # Generate summary
#     summary_ids = model.generate(
#         inputs,max_length=100,min_length=30,length_penalty=2.0,num_beams=5,early_stopping=True)
#     # Decode the summary tokens
#     summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True,clean_up_tokenization_spaces=True)
#     return summary
# print(generate_summary)
# # Serve the HTML page at the root URL
# @app.get("/", response_class=HTMLResponse)
# async def get_html_page(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request})

# # POST endpoint for text summarization
# @app.post("/summarize/")
# async def summarize_text(text_data: TextData):
#     input_text = text_data.text
#     summary = generate_summary(input_text)
#     print(generate_summary)
#     return Response(content={"summary": summary})

# # Run the app with Uvicorn
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000)






# # Load GPT-2 model and tokenizer
# model_name = "gpt2"
# tokenizer = GPT2Tokenizer.from_pretrained(model_name)
# tokenizer.pad_token = tokenizer.eos_token
# model = GPT2LMHeadModel.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id)

# # Set up Jinja2 templates to render HTML
# templates = Jinja2Templates(directory="templates")

# # Class for incoming text data
# class TextData(BaseModel):
#     text:str



# # Function to generate a summary
# def generate_summary(text):
#     # Tokenize the input text
#     inputs = tokenizer.encode(text, return_tensors="pt", max_length=300, truncation=True)
#     # Generate summary
#     summary_ids = model.generate(
#         inputs,
#         max_length=100,       # Maximum length of the generated summary
#         min_length=30,        # Minimum length of the generated summary
#         length_penalty=2.0,   # Penalty for longer lengths
#         num_beams=5,          # Use beam search for better output
#         early_stopping=True    # Stop when the best sequence is found
#     )
#     # Decode the summary tokens
#     summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
#     return summary

# # Example paragraph to summarize
# paragraph = """
# Artificial Intelligence (AI) is a rapidly advancing field that aims to create machines capable of performing tasks that typically require human intelligence. 
# These tasks include problem-solving, understanding natural language, and recognizing patterns. 
# As AI technology continues to evolve, it has the potential to revolutionize various industries, from healthcare to finance.
# """

# print("org paragraph",paragraph)

# # Generate and print the summary
# sum = generate_summary(paragraph)
# print("Summary:", sum)


# # Function to generate summary using GPT-2
# def generate_summary(text):
#     # Tokenize the input text
#     inputs = tokenizer.encode(text, return_tensors="pt")
   
#     # Generate summary
#     summary_ids = model.generate(inputs,max_length=200,length_penalty=2.0,num_beams=5,early_stopping=True)
#     # Decode the summary tokens
#     summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    
#     return summary
# Function to generate summary using GPT-2
# def generate_summary(text):
#     # Tokenize the input text
#     inputs = tokenizer.encode(
#         text,
#         return_tensors="pt",       # Return PyTorch tensors
#         max_length=512,             # Set a max length for the input
#         truncation=True,            # Truncate if longer than max_length
#         padding='max_length'        # Pad to the max length
#     )
    

#     # Generate summary
#     summary_ids = model.generate(
#         inputs,
#         max_new_tokens=200, #set equal to max length
#         length_penalty=2.0,
#         num_beams=5,
#         early_stopping=True
#     )
    
#     # Decode the summary tokens
#     summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    
#     return summary

# apple=" " "An apple is a round, edible fruit produced by an apple tree (Malus spp., among them the domestic or orchard apple; Malus domestica). Apple trees are cultivated worldwide and are the most widely grown species in the genus Malus. The tree originated in Central Asia, where its wild ancestor, Malus sieversii, is still found. Apples have been grown for thousands of years in Eurasia and were introduced to North America by European colonists." " "

# print(generate_summary(apple))


#this code works
# from fastapi import FastAPI, Request    
# from fastapi.responses import HTMLResponse
# from fastapi.responses import Response
# from fastapi.templating import Jinja2Templates
# from transformers import GPT2LMHeadModel, GPT2Tokenizer
# from pydantic import BaseModel
# from fastapi.middleware.cors import CORSMiddleware
# import torch

# # Define FastAPI app
# app = FastAPI()

# # Allow CORS (Cross-Origin Resource Sharing)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Allow all origins, can be restricted later
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Load GPT-2 model and tokenizer
# model_name = "gpt2"
# tokenizer = GPT2Tokenizer.from_pretrained(model_name)
# model = GPT2LMHeadModel.from_pretrained(model_name,pad_token_id=tokenizer.eos_token_id)



# # Set up Jinja2 templates to render HTML
# templates = Jinja2Templates(directory="templates")

# # Class for incoming text data
# class TextData(BaseModel):
#     text: str

# # Function to generate summary using GPT-2
# def generate_summary(text: str):
#     inputs = tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)
#     summary_ids = model.generate(inputs, max_length=100, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
#     summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
#     return summary

# # Serve the HTML page at the root URL
# @app.get("/", response_class=HTMLResponse)
# async def get_html_page(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request})

# # POST endpoint for text summarization
# @app.post("/summarize/")
# async def summarize_text(text_data: TextData):
#     input_text = text_data.text
#     summary = generate_summary(input_text)
#     return {"summary": summary}

# # Run the app with Uvicorn
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000)

# from fastapi import FastAPI
# from pydantic import BaseModel
# from transformers import GPT2Tokenizer, GPT2LMHeadModel

# app = FastAPI()

# # Load GPT-2 model and tokenizer
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# model = GPT2LMHeadModel.from_pretrained("gpt2")

# # Define input data model
# class TextRequest(BaseModel):
#     text: str

# # Function to summarize the text using GPT-2
# def summarize_text(text: str) -> str:
#     inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
#     summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
#     return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# # Route to accept a POST request and return the summary
# @app.post("/summarize")
# async def summarize(request: TextRequest):
#     summary = summarize_text(request.text)
#     return {"summary": summary}

# # Run the app using the command: uvicorn main:app --reload

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000)



# # /backend/app.py
# from fastapi import FastAPI
# from pydantic import BaseModel
# from transformers import GPT2LMHeadModel, GPT2Tokenizer

# # Define the FastAPI app
# app = FastAPI()

# # Define a request model to accept text input
# class TextRequest(BaseModel):
#     text: str

# # Load GPT-2 model and tokenizer
# model_name = "gpt2"
# tokenizer = GPT2Tokenizer.from_pretrained(model_name)
# model = GPT2LMHeadModel.from_pretrained(model_name)

# # Function to summarize the input text
# def summarize_text(text: str, max_length: int = 50):
#     inputs = tokenizer.encode(text, return_tensors="pt")
#     summary_ids = model.generate(inputs, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2, early_stopping=True)
#     summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
#     return summary

# # Route to accept a POST request with the input text and return the summary
# @app.post("/summarize")
# async def summarize(request: TextRequest):
#     summary = summarize_text(request.text)
#     return {"summary": summary}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000)

