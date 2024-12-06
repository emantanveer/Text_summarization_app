# Text_summarization_app

This is a simple web application that allows users to input text and receive a summarized version of the text. Initially, the app uses GPT-2 for summarization, but after testing, we found that GPT-2 may not provide the best performance for summarizing long or complex texts. Instead, it is recommended to use models like **BART** or **T5**, which are specifically optimized for summarization tasks.

## Features
- **Text Summarization**: Users can input text, and the app will attempt to summarize it.
- **User Interface**: The frontend allows users to input text and display the resulting summary.

## Technologies Used
- **Backend**: FastAPI
- **Model**: GPT-2 (initially used), but alternatives like BART or T5 are suggested for better summarization performance.
- **Frontend**: HTML, JavaScript, CSS
- **Serving**: Uvicorn for running the FastAPI server




