# Ericsson-interview
repo for interview with ericsson using langrapgh




### Setup 

instead of using pip, use uv for testing since its suppose to be fast. 

```bash
uv init . 
uv add python-dotenv langgraph "langchain[google-genai]" langsmith ipykernel faiss-cpu langchain-community
```

#### Add env variables

create a .env file and add the following variables 

```env
GOOGLE_API_KEY=your_google_genai_api_key"
````

