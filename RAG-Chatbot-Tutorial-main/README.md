# Simple RAG Chatbot Tutorial

Install dependencies.

```python
pip install -r requirements.txt
```

Create the VectorStore for pdf. In this tutorial, we will create a vector store for library general FAQ. 

```python
python vector_store.py
```

Build chatbot app with Streamlit.

```python
python chatbot_app.py 
```

Run Streamlit at local terminal 
```python
streamlit run chatbot_app.py
```


You'll also need to set up an Azure OpenAI account (and set the OpenAI key in your environment variable) for this to work.
You can also use other open source LLM for embedding and chatbot.
