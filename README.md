# PdfBot
A pdfbot for question answering

Create a virtual environment for you if you want
```
python -m venv virtualenv
virtualenv\Scripts\activate
```
Install all the necessary libraries

```
pip install -r requirements.txt
```
Run the pdfbot
```
python pdfaibot.py
```
If you don't have gpu, just remove device=cuda or try adding device=cpu or gpu ranked 1 based on your device from the following line
```
qa_pipeline = pipeline("question-answering",model=model,tokenizer=tokenizer,device="cuda")
```
