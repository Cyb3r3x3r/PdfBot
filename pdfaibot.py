#importing libraries
#from langchain_community.document_loaders import PyPDFLoader
import pdfplumber
import os,re,warnings
from typing import List,Dict
from transformers import AutoTokenizer,AutoModelForQuestionAnswering,pipeline
from transformers import logging as lg
from alive_progress import alive_bar
lg.set_verbosity_error()

# Ignores all type of warnings 
warnings.filterwarnings("ignore")



def extract_text_from_pdf(pdf_path):
    try:
        page_text = {}
        # opening the pdf file
        with pdfplumber.open(pdf_path) as pdf:
            for page_num,text in enumerate(pdf.pages,start=1):
                page_text[page_num] = text.extract_text()
        return page_text
    except Exception as e:
        print(f"Error : {e}")
        return None
    
#Now preprocessing the data remove unwanted symbols and characters
def preprocess_text(page_text):
    try:
        processed_text = {}
        for page,text in page_text.items():
            text = re.sub(r'[^\x20-\x7E]','',text)

            text = re.sub(r'[---]+','-',text)
            text = re.sub(r'[""]','"',text)
            text = re.sub(r"['']","'",text)

            text = re.sub(r'\s+',' ',text).strip()

            processed_text[page]=text
        return processed_text
    except Exception as e:
        print(f"Error in preprocessing : {e}")
        return None

# Now its time to divide the sentence in chunks. here we will divide the text into paragraphs
def chunk_text(page_text,chunk_size=300,split_by="paragraphs"):
    try:
        chunked_text = {}
        for page,text in page_text.items():
            chunks = re.split(r'\n\n+',text)
            chunked_text[page] = chunks
        return chunked_text
    except Exception as e:
        print(f"Error in chunking the text : {e}")
        return None

# After the chunking of the text we can now load the pretrained model

def setup_model():
    try:
        lg.set_verbosity_error()

        model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)

        #Create a pipeline for question answering

        qa_pipeline = pipeline("question-answering",model=model,tokenizer=tokenizer)
        print("Loading of the model done.")
        return qa_pipeline
    
    except Exception as e:
        print(f"Error loading the model : {e}")
        return None

# Function for question answering
def answer_question(qa_pipeline,question,chunked_text):
    answers = []
    try:
        for page,chunks in chunked_text.items():
            for chunk in chunks:
                result = qa_pipeline(question=question, context=chunk)
                answers.append({
                    "page":page,
                    "text":chunk,
                    "answer":result["answer"],
                    "score":result["score"]
                })

        answers = sorted(answers,key=lambda x:x["score"],reverse=True)
        return answers
    except Exception as e:
        print(f"Error in getting the answers: {e}")


if __name__ == "__main__":
    os.system('cls' if os.name=='nt' else 'clear')
    print("PDF AI bot")
    print(30*"*")
    print("A bot which answers your question by looking into your PDF")
    print(30*"*")
    pdf_path = input("Please enter the pdf path: ")
    steps = 3
    with alive_bar(steps,title="Processing the PDF document") as bar:
        page_text = extract_text_from_pdf(pdf_path)
        bar()
        if page_text ==None:
            exit()
        processed_text = preprocess_text(page_text)
        bar()
        chunked_text =chunk_text(processed_text)
        bar()
    qa_pipeline=setup_model()
    print(20*"*")
    print("Model is ready to answer the questions.(Type exit if you want to stop asking questions)")
    while True:
        question = input("Question : ")
        if question=="exit":
            break
        answers = answer_question(qa_pipeline,question,chunked_text)

        if answers:
            print(f"Model : {answers[0]['answer']}")
        else:
            print("Model : Sorry i wasn't able to find the answer.")