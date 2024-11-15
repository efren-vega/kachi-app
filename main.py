from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

app = FastAPI()

model_name = 'hackathon-pln-es/t5-small-spanish-nahuatl'
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.eval()

class TranslationRequest(BaseModel):
    sentence: str

@app.post("/translate/")
def translate_text(request: TranslationRequest):
    print(request.sentence)
    inputs = tokenizer('translate Spanish to Nahuatl: ' + request.sentence, return_tensors='pt')

    outputs = model.generate(**inputs)

    translation = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    return {"translation": translation}

@app.get("/")
def read_root():
    return {"Hello": "World"}