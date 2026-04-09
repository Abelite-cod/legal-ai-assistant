import os
from fastapi import UploadFile

PDF_DIR = "pdfs"
os.makedirs(PDF_DIR, exist_ok=True)

async def store_pdf(file: UploadFile):
    file_location = os.path.join(PDF_DIR, file.filename)
    with open(file_location, "wb") as f:
        f.write(await file.read())
    return file_location