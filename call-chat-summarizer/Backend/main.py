from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import re
import whisper
from transformers import pipeline

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load models once
whisper_model = whisper.load_model("base")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


# ------------------ HELPERS ------------------

def fix_emails(s: str) -> str:
    pattern = r"([\w.-]+)\.at([\w.-]+\.[a-zA-Z]{2,})"
    fixed = re.sub(pattern, r"\1@\2", s)
    return fixed.lower()


def chunk_text(text, max_chars=3000):
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunks.append(text[start:end])
        start = end
    return chunks


def summarize_large_text(text: str) -> str:
    chunks = chunk_text(text, max_chars=3000)
    summaries = []

    for i, chunk in enumerate(chunks):
        chunk = chunk.strip()
        if len(chunk) < 50:
            continue

        try:
            result = summarizer(
                chunk,
                max_length=200,
                min_length=30,
                do_sample=False
            )
            
            # Check if result is valid and not empty
            if result and len(result) > 0 and "summary_text" in result[0]:
                s = result[0]["summary_text"]
                summaries.append(s)
            else:
                print(f"Warning: Empty result for chunk {i}, skipping...")
                continue
                
        except Exception as e:
            print(f"Error processing chunk {i}: {e}")
            continue

    if not summaries:
        return "Error: Could not generate summary. Text might be too short or empty."

    combined_summary = " ".join(summaries)

    # Optional second pass for cleaner result
    if len(combined_summary) > 1000:
        try:
            result = summarizer(
                combined_summary,
                max_length=200,
                min_length=50,
                do_sample=False
            )
            
            if result and len(result) > 0 and "summary_text" in result[0]:
                combined_summary = result[0]["summary_text"]
            else:
                print("Warning: Empty result for second pass, returning combined summary")
                
        except Exception as e:
            print(f"Error in second pass summarization: {e}")
            # Return the combined summary from first pass if second pass fails

    return combined_summary


# ------------------ ENDPOINT ------------------

@app.post("/process-file/")
async def process_file(file: UploadFile = File(...)):
    try:
        print("Received file:", file.filename)

        file_location = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_location, "wb") as f:
            f.write(await file.read())

        filename = file.filename.lower()

        # ---------- AUDIO FILE ----------
        if filename.endswith((".m4a", ".mp3", ".wav")):
            result = whisper_model.transcribe(file_location)
            text = result["text"]

        # ---------- TEXT FILE ----------
        elif filename.endswith(".txt"):
            with open(file_location, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()

        else:
            return JSONResponse(
                {"error": "Unsupported file type. Use .m4a, .mp3, .wav or .txt"},
                status_code=400
            )

        text = fix_emails(text)
        text = text.strip()

        if not text:
            return JSONResponse({"error": "File is empty"}, status_code=400)

        print("Text length:", len(text))

        summary = summarize_large_text(text)

        return {
            "text": text,
            "summary": summary
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


# ------------------ RUN ------------------
# uvicorn main:app --reload
  