import os
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from google.genai.types import HttpOptions
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Ensure credentials map properly inside the backend folder structure
current_dir = Path(__file__).parent
load_dotenv(dotenv_path=current_dir / ".env")

if not os.getenv("GEMINI_API_KEY"):
    raise RuntimeError("CRITICAL ERR: GEMINI_API_KEY environment variable missing.")

app = FastAPI(title="Unified AI Resume Tailor Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TailorRequest(BaseModel):
    resume_text: str
    job_description: str

class TailoredResumeSchema(BaseModel):
    match_percentage: int = Field(..., description="An integer from 0 to 100")
    gap_analysis: str = Field(..., description="Text summary of missing criteria")
    tailored_summary: str = Field(..., description="Rewritten candidate overview summary")
    tailored_bullet_points: list[str] = Field(..., description="Array of updated action experiences")
    suggested_skills: list[str] = Field(..., description="Array of structural keywords and skills")

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options=HttpOptions(api_version="v1")
)

@app.post("/api/tailor")
def tailor_resume_endpoint(payload: TailorRequest):
    if not payload.resume_text.strip() or not payload.job_description.strip():
        raise HTTPException(status_code=400, detail="Inputs cannot be empty strings.")

    prompt_content = (
        f"You are a professional resume writer. Analyze this resume content:\n{payload.resume_text}\n\n"
        f"Tailor it completely to match this job description:\n{payload.job_description}\n\n"
        f"Output your final response as a JSON object matching the requested schema."
    )

    # UPDATED CLUSTER LAYOUT CONFIGURATIONS
    primary_model = 'gemini-3.8-flash'       # Primary stable GA model iteration
    fallback_model = 'gemini-3.5-flash-lite'   # High-availability low-latency failover node

    try:
        print(f"🔄 Routing request payload to primary stable cluster: {primary_model}...")
        response = client.models.generate_content(
            model=primary_model,
            contents=prompt_content,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=TailoredResumeSchema,
                temperature=0.3
            ),
        )
        return json.loads(response.text)
        
    except Exception as primary_error:
        print(f"⚠️ Primary model ({primary_model}) hit traffic throttling: {str(primary_error)}")
        print(f"🔄 Instantly switching to backup failover infrastructure: {fallback_model}...")
        
        try:
            response = client.models.generate_content(
                model=fallback_model,
                contents=prompt_content,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=TailoredResumeSchema,
                    temperature=0.3
                ),
            )
            print(f"✅ Failover successful via {fallback_model} cluster configuration!")
            return json.loads(response.text)
            
        except Exception as fallback_error:
            print(f"🔥 TOTAL CLUSTER FAILURE DETECTED: {str(fallback_error)}")
            raise HTTPException(
                status_code=500, 
                detail="All available Gemini API endpoints are currently congested. Please retry in a moment."
            )

# Map out Frontend Directory assets paths
frontend_dir = Path(__file__).parent.parent / "Resume-Tailor-Frontend"

if frontend_dir.exists():
    @app.get("/")
    def serve_frontend_root():
        return FileResponse(frontend_dir / "index.html")

    app.mount("/", StaticFiles(directory=frontend_dir), name="frontend")
else:
    @app.get("/")
    def system_warning_stub():
        return {"error": f"Could not map frontend structure components at: {frontend_dir.resolve()}"}
