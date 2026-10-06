import os
from google import genai
from google.genai import types
# Import HttpOptions to force the production API version if needed
from google.genai.types import HttpOptions 
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load the API key from your .env file
load_dotenv()

# Define what the AI MUST return using Pydantic
class TailoredResumeSchema(BaseModel):
    match_percentage: int = Field(description="The match score between 0 and 100 based on the job description alignment.")
    gap_analysis: str = Field(description="Brief summary of skills or keywords missing from the original resume.")
    tailored_summary: str = Field(description="A highly targeted professional summary optimized for this job description.")
    tailored_bullet_points: list[str] = Field(description="A list of action-oriented professional experience bullet points rewritten to emphasize keywords from the job description.")
    suggested_skills: list[str] = Field(description="A clean list of targeted hard and soft skills for this specific position.")

def test_gemini_json():
    # Initialize the Gemini Client forcing the standard production api_version
    client = genai.Client(
        api_key=os.getenv("GEMINI_API_KEY"),
        http_options=HttpOptions(api_version="v1")
    )
    
    sample_resume = "Software developer with 2 years experience. Skilled in Python and basic HTML. Built a simple calculator web app."
    sample_job_description = "Seeking a Backend Engineer proficient in Python, building secure REST APIs using FastAPI, and optimizing database connections."

    print("🔄 Sending data to Gemini 2.5/3.5 Flash in structured JSON mode...")

    # We use 'gemini-2.5-flash' (or 'gemini-3.5-flash') for modern compatibility
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=f"Analyze this resume: {sample_resume}\nTailor it to this job description: {sample_job_description}",
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=TailoredResumeSchema,
                temperature=0.3
            ),
        )
        print("\n✅ API Success! Received JSON Output:")
        print(response.text)
        
    except Exception as e:
        print(f"\n🔄 Fallback attempt with alternative model variant due to: {e}")
        # Secondary fallback if your tier profile uses the next iteration
        response = client.models.generate_content(
            model='gemini-3.5-flash',
            contents=f"Analyze this resume: {sample_resume}\nTailor it to this job description: {sample_job_description}",
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=TailoredResumeSchema,
                temperature=0.3
            ),
        )
        print("\n✅ API Success via standard cluster! Received JSON Output:")
        print(response.text)

if __name__ == "__main__":
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ Error: GEMINI_API_KEY not found in your .env file!")
    else:
        test_gemini_json()
