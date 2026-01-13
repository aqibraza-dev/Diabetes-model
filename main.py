import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal

# 1. Initialize API
app = FastAPI(title="Diabetes Risk API", version="1.0")

# 2. CORS Setup (Crucial for React Integration)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://med-ai-pro.vercel.app", "http://localhost:3000"], # Add your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Load the Trained Model
try:
    model = joblib.load('diabetes_model.pkl')
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# 4. Define Input Schema
# This matches the 'formData' state in your React DiabetesPredictor.jsx exactly
class DiabetesInput(BaseModel):
    age: float
    bmi: float
    waist_to_hip_ratio: float
    systolic_bp: float
    physical_activity_minutes_per_week: float
    diet_score: float
    sleep_hours_per_day: float
    screen_time_hours_per_day: float
    
    diastolic_bp: float
    heart_rate: float
    alcohol_consumption_per_week: float
    cholesterol_total: float
    ldl_cholesterol: float
    hdl_cholesterol: float
    triglycerides: float
    
    # Categorical fields (Strings from Select options)
    gender: Literal["Male", "Female", "Other"]
    ethnicity: Literal["White", "Hispanic", "Asian", "Black", "Other"]
    education_level: Literal["No formal", "Highschool", "Graduate", "Postgraduate"]
    income_level: Literal["Low", "Lower-Middle", "Middle", "Upper-Middle", "High"]
    smoking_status: Literal["Never", "Former", "Current"]
    employment_status: Literal["Employed", "Unemployed", "Student", "Retired"]
    
    # Binary fields (0 or 1 from Checkboxes)
    family_history_diabetes: int = Field(..., ge=0, le=1)
    hypertension_history: int = Field(..., ge=0, le=1)
    cardiovascular_history: int = Field(..., ge=0, le=1)

# 5. Prediction Endpoint
@app.post("/predictdiabetes")
async def predict(data: DiabetesInput):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Convert Pydantic object to DataFrame
        # The key names must match exactly what the model was trained on
        input_data = data.dict()
        df = pd.DataFrame([input_data])
        
        # Make Prediction
        # predict_proba returns [prob_class_0, prob_class_1]
        probability = model.predict_proba(df)[0][1] 
        
        # Risk Logic
        if probability >= 0.7:
            risk = "High"
            analysis = "Probability indicates distinct markers of Type 2 Diabetes."
        elif probability >= 0.4:
            risk = "Moderate"
            analysis = " elevated risk factors detected. Lifestyle intervention recommended."
        else:
            risk = "Low"
            analysis = "Metabolic indicators are within healthy ranges."

        return {
            "risk_assessment": risk,
            "probability_score": float(probability),
            "detailed_analysis": analysis
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# Health Check
@app.get("/")
def home():
    return {"status": "Diabetes API is running"}
