"""
EstatePulse AI: Enterprise-Grade Real Estate Valuation Engine & MLOps Platform
Author: Anurag Gurjar (https://github.com/anuraggurjar01)
"""
import sys

def main():
    print("=" * 60)
    print("  EstatePulse AI - Enterprise Real Estate Valuation Platform")
    print("  Architect & Lead: Anurag Gurjar")
    print("  GitHub: https://github.com/anuraggurjar01")
    print("  LinkedIn: https://www.linkedin.com/in/anuraggurjar1/")
    print("=" * 60)
    print("\nService Entrypoints:")
    print("  - FastAPI Backend : uvicorn src.api.main:app --reload --port 8000")
    print("  - Streamlit UI    : streamlit run app.py")
    print("  - MLflow Registry : mlflow ui --port 5000")
    print("  - Docker Compose  : docker compose up -d\n")

if __name__ == "__main__":
    main()

