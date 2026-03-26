# AI Smart Inbox Organizer

An AI-powered email classification system that helps users prioritize and manage their inbox efficiently.

 (<img width="353" height="464" alt="Screenshot 2026-03-25 231116" src="https://github.com/user-attachments/assets/84c94459-fe68-47d0-b2aa-64e65c06d0fe" />
)

## Features
- ML-based email classification
- LLM fallback using Gemini AI
- Smart categorization:
  - Important
  - Notification
  - Newsletter
  - Promotion
  - Job Alert
- Action suggestions for each category

## Architecture
- Frontend: Lovable (UI)
- Backend: FastAPI
- ML Model: Logistic Regression (scikit-learn)
- LLM: Google Gemini API

## Workflow
1. User inputs email
   |
3. ML model predicts category
   |
5. If confidence < threshold → LLM fallback
   |
7. Final category + action returned

## Example
Input:
"Reminder: Your appointment is tomorrow"

Output:
- Category: Important
- Action: Mark as high priority

## Tech Stack
- Python
- FastAPI
- scikit-learn
- Google Gemini API

## Demo
Lovable App: (https://cortex-mail.lovable.app/)

## Next steps/ Future improvement plan:
- Gmail / Outlook integration for real-time email classification
- Advanced analytics dashboard (email trends, response time insights
