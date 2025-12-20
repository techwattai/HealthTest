import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import List, Union
import sys
import os

# Add parent directory to path to allow importing from root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Configurations.config import settings

router = APIRouter()

class EmailRequest(BaseModel):
    to_emails: Union[str, List[str]]
    subject: str
    body: str

def send_email_background(to_emails: Union[str, List[str]], subject: str, body: str):
    sender_email = settings.MAIL_USERNAME
    sender_password = settings.MAIL_PASSWORD
    smtp_server = settings.MAIL_SERVER
    smtp_port = settings.MAIL_PORT or 465
    
    if not sender_email or not sender_password or not smtp_server:
        print("Error: Email credentials missing in configurations.")
        return

    if isinstance(to_emails, str):
        to_emails = [to_emails]

    # Create the email
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['Subject'] = subject
    # 'To' header - mainly for display. 
    msg['To'] = ", ".join(to_emails)

    msg.attach(MIMEText(body, 'plain'))

    try:
        # Connect to the server
        if smtp_port == 465:
            server = smtplib.SMTP_SSL(smtp_server, smtp_port)
        else:
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            
        server.login(sender_email, sender_password)
        # Send email
        server.sendmail(sender_email, to_emails, msg.as_string())
        server.quit()
        print(f"Email sent successfully to: {to_emails}")
    except Exception as e:
        print(f"Failed to send email: {e}")

@router.post("/send-email", tags=["Email Service"])
async def send_email_endpoint(email_request: EmailRequest, background_tasks: BackgroundTasks):
    """
    Endpoint to send an email to one or multiple recipients.
    This runs in the background to avoid blocking the response.
    """
    background_tasks.add_task(
        send_email_background,
        email_request.to_emails,
        email_request.subject,
        email_request.body
    )
    return {"message": "Email sending process has been queued."}
