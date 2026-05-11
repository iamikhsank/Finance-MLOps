import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

load_dotenv()

def send_email_alert(subject: str, message_body: str):
    """
    Sends an email alert using Gmail SMTP.
    Requires GMAIL_USER and GMAIL_APP_PASSWORD in .env
    """
    sender_email = os.getenv("GMAIL_USER")
    sender_password = os.getenv("GMAIL_APP_PASSWORD")
    receiver_email = os.getenv("ALERT_RECEIVER_EMAIL", sender_email) # Defaults to sending to yourself
    
    if not sender_email or not sender_password:
        print("Alerting Warning: GMAIL_USER or GMAIL_APP_PASSWORD is not set in .env. Alert was not sent.")
        return False
        
    try:
        # Set up the MIME
        message = MIMEMultipart()
        message['From'] = f"Finance MLOps Alert <{sender_email}>"
        message['To'] = receiver_email
        message['Subject'] = f"[MLOps Alert] {subject}"
        
        # Attach the body with UTF-8 encoding
        message.attach(MIMEText(message_body, 'plain', 'utf-8'))
        
        # Connect to Gmail SMTP Server
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls() # Secure the connection
        server.login(sender_email, sender_password)
        
        # Send Email
        text = message.as_string()
        server.sendmail(sender_email, receiver_email, text)
        server.quit()
        
        print(f"Alert email successfully sent to {receiver_email}: {subject}")
        return True
    except Exception as e:
        print(f"Failed to send email alert: {e}")
        return False

if __name__ == "__main__":
    # Test script if executed directly
    print("Testing Email Alert System...")
    success = send_email_alert("Test Alert", "This is a test message from your Finance MLOps Pipeline.")
    if success:
        print("Email system is working!")
    else:
        print("Email system failed. Please check your .env credentials.")
