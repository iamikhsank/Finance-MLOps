from ml_service.src.alerting import send_email_alert

if __name__ == "__main__":
    print("Testing MLOps Alerting System...")
    print("Sending test email to your inbox...")
    
    success = send_email_alert(
        subject="🚀 System Online", 
        message_body="Congratulations! Your Finance MLOps Alerting System is configured correctly and online.\n\nYou will now receive alerts for Data Drift and Model Retraining here."
    )
    
    if success:
        print("\n✅ Success! Check your Gmail inbox (or spam folder) for the message.")
    else:
        print("\n❌ Failed. Please ensure your .env file contains GMAIL_USER and GMAIL_APP_PASSWORD.")
