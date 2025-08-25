import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
import json
import os
from typing import List, Dict, Optional

class FraudAlertSystem:
    def __init__(self, config_file='alert_config.json'):
        self.config_file = config_file
        self.config = self.load_config()
        self.alert_history = []
        self.last_alert_time = {}
        
    def load_config(self):
        """Load email configuration from file or create default"""
        default_config = {
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'sender_email': 'your-email@gmail.com',
            'sender_password': 'your-app-password',
            'recipients': ['admin@company.com'],
            'alert_thresholds': {
                'high_risk': 0.8,
                'medium_risk': 0.6,
                'low_risk': 0.3
            },
            'alert_cooldown_minutes': 30,
            'max_alerts_per_hour': 10
        }
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    # Merge with defaults for missing keys
                    for key, value in default_config.items():
                        if key not in config:
                            config[key] = value
                    return config
            except Exception as e:
                print(f"Error loading config: {e}")
                return default_config
        else:
            # Create default config file
            self.save_config(default_config)
            return default_config
    
    def save_config(self, config):
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=4)
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def update_config(self, **kwargs):
        """Update configuration parameters"""
        for key, value in kwargs.items():
            if key in self.config:
                self.config[key] = value
        
        self.save_config(self.config)
    
    def send_email_alert(self, subject: str, body: str, recipients: Optional[List[str]] = None) -> bool:
        """Send email alert"""
        if recipients is None:
            recipients = self.config['recipients']
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.config['sender_email']
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = subject
            
            # Add body
            msg.attach(MIMEText(body, 'html'))
            
            # Create SMTP session
            context = ssl.create_default_context()
            
            with smtplib.SMTP(self.config['smtp_server'], self.config['smtp_port']) as server:
                server.starttls(context=context)
                server.login(self.config['sender_email'], self.config['sender_password'])
                
                # Send email
                text = msg.as_string()
                server.sendmail(self.config['sender_email'], recipients, text)
            
            print(f"Alert email sent successfully to {recipients}")
            return True
            
        except Exception as e:
            print(f"Error sending email alert: {e}")
            return False
    
    def create_fraud_alert_email(self, transaction_data: Dict, risk_level: str, fraud_probability: float) -> tuple:
        """Create fraud alert email content"""
        subject = f"🚨 FRAUD ALERT: High-Risk Transaction Detected (Risk Level: {risk_level.upper()})"
        
        # Format transaction data
        amount = f"${transaction_data.get('amount', 0):.2f}"
        merchant = transaction_data.get('merchant_category', 'Unknown')
        location = transaction_data.get('merchant_location', 'Unknown')
        transaction_id = transaction_data.get('transaction_id', 'Unknown')
        
        # Create HTML body
        body = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .alert {{ background-color: #ffebee; border-left: 4px solid #f44336; padding: 15px; margin: 10px 0; }}
                .high-risk {{ background-color: #ffebee; border-left: 4px solid #f44336; }}
                .medium-risk {{ background-color: #fff3e0; border-left: 4px solid #ff9800; }}
                .low-risk {{ background-color: #e8f5e8; border-left: 4px solid #4caf50; }}
                .details {{ background-color: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 5px; }}
                .risk-score {{ font-size: 24px; font-weight: bold; color: #f44336; }}
            </style>
        </head>
        <body>
            <div class="alert {risk_level}-risk">
                <h2>🚨 Fraud Detection Alert</h2>
                <p><strong>Risk Level:</strong> {risk_level.upper()}</p>
                <p><strong>Fraud Probability:</strong> <span class="risk-score">{fraud_probability:.1%}</span></p>
            </div>
            
            <div class="details">
                <h3>Transaction Details:</h3>
                <ul>
                    <li><strong>Transaction ID:</strong> {transaction_id}</li>
                    <li><strong>Amount:</strong> {amount}</li>
                    <li><strong>Merchant Category:</strong> {merchant}</li>
                    <li><strong>Location:</strong> {location}</li>
                    <li><strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</li>
                </ul>
            </div>
            
            <div class="details">
                <h3>Additional Information:</h3>
                <ul>
                    <li><strong>Card Type:</strong> {transaction_data.get('card_type', 'Unknown')}</li>
                    <li><strong>Hour of Day:</strong> {transaction_data.get('hour_of_day', 'Unknown')}</li>
                    <li><strong>Distance from Home:</strong> {transaction_data.get('distance_from_home', 0):.1f} km</li>
                    <li><strong>Online Order:</strong> {'Yes' if transaction_data.get('online_order', 0) else 'No'}</li>
                </ul>
            </div>
            
            <p><em>This is an automated alert from the Financial Fraud Detection System.</em></p>
        </body>
        </html>
        """
        
        return subject, body
    
    def should_send_alert(self, transaction_id: str, risk_level: str) -> bool:
        """Check if alert should be sent based on cooldown and rate limiting"""
        current_time = datetime.now()
        
        # Check cooldown period
        if transaction_id in self.last_alert_time:
            time_diff = current_time - self.last_alert_time[transaction_id]
            if time_diff.total_seconds() < self.config['alert_cooldown_minutes'] * 60:
                return False
        
        # Check rate limiting
        alerts_last_hour = [
            alert for alert in self.alert_history
            if alert['timestamp'] > current_time - timedelta(hours=1)
        ]
        
        if len(alerts_last_hour) >= self.config['max_alerts_per_hour']:
            return False
        
        return True
    
    def send_fraud_alert(self, transaction_data: Dict, risk_level: str, fraud_probability: float) -> bool:
        """Send fraud alert for a transaction"""
        transaction_id = str(transaction_data.get('transaction_id', 'unknown'))
        
        # Check if alert should be sent
        if not self.should_send_alert(transaction_id, risk_level):
            return False
        
        # Create email content
        subject, body = self.create_fraud_alert_email(transaction_data, risk_level, fraud_probability)
        
        # Send email
        success = self.send_email_alert(subject, body)
        
        if success:
            # Update alert history
            alert_record = {
                'transaction_id': transaction_id,
                'risk_level': risk_level,
                'fraud_probability': fraud_probability,
                'timestamp': datetime.now(),
                'email_sent': True
            }
            
            self.alert_history.append(alert_record)
            self.last_alert_time[transaction_id] = datetime.now()
            
            # Keep only last 1000 alerts
            if len(self.alert_history) > 1000:
                self.alert_history = self.alert_history[-1000:]
        
        return success
    
    def send_batch_alert(self, high_risk_transactions: List[Dict]) -> bool:
        """Send batch alert for multiple high-risk transactions"""
        if not high_risk_transactions:
            return True
        
        subject = f"🚨 BATCH FRAUD ALERT: {len(high_risk_transactions)} High-Risk Transactions Detected"
        
        # Create batch email content
        body = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .alert {{ background-color: #ffebee; border-left: 4px solid #f44336; padding: 15px; margin: 10px 0; }}
                .transaction {{ background-color: #f5f5f5; padding: 10px; margin: 5px 0; border-radius: 5px; }}
                .risk-score {{ font-weight: bold; color: #f44336; }}
            </style>
        </head>
        <body>
            <div class="alert">
                <h2>🚨 Batch Fraud Detection Alert</h2>
                <p><strong>Number of High-Risk Transactions:</strong> {len(high_risk_transactions)}</p>
                <p><strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <h3>High-Risk Transactions:</h3>
        """
        
        for i, transaction in enumerate(high_risk_transactions, 1):
            amount = f"${transaction.get('amount', 0):.2f}"
            merchant = transaction.get('merchant_category', 'Unknown')
            risk_level = transaction.get('risk_level', 'Unknown')
            fraud_prob = transaction.get('fraud_probability', 0)
            
            body += f"""
            <div class="transaction">
                <p><strong>Transaction {i}:</strong></p>
                <ul>
                    <li><strong>ID:</strong> {transaction.get('transaction_id', 'Unknown')}</li>
                    <li><strong>Amount:</strong> {amount}</li>
                    <li><strong>Merchant:</strong> {merchant}</li>
                    <li><strong>Risk Level:</strong> {risk_level}</li>
                    <li><strong>Fraud Probability:</strong> <span class="risk-score">{fraud_prob:.1%}</span></li>
                </ul>
            </div>
            """
        
        body += """
            <p><em>This is an automated batch alert from the Financial Fraud Detection System.</em></p>
        </body>
        </html>
        """
        
        return self.send_email_alert(subject, body)
    
    def get_alert_statistics(self, hours: int = 24) -> Dict:
        """Get alert statistics for the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_alerts = [
            alert for alert in self.alert_history
            if alert['timestamp'] >= cutoff_time
        ]
        
        if not recent_alerts:
            return {
                'total_alerts': 0,
                'high_risk_alerts': 0,
                'medium_risk_alerts': 0,
                'low_risk_alerts': 0,
                'avg_fraud_probability': 0
            }
        
        stats = {
            'total_alerts': len(recent_alerts),
            'high_risk_alerts': len([a for a in recent_alerts if a['risk_level'] == 'high']),
            'medium_risk_alerts': len([a for a in recent_alerts if a['risk_level'] == 'medium']),
            'low_risk_alerts': len([a for a in recent_alerts if a['risk_level'] == 'low']),
            'avg_fraud_probability': sum(a['fraud_probability'] for a in recent_alerts) / len(recent_alerts)
        }
        
        return stats
    
    def test_email_configuration(self) -> bool:
        """Test email configuration by sending a test email"""
        subject = "🧪 Test Email - Fraud Detection System"
        body = """
        <html>
        <body>
            <h2>Test Email</h2>
            <p>This is a test email from the Financial Fraud Detection System.</p>
            <p>If you receive this email, the email configuration is working correctly.</p>
            <p><em>Time: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</em></p>
        </body>
        </html>
        """
        
        return self.send_email_alert(subject, body) 
