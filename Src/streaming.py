#!/usr/bin/env python3
"""
Streaming Module for Financial Fraud Detection System
Implements Apache Kafka and Spark Streaming for real-time fraud detection
"""

import json
import time
import threading
from datetime import datetime
from typing import Dict, List, Optional, Callable
import pandas as pd
import numpy as np

# Kafka imports
try:
    from kafka import KafkaProducer, KafkaConsumer
    from kafka.errors import KafkaError
    KAFKA_AVAILABLE = True
except ImportError:
    print("Warning: kafka-python not available. Install with: pip install kafka-python")
    KAFKA_AVAILABLE = False

# Spark imports
try:
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import *
    from pyspark.sql.types import *
    from pyspark.streaming import StreamingContext
    from pyspark.streaming.kafka import KafkaUtils
    SPARK_AVAILABLE = True
except ImportError:
    print("Warning: PySpark not available. Install with: pip install pyspark")
    SPARK_AVAILABLE = False

class KafkaStreamingManager:
    """Manages Kafka streaming for real-time transaction processing"""
    
    def __init__(self, bootstrap_servers=['localhost:9092'], topic='transactions'):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.producer = None
        self.consumer = None
        self.is_running = False
        
        if not KAFKA_AVAILABLE:
            print("Kafka not available - using simulated streaming")
    
    def create_producer(self):
        """Create Kafka producer"""
        if not KAFKA_AVAILABLE:
            return None
            
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: str(k).encode('utf-8')
            )
            print(f"Kafka producer created for topic: {self.topic}")
            return self.producer
        except Exception as e:
            print(f"Failed to create Kafka producer: {e}")
            return None
    
    def create_consumer(self, group_id='fraud_detection_group'):
        """Create Kafka consumer"""
        if not KAFKA_AVAILABLE:
            return None
            
        try:
            self.consumer = KafkaConsumer(
                self.topic,
                bootstrap_servers=self.bootstrap_servers,
                group_id=group_id,
                auto_offset_reset='earliest',
                enable_auto_commit=True,
                value_deserializer=lambda x: json.loads(x.decode('utf-8'))
            )
            print(f"Kafka consumer created for topic: {self.topic}")
            return self.consumer
        except Exception as e:
            print(f"Failed to create Kafka consumer: {e}")
            return None
    
    def send_transaction(self, transaction_data: Dict):
        """Send transaction to Kafka topic"""
        if self.producer and KAFKA_AVAILABLE:
            try:
                future = self.producer.send(
                    self.topic,
                    key=str(transaction_data.get('transaction_id', 'unknown')),
                    value=transaction_data
                )
                future.get(timeout=10)
                return True
            except Exception as e:
                print(f"Failed to send transaction to Kafka: {e}")
                return False
        else:
            # Simulate sending
            print(f"Simulated: Sent transaction {transaction_data.get('transaction_id', 'unknown')}")
            return True
    
    def start_consuming(self, callback: Callable[[Dict], None]):
        """Start consuming messages from Kafka topic"""
        if not self.consumer or not KAFKA_AVAILABLE:
            print("Kafka consumer not available - using simulated consumption")
            return
        
        self.is_running = True
        
        def consume_messages():
            try:
                for message in self.consumer:
                    if not self.is_running:
                        break
                    
                    transaction = message.value
                    print(f"Received transaction: {transaction.get('transaction_id', 'unknown')}")
                    callback(transaction)
                    
            except Exception as e:
                print(f"Error consuming messages: {e}")
            finally:
                self.consumer.close()
        
        # Start consumption in a separate thread
        consumer_thread = threading.Thread(target=consume_messages)
        consumer_thread.daemon = True
        consumer_thread.start()
    
    def stop_consuming(self):
        """Stop consuming messages"""
        self.is_running = False
        if self.consumer:
            self.consumer.close()
        if self.producer:
            self.producer.close()

class SparkStreamingManager:
    """Manages Spark Streaming for real-time fraud detection"""
    
    def __init__(self, app_name="FraudDetectionStreaming"):
        self.app_name = app_name
        self.spark = None
        self.ssc = None
        
        if not SPARK_AVAILABLE:
            print("Spark not available - using simulated streaming")
    
    def create_spark_session(self):
        """Create Spark session for streaming"""
        if not SPARK_AVAILABLE:
            return None
            
        try:
            self.spark = SparkSession.builder \
                .appName(self.app_name) \
                .config("spark.sql.streaming.checkpointLocation", "/tmp/checkpoint") \
                .getOrCreate()
            
            print("Spark session created successfully")
            return self.spark
        except Exception as e:
            print(f"Failed to create Spark session: {e}")
            return None
    
    def create_streaming_context(self, batch_duration=5):
        """Create Spark Streaming context"""
        if not self.spark or not SPARK_AVAILABLE:
            return None
            
        try:
            self.ssc = StreamingContext(self.spark.sparkContext, batch_duration)
            print(f"Spark Streaming context created with {batch_duration}s batch duration")
            return self.ssc
        except Exception as e:
            print(f"Failed to create Streaming context: {e}")
            return None
    
    def create_kafka_stream(self, topic='transactions', kafka_params=None):
        """Create Kafka stream with Spark Streaming"""
        if not self.ssc or not SPARK_AVAILABLE:
            return None
            
        if kafka_params is None:
            kafka_params = {
                'bootstrap.servers': 'localhost:9092',
                'group.id': 'fraud_detection_group',
                'auto.offset.reset': 'earliest'
            }
        
        try:
            stream = KafkaUtils.createDirectStream(
                self.ssc,
                [topic],
                kafkaParams=kafka_params,
                valueDecoder=lambda x: json.loads(x.decode('utf-8'))
            )
            
            print(f"Kafka stream created for topic: {topic}")
            return stream
        except Exception as e:
            print(f"Failed to create Kafka stream: {e}")
            return None
    
    def process_transactions_stream(self, stream, fraud_detector):
        """Process transaction stream for fraud detection"""
        if not stream or not SPARK_AVAILABLE:
            return
        
        def process_batch(rdd):
            if rdd.isEmpty():
                return
            
            # Convert RDD to DataFrame
            transactions = rdd.map(lambda x: x[1]).collect()
            if not transactions:
                return
            
            # Process transactions
            for transaction in transactions:
                try:
                    prediction = fraud_detector.predict_single_transaction(transaction)
                    print(f"Transaction {transaction.get('transaction_id', 'unknown')}: "
                          f"Risk: {prediction['risk_level']}, "
                          f"Probability: {prediction['fraud_probability']:.3f}")
                except Exception as e:
                    print(f"Error processing transaction: {e}")
        
        stream.foreachRDD(process_batch)
    
    def start_streaming(self):
        """Start Spark Streaming"""
        if self.ssc and SPARK_AVAILABLE:
            try:
                self.ssc.start()
                self.ssc.awaitTermination()
            except KeyboardInterrupt:
                print("Stopping Spark Streaming...")
                self.ssc.stop()
        else:
            print("Spark Streaming not available")

class RealTimeFraudDetector:
    """Real-time fraud detection system using Kafka and Spark Streaming"""
    
    def __init__(self, fraud_detector, kafka_config=None, spark_config=None):
        self.fraud_detector = fraud_detector
        self.kafka_manager = KafkaStreamingManager(**kafka_config) if kafka_config else KafkaStreamingManager()
        self.spark_manager = SparkStreamingManager(**spark_config) if spark_config else SparkStreamingManager()
        self.is_running = False
        self.processed_count = 0
        self.fraud_count = 0
    
    def start_real_time_detection(self):
        """Start real-time fraud detection"""
        print("Starting real-time fraud detection system...")
        
        # Initialize Kafka
        self.kafka_manager.create_producer()
        self.kafka_manager.create_consumer()
        
        # Initialize Spark (if available)
        if SPARK_AVAILABLE:
            self.spark_manager.create_spark_session()
            self.spark_manager.create_streaming_context()
        
        # Start consuming transactions
        self.kafka_manager.start_consuming(self.process_transaction)
        
        self.is_running = True
        print("Real-time fraud detection started!")
    
    def process_transaction(self, transaction: Dict):
        """Process a single transaction for fraud detection"""
        try:
            # Predict fraud
            prediction = self.fraud_detector.predict_single_transaction(transaction)
            
            # Update statistics
            self.processed_count += 1
            if prediction['risk_level'] in ['high', 'very_high']:
                self.fraud_count += 1
            
            # Log results
            print(f"Transaction {transaction.get('transaction_id', 'unknown')}: "
                  f"Risk: {prediction['risk_level']}, "
                  f"Probability: {prediction['fraud_probability']:.3f}")
            
            # Send alert for high-risk transactions
            if prediction['risk_level'] in ['high', 'very_high']:
                self.send_fraud_alert(transaction, prediction)
                
        except Exception as e:
            print(f"Error processing transaction: {e}")
    
    def send_fraud_alert(self, transaction: Dict, prediction: Dict):
        """Send fraud alert for high-risk transaction"""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'transaction_id': transaction.get('transaction_id', 'unknown'),
            'risk_level': prediction['risk_level'],
            'fraud_probability': prediction['fraud_probability'],
            'transaction_data': transaction
        }
        
        print(f"🚨 FRAUD ALERT: {alert}")
        # Here you would integrate with your alert system
    
    def generate_test_transactions(self, count: int = 100, interval: float = 1.0):
        """Generate and send test transactions"""
        print(f"Generating {count} test transactions...")
        
        for i in range(count):
            if not self.is_running:
                break
            
            # Generate synthetic transaction
            transaction = self.fraud_detector.generate_synthetic_transaction()
            transaction['transaction_id'] = f"test_{i+1:06d}"
            
            # Send to Kafka
            self.kafka_manager.send_transaction(transaction)
            
            time.sleep(interval)
        
        print("Test transaction generation completed")
    
    def get_statistics(self) -> Dict:
        """Get real-time processing statistics"""
        return {
            'processed_count': self.processed_count,
            'fraud_count': self.fraud_count,
            'fraud_rate': self.fraud_count / (self.processed_count if self.processed_count > 0 else 1),
            'is_running': self.is_running
        }
    
    def stop(self):
        """Stop real-time fraud detection"""
        self.is_running = False
        self.kafka_manager.stop_consuming()
        print("Real-time fraud detection stopped")

# Simulated streaming for when Kafka/Spark are not available
class SimulatedStreaming:
    """Simulated streaming for testing without Kafka/Spark"""
    
    def __init__(self, fraud_detector, interval=1.0):
        self.fraud_detector = fraud_detector
        self.interval = interval
        self.is_running = False
        self.processed_count = 0
        self.fraud_count = 0
    
    def start_simulation(self, transaction_count=100):
        """Start simulated streaming"""
        print(f"Starting simulated streaming with {transaction_count} transactions...")
        self.is_running = True
        
        for i in range(transaction_count):
            if not self.is_running:
                break
            
            # Generate transaction
            transaction = self.fraud_detector.generate_synthetic_transaction()
            transaction['transaction_id'] = f"sim_{i+1:06d}"
            
            # Process for fraud detection
            try:
                prediction = self.fraud_detector.predict_single_transaction(transaction)
                
                self.processed_count += 1
                if prediction['risk_level'] in ['high', 'very_high']:
                    self.fraud_count += 1
                
                print(f"Transaction {transaction['transaction_id']}: "
                      f"Risk: {prediction['risk_level']}, "
                      f"Probability: {prediction['fraud_probability']:.3f}")
                
            except Exception as e:
                print(f"Error processing transaction: {e}")
            
            time.sleep(self.interval)
        
        print("Simulated streaming completed")
        self.is_running = False
    
    def stop_simulation(self):
        """Stop simulated streaming"""
        self.is_running = False
    
    def get_statistics(self) -> Dict:
        """Get simulation statistics"""
        return {
            'processed_count': self.processed_count,
            'fraud_count': self.fraud_count,
            'fraud_rate': self.fraud_count / (self.processed_count if self.processed_count > 0 else 1),
            'is_running': self.is_running
        }

if __name__ == "__main__":
    # Test the streaming module
    print("Testing Streaming Module...")
    
    # Test simulated streaming
    from detect_fraud import FraudDetector
    
    detector = FraudDetector()
    detector.load_models()
    
    simulator = SimulatedStreaming(detector, interval=0.5)
    simulator.start_simulation(10)
    
    stats = simulator.get_statistics()
    print(f"Simulation Statistics: {stats}") 
