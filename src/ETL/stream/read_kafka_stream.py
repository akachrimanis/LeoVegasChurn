from confluent_kafka import Consumer, KafkaException, KafkaError
import pandas as pd
import json


def read_kafka_stream(
    broker: str, topic: str, group_id: str, timeout: int = 10, max_messages: int = 100
) -> pd.DataFrame:
    """
    Read data from a Kafka stream and return it as a DataFrame.

    Args:
        broker (str): Kafka broker address (e.g., 'localhost:9092').
        topic (str): Kafka topic to consume messages from.
        group_id (str): Consumer group ID.
        timeout (int, optional): Timeout in seconds for message consumption. Default is 10 seconds.
        max_messages (int, optional): Maximum number of messages to consume. Default is 100.

    Returns:
        pd.DataFrame: DataFrame containing the consumed messages.
    """
    try:
        # Set up the Kafka consumer configuration
        conf = {
            "bootstrap.servers": broker,  # Kafka broker address
            "group.id": group_id,  # Consumer group ID
            "auto.offset.reset": "earliest",  # Start from the earliest available message
        }

        # Create a Consumer instance
        consumer = Consumer(conf)

        # Subscribe to the topic
        consumer.subscribe([topic])

        # List to store the messages
        messages = []

        # Consume messages from Kafka
        print(f"Consuming messages from Kafka topic: {topic}")
        count = 0
        while count < max_messages:
            msg = consumer.poll(timeout=timeout)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    print(
                        f"End of partition reached {msg.topic()} [{msg.partition()}] at offset {msg.offset()}"
                    )
                else:
                    raise KafkaException(msg.error())
            else:
                # Extract the message value and convert it to a dictionary (assuming it's in JSON format)
                message_value = msg.value().decode("utf-8")  # Decode bytes to string
                message_dict = json.loads(
                    message_value
                )  # Assuming the message is a JSON object
                messages.append(message_dict)
                count += 1

        # Close the consumer
        consumer.close()

        # Convert the messages to a pandas DataFrame
        data = pd.DataFrame(messages)
        print(f"Successfully consumed {len(messages)} messages.")
        return data

    except Exception as e:
        print(f"Error reading from Kafka stream: {e}")
        raise
