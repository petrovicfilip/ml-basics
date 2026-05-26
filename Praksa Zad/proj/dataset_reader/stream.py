import csv
import json
import time
import logging
from datetime import datetime

from kafka import KafkaProducer

# Constants
KAFKA_TOPIC = "ids"
KAFKA_BROKER = "localhost:9092"
TIMESTAMP_COL = "Timestamp"
TIME_FORMAT = "%d/%m/%Y %H:%M:%S"


logging.basicConfig(level=logging.INFO)


def process_csv(file_name: str):
    try:
        file = open(file_name, newline="", encoding="utf-8")
    except Exception as e:
        return e

    reader = csv.reader(file)

    try:
        headers = next(reader)
    except Exception as e:
        return e

    # Kafka producer (async like Go writer.Async = true)
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        max_block_ms=300000,          # cekaj do 5min za metadata
        request_timeout_ms=300000,    # timeout po requestu
        delivery_timeout_ms=600000,   # ukupni delivery timeout
        batch_size=16384,             # smanji batch ako se gomilaju
        linger_ms=10,
        acks=1                     # umesto 'all' ako je sporije
    )

    logging.info("Begin reading records")

    return send_immediately(reader, headers, producer)


def send_immediately(reader, headers, producer):
    count = 0
    for row in reader:
        try:
            data = read_row(row, headers)
        except Exception:
            break

        if data.get("Active Max") == "Active Max":
            continue

        try:
            write_kafka_message(data, producer)
            count += 1
            if count % 1000 == 0:          # ← log svakih 1000 poruka
                logging.info(f"Sent {count} messages...")
        except Exception:
            continue

    producer.flush()
    producer.close()
    logging.info(f"Done. Total sent: {count}")  # ← na kraju
    return None


def send_with_timestamps(reader, headers, producer):
    last_timestamp = None

    for row in reader:
        try:
            data = read_row(row, headers)
        except Exception:
            break

        ts_string = data.get(TIMESTAMP_COL)
        if not ts_string:
            logging.info("Timestamp column not found")
            continue

        try:
            current_timestamp = datetime.strptime(ts_string, TIME_FORMAT)
        except Exception:
            logging.warning(f"Invalid timestamp format: {ts_string}")
            continue

        if last_timestamp is not None:
            diff = (current_timestamp - last_timestamp).total_seconds()
            if diff > 0:
                time.sleep(diff)

        last_timestamp = current_timestamp

        try:
            write_kafka_message(data, producer)
        except Exception:
            continue


import math

def read_row(row, headers):
    data = {}
    for i, value in enumerate(row):
        if i < len(headers):
            try:
                f = float(value)
                # NaN i Infinity nisu validni JSON → zameni sa None (postaje null)
                if math.isnan(f) or math.isinf(f):
                    data[headers[i]] = None
                else:
                    data[headers[i]] = f
            except (ValueError, TypeError):
                data[headers[i]] = value  # string kolone (Timestamp, Label)
    return data


def write_kafka_message(data, producer):
    try:
        producer.send(KAFKA_TOPIC, value=data)
    except Exception as e:
        logging.error(f"Failed to write message: {e}")


def main():
    csv_files = [
        r"C:\Praksa\Praksa Zad\datasets\archive\02-14-2018.csv",
        r"C:\Praksa\Praksa Zad\datasets\archive\02-15-2018.csv",
        r"C:\Praksa\Praksa Zad\datasets\archive\02-16-2018.csv",
        r"C:\Praksa\Praksa Zad\datasets\archive\02-20-2018.csv",
        r"C:\Praksa\Praksa Zad\datasets\archive\02-21-2018.csv",
        r"C:\Praksa\Praksa Zad\datasets\archive\02-22-2018.csv",
        r"C:\Praksa\Praksa Zad\datasets\archive\02-23-2018.csv",
        r"C:\Praksa\Praksa Zad\datasets\archive\02-28-2018.csv",
        r"C:\Praksa\Praksa Zad\datasets\archive\03-01-2018.csv",
        r"C:\Praksa\Praksa Zad\datasets\archive\03-02-2018.csv",
    ]

    for file_name in csv_files:
        logging.info(f"Processing file: {file_name}")

        result = process_csv(file_name)
        if isinstance(result, Exception):
            logging.error(f"Error processing {file_name}: {result}")


if __name__ == "__main__":
    main()