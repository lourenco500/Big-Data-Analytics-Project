import json
import random
import time
from datetime import datetime

import pandas as pd
from kafka import KafkaProducer

producer = KafkaProducer(
    bootstrap_servers=["localhost:8098"],
    value_serializer=lambda m: json.dumps(m).encode("utf-8"),
)

TRIPS_POLL_SECONDS = 5
TRIPS_PER_CYCLE = 50

# Load the cleaned taxi parquet once
df = pd.read_parquet("data/clean/taxi_clean.parquet")
df["tpep_pickup_datetime"]  = df["tpep_pickup_datetime"].astype(str)
df["tpep_dropoff_datetime"] = df["tpep_dropoff_datetime"].astype(str)

# Extract only the relevant columns and convert to list of dicts for sampling
RECORDS = df[[
    "tpep_pickup_datetime", "tpep_dropoff_datetime",
    "PULocationID", "DOLocationID", "passenger_count",
    "trip_distance", "fare_amount", "tip_amount",
    "total_amount", "payment_type",
]].to_dict(orient="records")

# For simplicity, we sample from the full dataset each cycle
def sample_trips(k):
    return random.sample(RECORDS, k=min(k, len(RECORDS)))


if __name__ == "__main__":
    while True:
        for trip in sample_trips(TRIPS_PER_CYCLE):
            # Each cycle re-emits every current trip with a fresh observation timestamp - the join sees them as new events
            print(f"trip  @ {datetime.now()} | {trip}")
            producer.send("trips", trip)

            # Also publish to long_trips if distance > 10 miles
            if trip["trip_distance"] > 10:
                producer.send("long_trips", trip)

        producer.flush()
        time.sleep(TRIPS_POLL_SECONDS)