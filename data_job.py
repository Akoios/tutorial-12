import google.auth
from google.cloud import bigquery
from google.cloud import bigquery_storage_v1beta1
from google.cloud import storage
import pandas as pd
import os

# Access to Google Cloud Services

credentials, your_project_id = google.auth.default(
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
)

bqclient = bigquery.Client(
    credentials=credentials,
    project=your_project_id,
)

bqstorageclient = bigquery_storage_v1beta1.BigQueryStorageClient(
    credentials=credentials
)

# Download query results and store them in a Dataframe

query_string = """
SELECT 
  Country, MarketSegment, ArrivalDateMonth, DepositType, CustomerType, LeadTime, ArrivalDateYear, ArrivalDateWeekNumber, ArrivalDateDayOfMonth, RequiredCarParkingSpaces, IsCanceled 
FROM 
  `datasets.hotel_reservations` 
ORDER BY 
  ReservationStatusDate DESC
LIMIT 10000
"""

dataframe = (
    bqclient.query(query_string)
    .result()
    .to_dataframe(bqstorage_client=bqstorageclient)
)

# Create a CSV file and upload it to Google Storage

dataframe.to_csv('hotel_reservations.csv', index=False)
client = storage.Client()
bucket = client.get_bucket('tutorial-datasets')
blob = bucket.blob('hotel_reservations.csv')
blob.upload_from_filename('hotel_reservations.csv')
blob.make_public()
