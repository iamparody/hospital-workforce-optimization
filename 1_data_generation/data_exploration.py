import pandas as pd
from sqlalchemy import create_engine

engine = create_engine(
    "postgresql+psycopg2://postgres:Iamparody_94@localhost:5432/analytics"
)

df = pd.read_sql("SELECT * FROM patient_clinical_timeseries LIMIT 10;", engine)
print(df)
