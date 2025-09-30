import pyarrow.parquet as pq

pq_path = '/home/joao.lima/data/quick_samples/out_temp_envSmoothsr20/beat_timestamps.parquet'

table = pq.read_table(
    source=pq_path,
    columns=['timestamps'],
    filters=[('key', '=', 'JN.wav')]
)

ts = table['timestamps'][0]
print(type(ts))