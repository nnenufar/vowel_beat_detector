import pyarrow.parquet as pq
import numpy as np

pq_path = '/home/joao.lima/data/quick_samples/out_temp_mfcc/beat_timestamps.parquet'

table = pq.read_table(
    source=pq_path,
    columns=['beats', 'envelope', 'feats', 'intervals'],
    filters=[('key', '=', 'JN.wav')]
)

ts = np.array(table['feats'][0].as_py())
print(ts.shape, ts, type(ts))