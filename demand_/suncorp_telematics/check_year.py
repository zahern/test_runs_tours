import pandas as pd
df = pd.read_csv( /mnt/hpccs01/home/ahernz/test_runs_tours/demand_/suncorp_telematics/data/hard_braking_events.csv, 
                 usecols=[EventStart], parse_dates=[EventStart], nrows=100000)
print(Min:, df[EventStart].min())
print(Max:, df[EventStart].max())
print(Year dist:)
print(df[EventStart].dt.year.value_counts().sort_index())
