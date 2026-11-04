import pandas as pd

df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
try:
    df.drop_duplicates(subset=['a', 'nonexistent'], keep='first')
except Exception as e:
    print(f"Error type: {type(e).__name__}")
    print(f"Error str: {str(e)}")
