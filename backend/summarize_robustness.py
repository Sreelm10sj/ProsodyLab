import pandas as pd

df = pd.read_csv("stylization_validation/robustness_results.csv")

# Compute matching
df['Raw_Match'] = df['Raw Class'] == df['Clean Class']
df['Sty_Match'] = df['Sty Class'] == df['Clean Class']

# We only care about phrases that had valid Clean Slopes
df = df[df['Clean Slope'] != 'None']

def aggregate(df_group):
    # Mean of matches
    raw_acc = df_group['Raw_Match'].mean() * 100
    sty_acc = df_group['Sty_Match'].mean() * 100
    
    # Error in slope
    raw_err = df_group['Raw Slope Error'].mean()
    sty_err = df_group['Sty Slope Error'].mean()
    
    # Error in contour (RMSE)
    raw_rmse = df_group['Raw RMSE'].mean()
    sty_rmse = df_group['Sty RMSE'].mean()
    
    return pd.Series({
        'Raw Acc%': raw_acc,
        'Sty Acc%': sty_acc,
        'Raw Slope Err': raw_err,
        'Sty Slope Err': sty_err,
        'Raw RMSE': raw_rmse,
        'Sty RMSE': sty_rmse,
    })

print("\n--- AGGREGATE RESULTS BY CONDITION ---")
summary = df.groupby(['Condition', 'Severity']).apply(aggregate).reset_index()
print(summary.to_string(index=False))

print("\n--- AGGREGATE TOTAL (ALL CORRUPTIONS) ---")
print(df.apply(aggregate, axis=0).to_string())

