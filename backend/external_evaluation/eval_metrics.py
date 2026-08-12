import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

df = pd.read_csv('sbc_eval_results/sbcsae_results.csv')
y_true = df['human_reference_class']
y_raw = df['raw_prediction']
y_styl = df['stylized_prediction']
y_raw_term = df['raw_terminal_prediction']
y_styl_term = df['styl_terminal_prediction']

labels = ['Falling', 'Level', 'Rising']

print("Accuracy:")
print("Raw Overall:", accuracy_score(y_true, y_raw))
print("Styl Overall:", accuracy_score(y_true, y_styl))
print("Raw Terminal:", accuracy_score(y_true, y_raw_term))
print("Styl Terminal:", accuracy_score(y_true, y_styl_term))

print("\nMacro F1:")
print("Raw Overall:", f1_score(y_true, y_raw, average='macro', labels=labels))
print("Styl Overall:", f1_score(y_true, y_styl, average='macro', labels=labels))
print("Raw Terminal:", f1_score(y_true, y_raw_term, average='macro', labels=labels))
print("Styl Terminal:", f1_score(y_true, y_styl_term, average='macro', labels=labels))

print("\nValid Terminal Fractions:")
print("Raw Term Valid:", len(df[df['raw_terminal_prediction'] != 'Insufficient_voicing']) / len(df))
print("Styl Term Valid:", len(df[df['styl_terminal_prediction'] != 'Insufficient_voicing']) / len(df))
