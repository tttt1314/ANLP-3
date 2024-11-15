import pandas as pd
import os

# File paths
folder = './eval_results'
file1 = 'cosine_similarity_results.csv'
file2 = 'musiccaps-public.csv'
file3 = 'kl_divergence_results.csv'

# Read CSV files
# join the folder path with the file path

df_clap = pd.read_csv(os.path.join(folder, file1))
df_table = pd.read_csv(os.path.join(".", file2))
df_kl = pd.read_csv(os.path.join(folder, file3))

## KL Analysis
# Get the first 10 filenames from df_kl
filenames_kl = df_kl['filename'].head(10).tolist()

# Extract ytid from filenames
ytid_kl = [filename.split('.')[0] for filename in filenames_kl]
# Find captions from df_table based on ytid_kl
captions_kl = df_table[df_table['ytid'].isin(ytid_kl)]['caption'].tolist()

# Get the last 10 filenames from df_kl
best_filenames_kl = df_kl['filename'].tail(10).tolist()
# Extract ytid from best_filenames_kl
best_ytid_kl = [filename.split('.')[0] for filename in best_filenames_kl]
# Find captions from df_table based on best_ytid_kl
best_captions_kl = df_table[df_table['ytid'].isin(best_ytid_kl)]['caption'].tolist()

## CLAP Analysis
# reduce the df_table['ytid'] only start from second character
df_table['ytid'] = df_table['ytid'].str[1:]
df_clap = df_clap.sort_values('similarity')
filenames_clap = df_clap['filename'].head(10).tolist()
ytid_clap = [filename.split('.')[0] for filename in filenames_clap]
captions_clap = df_table[df_table['ytid'].isin(ytid_clap)]['caption'].tolist()


# Get the last 10 filenames from df_clap
best_filenames_clap = df_clap['filename'].tail(10).tolist()
# Extract ytid from best_filenames_clap
best_ytid_clap = [filename.split('.')[0] for filename in best_filenames_clap]
# Find captions from df_table based on best_ytid_clap
best_captions_clap = df_table[df_table['ytid'].isin(best_ytid_clap)]['caption'].tolist()


# Print the best captions to a file
with open('analysis.txt', 'w') as f:
    f.write('\n\n')
    f.write('KL Divergence Analysis(Best 10)\n')
    f.write('----------------------\n')
    for i, caption in enumerate(best_captions_kl):
        f.write(f'{i+1}. {caption}\n')
    f.write('\n\n')
    f.write('CLAP Analysis(Best 10)\n')
    f.write('----------------------\n')
    for i, caption in enumerate(best_captions_clap):
        f.write(f'{i+1}. {caption}\n')


# Print the to a file
with open('analysis.txt', 'a') as f:
    f.write('----------------------\n')
    f.write('----------------------\n')
    f.write('\n\n')
    f.write('KL Divergence Analysis(Worst 10)\n')
    f.write('----------------------\n')
    for i, caption in enumerate(captions_kl):
        f.write(f'{i+1}. {caption}\n')
    f.write('\n\n')
    f.write('CLAP Analysis(Worst 10)\n')
    f.write('----------------------\n')
    for i, caption in enumerate(captions_clap):
        f.write(f'{i+1}. {caption}\n')