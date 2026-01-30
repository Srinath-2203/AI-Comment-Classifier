import pandas as pd
import random

# Load seed data
df = pd.read_csv("data/comments_seed.csv")

intensifiers = ["very", "really", "extremely", "quite", "so"]
prefixes = ["honestly", "frankly", "seriously", ""]
suffixes = ["", "right now", "these days", "to be honest"]

expanded = []

for _, row in df.iterrows():
    base = row["comment"]
    label = row["label"]

    expanded.append((base, label))

    for _ in range(15):  # 15 variations per sentence
        sentence = base

        if random.random() < 0.5:
            sentence = random.choice(prefixes) + " " + sentence

        if random.random() < 0.5:
            words = sentence.split()
            if len(words) > 2:
                idx = random.randint(1, len(words)-1)
                words.insert(idx, random.choice(intensifiers))
                sentence = " ".join(words)

        if random.random() < 0.5:
            sentence = sentence + " " + random.choice(suffixes)

        expanded.append((sentence.strip(), label))

# Shuffle and save
final_df = pd.DataFrame(expanded, columns=["comment", "label"])
final_df = final_df.sample(frac=1).reset_index(drop=True)

print("Final dataset size:", len(final_df))

final_df.to_csv("data/comments.csv", index=False)
