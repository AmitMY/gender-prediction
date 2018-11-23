from json import load

results = load(open("results.json"))

models = ["Spacy", "RNN", "CNN", "RCNN", "LSTM", "LSTMAttention", "SelfAttention"]
table_header = ["Train", "Dev"] + models
table_divider = ["-" * len(s) for s in table_header]
rows = [table_header, table_divider]

print(results)
for k, m_res in results.items():
    nums = ["{0:.2f}".format(m_res[m]["best"] * (1 if m_res[m]["best"] > 1 else 100)) if m in m_res else "?" for m in
               models]
    rows.append([k] + nums)

print("\n".join(["| " + " | ".join(map(str, r)) + " |" for r in rows]))
