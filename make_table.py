from json import load

results = load(open("results.json"))

models = list(sorted(list(set.union(*[set(r.keys()) for r in results.values()]))))
table_header = ["Model"] + models
rows = [table_header]

print("\n")
for k, m_res in results.items():
    nums = [m_res[m]["best"] * (1 if m_res[m]["best"] > 1 else 100) if m in m_res else 0 for m in models]
    maxn = max(nums)
    nums = [("__**" if n == maxn else "") + "{0:.2f}".format(n) + ("**__" if n == maxn else "") if n > 0 else "?"
            for n in nums]
    rows.append([" ".join(k.split("|"))] + nums)

columns = list(zip(*rows))
columns = columns[:1] + [["-" * len(s) for s in columns[0]]] + columns[1:]

print("\n".join(["| " + " | ".join(map(str, r)) + " |" for r in columns]))
