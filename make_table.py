from json import load

results = load(open("results.json"))

models = list(sorted(list(set.union(*[set(r.keys()) for r in results.values()]))))
table_header = ["Train", "Dev"] + models
table_divider = ["-" * len(s) for s in table_header]
rows = [table_header, table_divider]

print("\n")
for k, m_res in results.items():
    nums = [m_res[m]["best"] * (1 if m_res[m]["best"] > 1 else 100) if m in m_res else 0 for m in models]
    maxn = max(nums)
    nums = [("__**" if n == maxn else "") + "{0:.2f}".format(n) + ("**__" if n == maxn else "") if n > 0 else "?"
            for n in nums]
    rows.append([k] + nums)

print("\n".join(["| " + " | ".join(map(str, r)) + " |" for r in rows]))
