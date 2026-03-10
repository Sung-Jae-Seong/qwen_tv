import glob, re
logs = glob.glob('./logs/*.log')
best_val = 999.0
best_log = ''
results = []
for log in logs:
    with open(log, 'r', errors='ignore') as f:
        matches = re.findall(r"'eval_loss': '([\d.]+)'", f.read())
    if matches:
        min_loss = min([float(m) for m in matches])
        last_loss = float(matches[-1])
        if min_loss < best_val:
            best_val = min_loss
            best_log = log
        results.append((log.split('/')[-1], min_loss, last_loss))

results.sort(key=lambda x: x[1])
for r in results:
    print(f'{r[0]}: min={r[1]}, last={r[2]}')

print('---')
print(f'Best: {best_log.split("/")[-1]} with eval_loss = {best_val}')
