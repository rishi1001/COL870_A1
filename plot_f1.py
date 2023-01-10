import matplotlib.pyplot as plt

filename = './topology_generated/f1_results_'
architectures = ['linear_3','gnn_1','gnn_2','gnn_3','gnn_4','gnn_5']

f1_val_list = []
f1_test_list = []

for arch in architectures:
    fn = filename+arch
    with open(fn, 'r') as f:
        f1_test_list.append(float(f.readline()))
        f1_val_list.append(float(f.readline()))

print(f1_test_list)
print(f1_val_list)

plt.clf()
plt.plot(architectures,f1_test_list, label='test')
plt.plot(architectures,f1_val_list, label='val')
plt.legend()
plt.title("Macro F1 vs Architecture")
plt.xlabel("Architecture")
plt.ylabel("F1")
plt.draw()
plt.savefig(f"./topology_generated/arch-f1.png")
plt.close()

