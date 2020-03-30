import pandas as pd
import matplotlib.pyplot as plt
import os
root = os.path.join(os.getcwd(), 'Data_3')
paths = os.path.join(root,'scalars.json')
paths2 = os.path.join(root,'scalars2.json')
data = pd.read_json(paths)
data.drop(0, axis=1, inplace=True)
print(data.head())
data = data.values
data2 = pd.read_json(paths2)
data2.drop(0, axis=1, inplace=True)
print(data2.head())
data2 = data2.values
plt.plot(data[:, 0], data[:, 1])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation accuracy against Epoches', fontweight='bold')
plt.grid()
plt.show()

plt.plot(data2[:, 0], data2[:, 1], 'r--')
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.title('Validation loss against Epoches', fontweight='bold')
plt.grid()
plt.show()