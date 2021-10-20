import os
import numpy as np
import matplotlib.pyplot as plt
import yaml
import pandas as pd

def generate_plot(history: dict, output: str, title: str, xlabel = 'Epoch', ylabel = 'Accuracy', name_keys=None):
    figure = plt.figure()
    labels = []
    for key, name in name_keys:
        plt.plot(np.array(history[key]))
        labels.append(name)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(labels, loc='upper left')
    if output:
        plt.savefig(output, bbox_inches='tight')
        print(f"Figure saved to {output}")
    plt.show()

def yaml_load(filename: str):
    with open(filename, 'rt') as f:
        return yaml.load(f, Loader=yaml.FullLoader)

files = glob.glob('../results/CC22/**/history.yaml', recursive=True)

for f in files:
    try:
        print(f"Processing file: {f}")
        x = yaml_load(f)
        num_epochs = len(x['accuracy'])
        splitted = f.split('/')
        name = f'acc-{splitted[3]}-GCN-{splitted[5]}-{splitted[6]}-{num_epochs}epochs-{splitted[4]}.png'
        path = f'/home/nonroot/experiment/results/{name}'
        generate_plot(x, output=path, title='Model Accuracy', xlabel='Epoch', ylabel='Accuracy', name_keys=[('accuracy', 'Train'), ('val_accuracy', 'Accuracy')])
    except Exception as e:
        print(f'Error in file {f}: {e}')
        continue

df = pd.DataFrame(columns=['dataset', 'graph', 'representation', 'index', 'train accuracy', 'validation accuracy'])
i = 0

for f in files:
    try:
        print(f"Processing file: {f}")
        x = yaml_load(f)
        acc = max(x['accuracy'])
        val_acc = max(x['val_accuracy'])
        splitted = f.split('/')
        df.loc[i] = [splitted[3], splitted[5], splitted[6], splitted[4], acc, val_acc]
        i+=1
    except Exception as e:
        print(f'Error in file {f}: {e}')
        continue
