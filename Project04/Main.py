import matplotlib.pyplot as plt
import seaborn as sb
import TestingModel as tm
import LoadData as ld
import pandas as pd

data_train_X, data_test_X, data_train_y, data_test_y = ld.hold_out()

train_data = data_train_X.join(data_train_y)
train_data.hist(figsize=(20, 15))
plt.show()


plt.figure(figsize=(20, 10))
sb.heatmap(train_data.corr(), annot=True, cmap='YlGnBu')
plt.show()


classes = ['p', 'e']
counts = [4208, 3916]
colors = ['blue', 'orange']

bars = plt.bar(classes, counts, color=colors)
plt.legend(bars, classes)

for bar, count in zip(bars, counts):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(count), ha='center', va='bottom')

plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Samples of class')
plt.show()

pie, texts, autotexts = plt.pie(counts, labels=classes, colors=colors, autopct='%1.1f%%', startangle=90)
plt.legend(pie, classes, title="Classes", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
plt.title('Samples of class')
plt.show()


knn_acc, bayes_acc, tree_acc, gd_acc = tm.testing()

classes = ["KNN", "Bayes", "DT", "GD"]
counts = [knn_acc, bayes_acc, tree_acc, gd_acc]

data = pd.DataFrame({'Class': classes, 'Count': counts})

ax = sb.barplot(x='Class', y='Count', data=data, palette='viridis', legend=False)

for p in ax.patches:
    ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points')

plt.title("Accuracy score of 3 Models")
plt.show()