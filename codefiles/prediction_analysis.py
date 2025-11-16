import matplotlib.pyplot as plt
import pandas as pd

def plot_prediction_accuracy_per_class(y_true, y_pred, class_labels=None, figsize=(8,6), title='Correct vs Incorrect Predictions per Class'):
    results_df = pd.DataFrame({'Actual': y_true, 'Predicted': y_pred})
    results_df['Correct'] = results_df['Actual'] == results_df['Predicted']

    correct_counts = results_df[results_df['Correct']].groupby('Actual').size()
    incorrect_counts = results_df[~results_df['Correct']].groupby('Actual').size()

    # Ensure all classes are represented
    all_classes = sorted(set(y_true) | set(y_pred))
    correct_counts = correct_counts.reindex(all_classes, fill_value=0)
    incorrect_counts = incorrect_counts.reindex(all_classes, fill_value=0)

    # Plotting
    plt.figure(figsize=figsize)
    plt.bar([i - 0.2 for i in all_classes], correct_counts.values, width=0.4, label='Correct', color='green')
    plt.bar([i + 0.2 for i in all_classes], incorrect_counts.values, width=0.4, label='Incorrect', color='red')
    plt.xticks(all_classes, class_labels if class_labels else all_classes)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()
