import matplotlib.pyplot as plt


class CommonFunction:
    def __init__(self):
        print('Common Function')
        
    def plot_training(self, history, title='Training and Validation Accuracy'):
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(len(acc))
    
        plt.plot(epochs, acc, 'g.', label='accuracy')
        plt.plot(epochs, val_acc, 'g', label='val_acc')
        plt.title(title)
    
        plt.plot(epochs, loss, 'r.', label='loss')
        plt.plot(epochs, val_loss, 'r-', label='val_loss')
        plt.legend()
        plt.show()
    
        plt.savefig('acc_vs_epochs.png')
