import matplotlib.pyplot as plt


class CommonFunction:
    def __init__(self):
        print('Common Function')
        
    def plot_training(self, history, title='TBE-CNN'):
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(len(acc))

        plt.subplot(2, 1, 1)
        plt.plot(epochs, acc, 'g.', label='accuracy')
        plt.plot(epochs, val_acc, 'g', label='val_acc')
        plt.title('Accuracy on Training & Validation, Model: ' + title)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        
        plt.subplots_adjust(bottom=0.2, hspace=1)

        plt.subplot(2, 1, 2)
        plt.plot(epochs, loss, 'r.', label='loss')
        plt.plot(epochs, val_loss, 'r-', label='val_loss')
        plt.title('Loss on Training & Validation, Model: ' + title)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')

        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        
        plt.draw()
        plt.savefig('acc_vs_epochs_' + title + '.png')
