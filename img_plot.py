import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def load_data(file, ax, title):
    with open(file, 'r') as f:
        records = f.readlines()[2:]
    train_loss = []
    test_loss = []
    for record in records:
        items = record.strip('\n').split('\t')
        train_loss.append(float(items[-3]))
        test_loss.append(float(items[-9]))
    ax.plot(train_loss, label='train robust loss')
    ax.plot(test_loss, label='test robust loss')
    font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 18}
    ax.set_title(title, fontsize=18, y=1.03)
    ax.grid()
    ax.legend(prop=font)


def ploting():
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    load_data('baseline.log', axes[0], 'Robust Loss of baseline')
    load_data('cutmix_m4.log', axes[1], 'Robust Loss of our methods')
    plt.savefig('./res.jpg')
    plt.show()


def ploting2():
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
    cutmix = [(52.80, 54.96), (55.17, 55.55), (55.12, 56.10), (55.65, 56.17), (54.32, 55.05)]
    mixup = [(49.49, 51.14), (54.05, 55.12), (54.10, 55.73)]
    for idx, (b,f) in enumerate(cutmix):
        ax[0].scatter(f, b, label=f'm={idx+1}')
    for idx, (b,f) in enumerate(mixup):
        ax[1].scatter(f, b, label=f'm={idx+1}')

    def to_percent(temp, position):
        return '%3.1f' % temp + '%'

    ax[0].yaxis.set_major_formatter(ticker.FuncFormatter(to_percent))
    ax[0].xaxis.set_major_formatter(ticker.FuncFormatter(to_percent))
    ax[1].yaxis.set_major_formatter(ticker.FuncFormatter(to_percent))
    ax[1].xaxis.set_major_formatter(ticker.FuncFormatter(to_percent))
    ax[0].set_title('with cutmix')
    ax[1].set_title('with mixup')
    ax[1].set_xlabel('Best Robust Accuracy')
    ax[0].set_ylabel('Robust Accuracy')
    ax[0].set_ylabel('Final Robust Accuracy')
    ax[1].set_ylabel('Final Robust Accuracy')
    ax[0].grid()
    ax[1].grid()
    ax[0].legend()
    ax[1].legend()
    plt.savefig('subplots.png')
    plt.show()


if __name__ == "__main__":
    #ploting(['test.log', 'output.log'])
    ploting()