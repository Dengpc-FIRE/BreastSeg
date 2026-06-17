
def adjust_learning_rate(epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = 0.0012 * (0.5 ** (epoch // 50))
    print(lr)

if __name__ == '__main__':
    for i in range(300,500):
        adjust_learning_rate(i)
