import numpy as np

def main():

    trainval_num = 7481
    train_ratio = 0.85

    train_list =[]
    val_list =[]
    for i in range(trainval_num):
        r = np.random.random()
        if r <= train_ratio:
            train_list.append(i)
        else:
            val_list.append(i)

    train_list = np.asarray(train_list)
    val_list = np.asarray(val_list)

    np.savetxt('./train.txt', train_list, fmt='%06d')
    np.savetxt('./val.txt', val_list, fmt='%06d')


if __name__ == '__main__':
    main()
