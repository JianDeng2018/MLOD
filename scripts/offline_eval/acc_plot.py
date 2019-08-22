import os
import numpy as np
import matplotlib.pyplot as plt

def main():
    base_dir = '/media/j7deng/Data/track/avod_results/m_scale2/'
    checkpointname = 'avod_fpn_car_results_0.1'
    result_main = base_dir + checkpointname + '_main.txt'
    result_br0 = base_dir + checkpointname + '_br_0.txt'
    result_br1 = base_dir + checkpointname + '_br_1.txt'
    result_br2 = base_dir + checkpointname + '_br_2.txt'
    result_br3 = base_dir + checkpointname + '_br_3.txt'
    result_br4 = base_dir + checkpointname + '_br_4.txt'
    result_list = [result_main,result_br0,result_br1,result_br2,result_br3, result_br4]

    field = 'pedestrian_detection_3D'

    acc_list = []
    steps_list = []

    for file in result_list:
        f = open(file,"r")
        line = f.readlines()
        f.close()
        
        acc = []
        steps = []
        for l in line:
            l_split = l.split( )
            #accuracy
            if l_split[0] == field:
                acc.append(float(l_split[4]))
            # global steps
            if l_split[0].isdigit():
                steps.append(int(l_split[0]))
        acc_list.append(acc)
        steps_list.append(steps)

    plt.plot(steps_list[0],acc_list[0],label='main')
    plt.plot(steps_list[1],acc_list[1],label='Bev')
    plt.plot(steps_list[2],acc_list[2],label='Image1')
    plt.plot(steps_list[3],acc_list[3],label='Image2')
    plt.plot(steps_list[4],acc_list[4],label='Image3')
    plt.plot(steps_list[5],acc_list[5],label='Image3')

    plt.legend()
    plt.show()



if __name__=='__main__':
    main()
