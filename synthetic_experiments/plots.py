from matplotlib import pyplot as plt
import numpy as np

num_expr = 20

n = 256

d = 128

k_star = 16

k = 64 # The sparsity of weights during training

    
num_steps = 750


file_name_loss = "./topkwt_slinearreg_loss_{}steps_{}exprs_n{}_d{}_k{}_kstar{}.npy".format( num_steps, num_expr, n,d,k,k_star)

file_name_dist = "./topkwt_slinearreg_dist_{}steps_{}exprs_n{}_d{}_k{}_kstar{}.npy".format( num_steps, num_expr, n,d,k,k_star)

file_name_loss_kiht = "./kiht_slinearreg_loss_{}steps_{}exprs_n{}_d{}_k{}_kstar{}.npy".format( num_steps, num_expr, n,d,k,k_star)

file_name_dist_kiht = "./kiht_slinearreg_dist_{}steps_{}exprs_n{}_d{}_k{}_kstar{}.npy".format( num_steps, num_expr, n,d,k,k_star)


loss_topkwt = np.load(file_name_loss)

loss_kiht = np.load(file_name_loss_kiht)

dist_topkwt = np.load(file_name_dist)

dist_kiht = np.load(file_name_dist_kiht)


loss_topkwt_mean = np.mean(loss_topkwt, axis = 0)

loss_topkwt_var = np.var(loss_topkwt, axis = 0)

loss_kiht_mean = np.mean(loss_kiht, axis = 0)

loss_kiht_var = np.var(loss_kiht, axis = 0)

dist_topkwt_mean = np.mean(dist_topkwt, axis = 0)

dist_topkwt_var = np.var(dist_topkwt, axis = 0)

dist_kiht_mean = np.mean(dist_kiht, axis = 0)

dist_kiht_var = np.var(dist_kiht, axis = 0)




plot_t = np.linspace(0,750,750)

fig, ax = plt.subplots(1,2, figsize= (10.5,5))

ax[0].plot(plot_t, loss_topkwt_mean, label='topk-WT', color = 'blue')

ax[0].fill_between(plot_t, loss_topkwt_mean-np.sqrt(loss_topkwt_var), loss_topkwt_mean+np.sqrt(loss_topkwt_var), fc='blue', alpha = 0.3 )

ax[0].plot(plot_t, loss_kiht_mean, label='k-IHT', color = 'red')

ax[0].fill_between(plot_t, loss_kiht_mean-np.sqrt(loss_kiht_var), loss_kiht_mean+np.sqrt(loss_kiht_var), fc='red', alpha = 0.3 )

ax[0].set_xlabel('# of steps')

ax[0].set_ylabel('Loss')

ax[0].legend()


ax[1].plot(plot_t, dist_topkwt_mean, label='topk-WT', color = 'blue')

ax[1].fill_between(plot_t, dist_topkwt_mean-np.sqrt(dist_topkwt_var), dist_topkwt_mean+np.sqrt(dist_topkwt_var), fc='blue', alpha = 0.3 )

ax[1].plot(plot_t, dist_kiht_mean, label='k-IHT', color = 'red')

ax[1].fill_between(plot_t, dist_kiht_mean-np.sqrt(dist_kiht_var), dist_kiht_mean+np.sqrt(dist_kiht_var), fc='red', alpha = 0.3 )

ax[1].set_xlabel('# of steps')

ax[1].set_ylabel('L2 distance to the optimal solution')

ax[0].legend()

ax[1].legend()

figname = 'slinreg.png'

plt.savefig(figname)