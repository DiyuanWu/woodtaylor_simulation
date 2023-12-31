from matplotlib import pyplot as plt
import numpy as np

import torch


from torchvision import datasets, transforms
from torch.utils.data import DataLoader

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





file_name_loss = "./topkwt_slinearreg_mnist_loss_{}steps_{}exprs.npy".format( num_steps, num_expr)

file_name_dist = "./topkwt_slinearreg_mnist_dist_{}steps_{}exprs.npy".format( num_steps, num_expr)

file_name_loss_kiht = "./kiht_slinearreg_mnist_loss_{}steps_{}exprs.npy".format( num_steps, num_expr)

file_name_dist_kiht = "./kiht_slinearreg_mnist_dist_{}steps_{}exprs.npy".format( num_steps, num_expr)

file_name_origin_signals = "./slinearreg_mnist_orisigals.npy"

file_name_recov_signals = "./topkwt_slinearreg_mnist_recsigals.npy"

file_name_recov_signals_kiht = "./kiht_slinearreg_mnist_recsigals.npy"


loss_topkwt = np.load(file_name_loss)

loss_kiht = np.load(file_name_loss_kiht)

dist_topkwt = np.load(file_name_dist)

dist_kiht = np.load(file_name_dist_kiht)


origin_signals = np.load(file_name_origin_signals)

recovered_signals =  np.load(file_name_recov_signals) 

recovered_signals_kiht =  np.load(file_name_recov_signals_kiht)

print((np.count_nonzero(recovered_signals_kiht[0]), np.count_nonzero( recovered_signals[0] )))


# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Download and load the training dataset
train_dataset = datasets.MNIST(root='./dataset', train=True, transform=transform, download=True)

# Create a DataLoader to iterate over the training dataset
train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)


num_dataset = train_dataset.data.shape[0]

mnist_shape = train_dataset.data[0].shape

d = train_dataset.data[0].view(-1).shape[0]


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

figname = 'slinreg_mnist.png'

plt.savefig(figname)



fig, ax = plt.subplots(5,3, figsize= (10,6))

for i in range(5):

    ax[i, 0].imshow( origin_signals[i].astype(int).reshape(mnist_shape), cmap = 'gray', vmin=0, vmax = 255 )

    ax[i, 1].imshow( recovered_signals[i].astype(int).reshape(mnist_shape), cmap = 'gray', vmin=0, vmax = 255  )

    ax[i, 2].imshow( recovered_signals_kiht[i].astype(int).reshape(mnist_shape), cmap = 'gray' , vmin=0, vmax = 255 )



ax[0,0].set_title( 'origin signal' )

ax[0,1].set_title( ' recovered by topk-WT ' )

ax[0,2].set_title( ' recovered by k-IHT') 

fig.tight_layout()

figname = 'slinreg_mnist_recovery.png'

plt.savefig(figname)

    

