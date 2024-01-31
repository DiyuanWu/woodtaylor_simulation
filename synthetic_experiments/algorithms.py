import torch


def topk(vector_in, k):

    in_shape = vector_in.shape

    vector = vector_in.view(-1)

    # Find the indices of the top k elements with the largest absolute values
    top_k_indices = torch.topk(torch.abs(vector), k=k).indices

    # Create a mask with zeros and ones
    mask = torch.zeros_like(vector)
    mask[top_k_indices] = 1

    # Apply the mask to zero out elements other than the top k
    result_vector = vector * mask

    result_vector = result_vector.view(in_shape)

    return result_vector, mask





def OBC(w_in, H_inv_in,d,k):

    '''
    The computation of optimal brain compression algorithm 
    '''
    in_shape = w_in.shape

    w = w_in

    H_inv = H_inv_in

    mask = torch.ones(w.shape).view(-1)

    eps = 1e-3

    for i in range(d-k):

        diag_hinv = torch.diag(H_inv)

        diag_hinv_temp = diag_hinv + eps

        #print(diag_hinv)

        # add some maximum value to w where it is zero
        ### mask_w_zero = (w == 0) # mask saying where w is zero
        ### w[mask_w_zero] = w.max() + 1

        #print(diag_hinv)
        val =  torch.div(w**2, diag_hinv_temp ).view(-1)

        val[val == 0] = val.max() +100
        ### w.mul_(mask_w_zero) # restore zeros in w


        p = torch.argmin(val).item()

        hessian_inv_col_p = H_inv[:, p].view(-1,1)
        hessian_inv_row_p = H_inv[p, :].view(1,-1)

        #print(hessian_inv_col_p.shape)
        
        mask[p] = 0
        
        hessian_inv_pp = diag_hinv_temp[p]

        # update w

        # print(w)
        w_update = (w[0,p] / hessian_inv_pp) * hessian_inv_col_p.view(in_shape)
        w.sub_(w_update)

        # update Hinv

        H_inv.sub_((1/hessian_inv_pp) * hessian_inv_col_p @ hessian_inv_row_p)

        # update mask
        w.mul_(mask)

    return w, mask


