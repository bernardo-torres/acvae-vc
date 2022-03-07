import torch
import time
from data import data_load

def train(model, num_epochs, learning_rate_, learning_rate__, learning_rate___, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    #model = ACVAE(nb_label=nb_label,lambda_p=lambda_p,lambda_s=lambda_s).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()

    losses = []

    for epoch in range(num_epochs):
        epoch += 1
        
        if (epoch == 3000):
            learning_rate = learning_rate_ 
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
        if (epoch == 6000):
            learning_rate = learning_rate__
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
        if (epoch == 8000):
            learning_rate = learning_rate___
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
        
        start_time = time.time()

        print('Epoch: %d' % epoch)

        x_, label_ = data_load(batch_size)
        optimizer.zero_grad()
        loss, loss_list = model.calc_loss(x_, label_)
        loss.backward()
        losses.append(loss_list)
        optimizer.step()
        
        if epoch % 100 == 0:
            test_conv(model, epoch)
        if epoch % 100 == 0:
            model_save(model, model_dir, model_name)
        if epoch % 2000 == 0:
            model_save(model, model_dir, model_name + "_" + str(epoch))

        if epoch % 100 == 0:
            save_figure(losses, epoch)
        
        elapsed_time = time.time() - start_time
        print('Time Elapsed for one epoch: %02d:%02d:%02d' % (elapsed_time // 3600, (elapsed_time % 3600 // 60), (elapsed_time % 60 // 1)))

    model_save(model, model_dir, model_name)

    save_figure(losses, epoch)
