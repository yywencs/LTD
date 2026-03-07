import os
import time
from tensorboardX import SummaryWriter

from validate import validate
from data import create_dataloader
from earlystop import EarlyStopping
from networks.trainer import Trainer
from options.train_options import TrainOptions
from data.datasets import RealFakeDataset
import numpy as np
import torch
import random


def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.isTrain = False
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True
    val_opt.data_label = 'val'
    val_opt.jpg_method = ['pil']
    if len(val_opt.blur_sig) == 2:
        b_sig = val_opt.blur_sig
        val_opt.blur_sig = [(b_sig[0] + b_sig[1]) / 2]
    if len(val_opt.jpg_qual) != 1:
        j_qual = val_opt.jpg_qual
        val_opt.jpg_qual = [int((j_qual[0] + j_qual[-1]) / 2)]

    return val_opt

def set_seed(SEED):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    

if __name__ == '__main__':
    opt = TrainOptions().parse()
    val_opt = get_val_opt()
    set_seed(3407)

    model = Trainer(opt)
    
    train_dataset = RealFakeDataset(opt)
    data_loader = create_dataloader(opt, None, train_dataset)
    val_loader = create_dataloader(val_opt)


    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))
    val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val"))
    save_path = os.path.join(opt.checkpoints_dir, opt.name)
    
    early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.001, verbose=True)
    start_time = time.time()
    training_time = 0
    print ("Length of data loader: %d" %(len(data_loader)))
    mx_acc = 0
    
    for epoch in range(1, opt.niter + 1):
        training_epoch_start = time.time()
        for i, data in enumerate(data_loader):
            model.total_steps += 1

            model.set_input(data)
            model.optimize_parameters()
            
            print(
                f"\r[Epoch {epoch:02d}/{opt.niter:02d} | Batch {i + 1:04d}/{len(data_loader):04d} | Time {training_time + time.time() - start_time:1.1f}s] loss: {model.loss.item():1.4f}",
                end="",
            )
            train_writer.add_scalar('loss', model.loss, model.total_steps)
            

            if model.total_steps in [10,30,50,100,1000,5000,10000] and False: # save models at these iters 
                model.save_networks('model_iters_%s.pth' % model.total_steps)

        training_time += time.time() - training_epoch_start
    
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d' % (epoch))
            model.save_networks( f'model_epoch_{epoch}.pth')

        # Validation
        model.eval()
        
        ap, r_acc, f_acc, acc = validate(model.model, val_loader)
        val_writer.add_scalar('accuracy', acc, model.total_steps)
        val_writer.add_scalar('ap', ap, model.total_steps)
        print("(Val @ epoch {}) acc: {}; ap: {}".format(epoch, acc, ap))

        early_stopping(acc, model)
        if early_stopping.early_stop:
            cont_train = model.adjust_learning_rate()
            if cont_train:
                print("Learning rate dropped by 10, continue training...")
                early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.002, verbose=True)
            else:
                print("Early stopping.")
                break
        model.train()

