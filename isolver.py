#用于测试基线方法
from turtle import shape
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from utils.utils import *

from data_factory.data_loader_STAT_backup import get_loader_segment

from tqdm import tqdm
import torch.autograd as autograd
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, precision_recall_curve, roc_auc_score

from src.eval_methods import pot_eval, bf_search
from src.diagnosis import *
from pprint import pprint

# from optimizer import *
from time import time

from sklearn.preprocessing import MinMaxScaler

class EarlyStopping:
    def __init__(self, model_name, patience=3, verbose=False, dataset_name='', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name
        self.model_name = model_name

    def __call__(self, val_loss, model, optimizer, scheduler, epoch, accuracy_list):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_model(self, model, optimizer, scheduler, epoch, accuracy_list)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # if self.verbose:
                # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_model(self, model, optimizer, scheduler, epoch, accuracy_list)
            self.counter = 0
    def save_model(self, val_loss, model, optimizer, scheduler, epoch, accuracy_list):
        # if self.verbose:
        #     print('Validation loss decreased (%f --> %f).  Saving model ...'%(float(self.val_loss_min),float(val_loss)))
        print('Saving model ...')
        folder = f'compare/checkpoints/{self.model_name}_{self.dataset}/'
        os.makedirs(folder, exist_ok=True)
        file_path = f'{folder}/model.ckpt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'accuracy_list': accuracy_list}, file_path)
        self.val_loss_min = val_loss
            

#Use GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class color:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    RED = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

#=======f1======
def get_f_score(prec, rec):
    if prec == 0 and rec == 0:
        f_score = 0
    else:
        f_score = 2 * (prec * rec) / (prec + rec)
    return f_score


def get_y_pred_label(y_pred,threshold):
    y_pred_label = [1.0 if (score > threshold) else 0 for score in y_pred ]
    return y_pred_label

def get_best_f1_threshold(y_test, score_t_test):
    print(y_test.shape,score_t_test.shape)
    prec, rec, thres = precision_recall_curve(y_test, score_t_test, pos_label=1)
    fscore = [get_f_score(precision, recall) for precision, recall in zip(prec, rec)]
    opt_num = np.squeeze(np.argmax(fscore))
    opt_thres = thres[opt_num]
    pred_labels = np.where(score_t_test > opt_thres, 1, 0)
    return opt_thres,pred_labels,prec, rec, thres

def print_p_r_f1_label(y_test,y_pred_label):
    prec=precision_score(y_test,y_pred_label,pos_label=1)
    recall=recall_score(y_test,y_pred_label,pos_label=1)
    f1=f1_score(y_test,y_pred_label,pos_label=1)
    print('precision=',prec)
    print('recall=',recall)
    print('f1=',f1)

def print_p_r_fb1_label(y_test, score_t_test):
    opt_thres,pred_labels,prec, rec, thres = get_best_f1_threshold(y_test, score_t_test)
    print_p_r_f1_label(y_test,pred_labels)
#===============




# def adjust_learning_rate(optimizer, epoch, lr_):
#     lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
#     if epoch in lr_adjust.keys():
#         lr = lr_adjust[epoch]
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = lr
#         print('Updating learning rate to {}'.format(lr))
        
class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):

        self.__dict__.update(Solver.DEFAULTS, **config)

        _, self.train_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                               mode='train',
                                               dataset=self.dataset)
        self.cur_dataset , self.test_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='test',
                                              dataset=self.dataset)
        # self.build_model()
        self.device = torch.device(device)
        self.criterion = nn.MSELoss()

    def load_model(self, modelname, dims):
        import model.models
        model_class = getattr(model.models, modelname)
        if 'Omni' in modelname:
            model = model_class(dims,self.win_size).double()
        else:
            model = model_class(dims).double()
        optimizer = torch.optim.AdamW(model.parameters() , lr=model.lr, weight_decay=1e-5)
        # optimizerW = torch.optim.Adam(model.parameters() , lr=model.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
        fname = f'compare/checkpoints/{self.model}_{self.dataset}/model.ckpt'
        if os.path.exists(fname) and (self.mode == 'test'):
            print(f"{color.GREEN}Loading pre-trained model: {model.name}{color.ENDC}")
            checkpoint = torch.load(fname)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            epoch = checkpoint['epoch']
            accuracy_list = checkpoint['accuracy_list']
        else:
            print(f"{color.GREEN}Creating new model: {model.name}{color.ENDC}")
            epoch = -1; accuracy_list = []
        return model, optimizer, scheduler, epoch, accuracy_list



    def train(self):
        model, optimizer, scheduler, epoch, accuracy_list = self.load_model(self.model, self.input_c)
        model.to(device)
        model.train()

        print("======================TRAIN MODE======================")
        
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        early_stopping = EarlyStopping(self.model, patience=1000, verbose=True, dataset_name=self.dataset)

        rec_losses = []

        accuracy_list = []
        num_epochs = self.num_epochs; e = epoch + 1; start = time()
        print(f'with epochs {self.num_epochs}')
        loss = None
        for epoch in tqdm(range(self.num_epochs)):       
            if 'TimeSeriesTransformer' in self.model:
                for i, (input_data, labels) in enumerate(self.train_loader):
                    # print('here', input_data.shape)
                    # if 'USAD' in model.name:
                    # data = input_data
                    l = nn.MSELoss(reduction = 'mean')# if training else 'none')
                    n = epoch + 1; w_size = self.win_size
                    mses, klds = [], []
                    input_data = input_data.type(torch.DoubleTensor).to(device)
                    for i, d in enumerate(input_data): 
                        y_pred = model(d)
                        MSE = l(y_pred, d)
                        loss = MSE
                        mses.append(torch.mean(MSE).item());
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(mses)},\tLoss = {np.mean(loss.item())}')
                loss = np.mean(loss.item())
                # accuracy_list.append((np.mean(l1s) + np.mean(l2s), optimizer.param_groups[0]['lr']))
            elif 'MAD_GAN' in self.model:
                for i, (input_data, y_input_data) in enumerate(self.train_loader):
                    # print(input_data.shape,'input_data')
                    l = nn.MSELoss(reduction = 'none')
                    bcel = nn.BCELoss(reduction = 'mean')
                    msel = nn.MSELoss(reduction = 'mean')
                    real_label, fake_label = torch.tensor([0.9]), torch.tensor([0.1]) # label smoothing
                    real_label, fake_label = real_label.type(torch.DoubleTensor).to(device), fake_label.type(torch.DoubleTensor).to(device)
                    n = epoch + 1; w_size = model.n_window
                    mses, gls, dls = [], [], []
                    for d in input_data:
                        # training discriminator
                        model.discriminator.zero_grad()
                        d = d.type(torch.DoubleTensor).to(device)
                        _, real, fake = model(d)
                        dl = bcel(real, real_label) + bcel(fake, fake_label)
                        dl.backward()
                        model.generator.zero_grad()
                        optimizer.step()
                        # training generator
                        z, _, fake = model(d)
                        mse = msel(z, d.view(1,-1)) 
                        gl = bcel(fake, real_label)
                        tl = gl + mse
                        tl.backward()
                        model.discriminator.zero_grad()
                        optimizer.step()
                        mses.append(mse.item()); gls.append(gl.item()); dls.append(dl.item())
                        # tqdm.write(f'Epoch {epoch},\tMSE = {mse},\tG = {gl},\tD = {dl}')
                tqdm.write(f'Epoch {epoch},\tLoss = {np.mean(mses+gls)},\tMSE = {np.mean(mses)},\tG = {np.mean(gls)},\tD = {np.mean(dls)}')
                loss = np.mean(mses+gls)
                # return np.mean(gls)+np.mean(dls), optimizer.param_groups[0]['lr']
            elif 'OmniAnomaly' in self.model:
                for i, (input_data, labels) in enumerate(self.train_loader):
                    # print('here', input_data.shape)
                    # if 'USAD' in model.name:
                    # data = input_data
                    l = nn.MSELoss(reduction = 'mean')# if training else 'none')
                    n = epoch + 1; w_size = self.win_size
                    mses, klds = [], []
                    input_data = input_data.type(torch.DoubleTensor).to(device)
                    for i, d in enumerate(input_data): 
                        y_pred, mu, logvar, hidden = model(d, hidden if i else None) #RuntimeError: input.size(-1) must be equal to input_size. Expected 38, got 228
                        MSE = l(y_pred, d)
                        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=0)
                        loss = MSE + model.beta * KLD
                        mses.append(torch.mean(MSE).item()); klds.append(model.beta * torch.mean(KLD).item())
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(mses)},\tKLD = {np.mean(klds)},\tLoss = {np.mean(loss.item())}')
                loss = np.mean(loss.item())
                # accuracy_list.append((np.mean(l1s) + np.mean(l2s), optimizer.param_groups[0]['lr']))
            elif 'USAD' in self.model:
                for i, (input_data, labels) in enumerate(self.train_loader):
                    # print('here', input_data.shape)
                    # if 'USAD' in model.name:
                    # data = input_data
                    l = nn.MSELoss(reduction = 'none')
                    n = epoch + 1; w_size = self.win_size
                    l1s, l2s = [], []
                    for d in input_data:
                        d = d.type(torch.DoubleTensor).to(device)
                        d = d.view(1,-1)
                        # print(d.shape)
                        feats = self.input_c
                        ae1s, ae2s, ae2ae1s = model(d)
                        l1 = (1 / n) * l(ae1s, d) + (1 - 1/n) * l(ae2ae1s, d)
                        l2 = (1 / n) * l(ae2s, d) - (1 - 1/n) * l(ae2ae1s, d)
                        l1s.append(torch.mean(l1).item()); l2s.append(torch.mean(l2).item())
                        loss = torch.mean(l1 + l2)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    # scheduler.step()
                tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)},\tL2 = {np.mean(l2s)},\tL1 + L2 = {np.mean(loss.item())}')
                loss = np.mean(loss.item())
                # accuracy_list.append((np.mean(l1s) + np.mean(l2s), optimizer.param_groups[0]['lr']))
            elif 'DAGMM' in model.name:
                for i, (input_data, y_input_data) in enumerate(self.train_loader):
                    l = nn.MSELoss(reduction = 'none')
                    n = epoch + 1; w_size = model.n_window
                    l1s = []; l2s = []
                    input_data = input_data.type(torch.DoubleTensor).to(device)
                    for d in input_data:
                        _, x_hat, z, gamma = model(d)
                        # l1, l2 = l(x_hat, d), l(gamma, d)
                        l1, l2 = l(x_hat, d.view(-1)), l(gamma, d.view(-1))
                        l1s.append(torch.mean(l1).item()); l2s.append(torch.mean(l2).item())
                        loss = torch.mean(l1) + torch.mean(l2)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    # scheduler.step()
                tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)},\tL2 = {np.mean(l2s)}')
            elif 'MSCRED' in model.name:
                for i, (input_data, y_input_data) in enumerate(self.train_loader):
                    l = nn.MSELoss(reduction = 'none')
                    n = epoch + 1; w_size = model.n_window
                    l1s = []
                    input_data = input_data.type(torch.DoubleTensor).to(device)
                    for d in input_data:
                        for i in range(d.shape[0]):
                            # print(d.shape)
                            x = model(d)
                            # print('a',x.shape, d.shape)
                            loss = torch.mean(l(x, d.view(-1)))
                        l1s.append(torch.mean(loss).item())
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(l1s)}')

            early_stopping(loss, model, optimizer, scheduler, epoch, accuracy_list)
            if early_stopping.early_stop:
                print("Early stopping with patience ", early_stopping.patience)
                break
        print(color.BOLD+'Training time: '+"{:10.4f}".format(time()-start)+' s'+color.ENDC)
        # print(f'Saving model ...')
        # self.save_model(model, optimizer, scheduler, epoch, accuracy_list)
        

    def test(self):
        model, optimizer, scheduler, epoch, accuracy_list = self.load_model(self.model, self.input_c)

        model.to(device)
        model.eval()

        print("======================TEST MODE======================")

        test_labels = []
        y_all = []
        test_exp_labels=[]
        attens_energy = []
        ds = []
        ds = []
        l = nn.MSELoss(reduction = 'none')
        
        for i, (input_data, y_input_data, lable_input_data) in enumerate(self.test_loader):
            data = input_data.to(device)
            lable_input = lable_input_data#.to(device)
            if 'MAD_GAN' in self.model:
                outputs = []
                for d in data: 
                    torch.set_grad_enabled(False)
                    d = d.type(torch.DoubleTensor).to(device)
                    z, _, _ = model(d)
                    outputs.append(z.view(self.win_size,-1))
                outputs = torch.stack(outputs)
                loss = l(outputs, data)#(b,w,n)
                loss = torch.mean(loss, dim=1) #(b,n)
                exp_loss = loss #(b,n)
                loss = torch.mean(loss, dim=-1) #(b)

            elif 'OmniAnomaly' in self.model:
                y_preds = []
                ds = []
                da = []
                for i, d in enumerate(data):
                    torch.set_grad_enabled(False)
                    d = d.type(torch.DoubleTensor).to(device)
                    y_pred, _, _, hidden = model(d, hidden if i else None)
                    # print('a',y_pred.shape,d[:-1,:].shape)
                    y_preds.append(y_pred)
                    ds.append(d[-1,:])
                y_pred = torch.stack(y_preds)
                da = torch.stack(ds)
                # print('da',da.shape,y_pred.shape)
                # print('a',y_pred.shape)a torch.Size([256, 25])
                # print('b',d.shape)b torch.Size([6, 25])
                MSE = l(y_pred, da.view(y_pred.shape[0], -1))
                loss = torch.mean(MSE, dim=-1)

            elif 'USAD' in self.model:
                ae1s, ae2s, ae2ae1s = [], [], []
                feats = self.input_c
                # print('here', data.shape)
                for d in data:
                    torch.set_grad_enabled(False)
                    d = d.view(1,-1)
                    ae1, ae2, ae2ae1 = model(d)
                    ae1s.append(ae1.view(self.win_size,-1)); ae2s.append(ae2.view(self.win_size,-1)); ae2ae1s.append(ae2ae1.view(self.win_size,-1))
                ae1s, ae2s, ae2ae1s = torch.stack(ae1s), torch.stack(ae2s), torch.stack(ae2ae1s)
                loss = (0.1 * l(ae1s, data) + 0.9 * l(ae2ae1s, data)) #(b,w,n)
                loss = torch.mean(loss, dim=1) #(b,w)
                # exp_loss = loss
                loss = torch.mean(loss, dim=-1)#(b)
            elif 'DAGMM' in self.model:
                ae1s = []
                for d in data: 
                    _, x_hat, _, _ = model(d)
                    ae1s.append(x_hat)
                ae1s = torch.stack(ae1s)
                # print(ae1s.shape,data.shape) #torch.Size([32, 228]) torch.Size([32, 6, 38])
                loss = l(ae1s.view(-1, *(self.win_size,self.input_c)), data)[:, data.shape[1]-self.input_c:data.shape[1]]#(b,w,n)
                loss = torch.mean(loss, dim=1) #(b,w)
                # exp_loss = loss
                loss = torch.mean(loss, dim=-1) #(b)
            elif 'MSCRED' in self.model:
                xs = []
                for d in data: 
                    x = model(d)
                    xs.append(x)
                xs = torch.stack(xs)
                # print(ae1s.shape,data.shape) #torch.Size([32, 228]) torch.Size([32, 6, 38])
                loss = l(xs.view(-1, *(self.win_size,self.input_c)), data)[:, data.shape[1]-self.input_c:data.shape[1]]#(b,w,n)
                loss = torch.mean(loss, dim=1) #(b,w)
                # exp_loss = loss
                loss = torch.mean(loss, dim=-1) #(b)
            # print(loss.shape)
            attens_energy.append(loss.detach().cpu().numpy())
            for lable in lable_input:
                test_labels.append(lable)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)

        self.scaler = MinMaxScaler()
        # test_energy_veiw = np.array().view(-1, 1)
        self.scaler.fit(test_energy.reshape(-1,1))
        test_energy = self.scaler.transform(test_energy.reshape(-1,1))

        score = test_energy
        # label = test_labels[:-1]
        label = np.array(test_labels)

        print('aa',score.shape,label.shape)
        bf_eval = bf_search(score, label, start=0.001, end=1, step_num=150, verbose=False)
        pprint(bf_eval)