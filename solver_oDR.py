import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from utils.utils import *
from model.WPS import Generator, Discriminator
from data_factory.data_loader_STAT_backup import get_loader_segment

from tqdm import tqdm
import torch.autograd as autograd
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, precision_recall_curve, roc_auc_score

from src.eval_methods import pot_eval, bf_search
from pprint import pprint

from utils.optimizer import *

class EarlyStopping:
    def __init__(self, model_save_path, patience=3, verbose=False, dataset_name='', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name
        self.model_save_path = model_save_path

    def __call__(self, val_loss, model_G, model_D, optimizer_G, optimizer_D, epoch):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_model(self, model_G, model_D, optimizer_G, optimizer_D, epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_model(self, model_G, model_D, optimizer_G, optimizer_D, epoch)
            self.counter = 0
    def save_model(self, val_loss, model_G, model_D, optimizer_G, optimizer_D, epoch):
        # if self.verbose:
        #     print('Validation loss decreased (%f --> %f).  Saving model ...'%(float(self.val_loss_min),float(val_loss)))
        print('Saving model ...')
        folder = f'{self.model_save_path}_{self.dataset}/'
        os.makedirs(folder, exist_ok=True)
        file_path = f'{folder}/model.ckpt'
        torch.save({
            'epoch': epoch,
            'model_G_state_dict': model_G.state_dict(),
            'model_D_state_dict': model_D.state_dict(),
            'optimizer_state_dict': optimizer_G.state_dict(),
            'scheduler_state_dict': optimizer_D.state_dict(),
            # 'accuracy_list': accuracy_list
            }, file_path)
        self.val_loss_min = val_loss

#Use GPU
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

lambda_gp = 10


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


class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):

        self.__dict__.update(Solver.DEFAULTS, **config)

        self.train_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                               mode='train',
                                               dataset=self.dataset)
        self.vali_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='val',
                                              dataset=self.dataset)
        self.test_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='test',
                                              dataset=self.dataset)
        self.thre_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='thre',
                                              dataset=self.dataset)
        
        print(self.win_size)

        # self.build_model()
        self.device = torch.device(device)
        self.criterion = nn.MSELoss()

    def vali(self, vali_loader):
        self.model.eval()

        loss_1 = []
        loss_2 = []
        for i, (input_data, _) in enumerate(vali_loader):
            input = input_data.double().to(device)
            output, series, prior, _ = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            series_loss = series_loss / len(prior)
            prior_loss = prior_loss / len(prior)

            rec_loss = self.criterion(output, input)
            loss_1.append((rec_loss - self.k * series_loss).item())
            loss_2.append((rec_loss + self.k * prior_loss).item())

        return np.average(loss_1), np.average(loss_2)

    def train(self):
        generator = Generator(win_size=self.win_size, latent_dim=self.latent_dim, output_c=self.output_c)
        discriminator = Discriminator(win_size=self.win_size, input_c=self.input_c, output_c=self.output_c)
        optimizer_G = torch.optim.Adam(generator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=self.lr, betas=(self.b1, self.b2))

        if torch.cuda.is_available():
            # self.generator.to(self.device)
            generator.to(self.device)
            discriminator.to(self.device)


        print("======================TRAIN MODE======================")

        
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        early_stopping = EarlyStopping(path, patience=15, verbose=False, dataset_name=self.dataset)


        rec_losses = []
        for epoch in tqdm(range(self.num_epochs)):
            for i, (input_data, labels) in enumerate(self.train_loader):

                optimizer_G.zero_grad()
                optimizer_D.zero_grad()
                
                real_input = input_data.float().to(self.device)
                z = torch.FloatTensor(np.random.normal(0, 1, (input_data.shape[0], self.latent_dim))).to(device)
                
                # Generate a batch of input
                fake_input = generator(z)

                # Real input
                real_validity = discriminator(real_input)
                # Fake input 
                fake_validity = discriminator(fake_input)
                # Gradient penalty
                gradient_penalty = compute_gradient_penalty(discriminator, real_input, fake_input)
                # Adversarial loss
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

                d_loss.backward()
                optimizer_D.step()

                optimizer_G.zero_grad()

                # Train the generator every n_critic steps
                if i % 5 == 0:

                    # -----------------
                    #  Train Generator
                    # -----------------

                    # Generate a batch of input
                    fake_input = generator(z)
                    # Loss measures generator's ability to fool the discriminator
                    # Train on fake input
                    fake_validity = discriminator(fake_input)
                    g_loss = -torch.mean(fake_validity)

                    g_loss.backward()
                    optimizer_G.step()
                    
                    rec_loss = self.criterion(fake_input, real_input)
                    rec_losses.append(rec_loss.detach().cpu())

            if epoch % 2 == 0:
                tqdm.write(
                    "Epoch: {0}, Steps: {1} | g_loss Loss: {2:.7f} d_loss Loss: {3:.7f} MSE: {4:.7f}".format(
                        epoch + 1, i, g_loss.item(), d_loss.item(), np.average(rec_losses)))
            loss = np.average(rec_losses)
            early_stopping(loss, generator, discriminator, optimizer_G, optimizer_D, epoch)
            if early_stopping.early_stop:
                print("Early stopping with patience ", early_stopping.patience)
                break

    def test(self):
        generator = Generator(win_size=self.win_size, latent_dim=self.latent_dim, output_c=self.output_c)
        discriminator = Discriminator(win_size=self.win_size, input_c=self.input_c, output_c=self.output_c)

        fname = f'{self.model_save_path}_{self.dataset}/model.ckpt'
        checkpoint = torch.load(fname)
        generator.load_state_dict(checkpoint['model_G_state_dict'])
        discriminator.load_state_dict(checkpoint['model_D_state_dict'])

        generator.to(device)
        discriminator.to(device)
        generator.eval()
        discriminator.eval()

        print("======================TEST MODE======================")
        criterion = nn.L1Loss(reduction='none')
        test_labels = []
        attens_energy = []

        inputs = []
        # outputs = []
        for i, (input_data, labels) in enumerate(self.test_loader):
            inputs.append(input_data)
            # print(input_data.shape)

            input = input_data.float().to(device)
            
            z = torch.FloatTensor(np.random.normal(0, 1, (input.shape[0], self.latent_dim))).to(device)
            output = generator(z)
            loss = torch.mean(criterion(input, output), dim=-1)
            # loss = (self.alpha)*torch.mean(criterion(input, output), dim=-1) + (1 - self.alpha) * (torch.mean((discriminator(input)), dim=-1).unsqueeze(-1).expand(-1,self.win_size))

            cri = loss
            cri = cri.detach().cpu().numpy()
            # print('cri',cri.shape)
            # print('labels',labels.shape)
            attens_energy.append(cri)
            test_labels.append(labels)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        test_labels = np.array(test_labels)

        # np.save('inputs_'+self.dataset,inputs)
        # np.save('outputs_'+self.dataset,outputs)

        gt = test_labels.astype(int)

        # print("pred:   ", pred.shape)
        print("gt:     ", gt.shape)
        
        np.save('test_energy'+self.dataset,test_energy)
        # np.save('predb'+self.dataset,pred)
        np.save('gtb'+self.dataset,gt)

        score = test_energy
        label = gt
        bf_eval = bf_search(score, label, start=0.001, end=1, step_num=150, verbose=False)
        pprint(bf_eval)