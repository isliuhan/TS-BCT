# -*- coding: utf-8 -*-

import datetime
from optim.pretrain import *
import queue
import argparse
import torch
from utils.utils import get_config_from_json
from optim.train import supervised_train


from optim.train import supervised_train_OOD
from optim.train import supervised_train_con



'''
'MiddlePhalanxOutlineAgeGroup', 
'ProximalPhalanxOutlineAgeGroup', 
'SwedishLeaf', 
'MixedShapesRegularTrain', 
'Crop'
'''
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--save_freq', type=int, default=200,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='')
    parser.add_argument('--K', type=int, default=4, help='Number of augmentation for each sample')
    parser.add_argument('--alpha', type=float, default=0.3, help='Past future split point')
    parser.add_argument('--feature_size', type=int, default=64,
                        help='feature_size')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of training epochs')
    parser.add_argument('--pretrained_epoch', type=int, default=2,
                        help='number of pretraining epochs')
    parser.add_argument('--patience', type=int, default=200,
                        help='training patience')

    parser.add_argument('--aug_type', type=str, default='none', help='Augmentation type')
    parser.add_argument('--w_aug_type', type=str, default='none', help='Augmentation type')
    parser.add_argument('--s_aug_type', type=str, default='none', help='Augmentation type')


    parser.add_argument('--piece_size', type=float, default=0.2,
                        help='piece size for time series piece sampling')
    parser.add_argument('--stride', type=float, default=0.2,
                        help='stride for forecast model')
    parser.add_argument('--horizon', type=float, default=0.1,
                        help='horizon for forecast model')
    parser.add_argument('--class_type', type=str, default='3C', help='Classification type')
    parser.add_argument('--gpu', type=str, default='2', help='gpu id')
    parser.add_argument('--ema_m', type=float, default=0.999, help='ema momentum for eval_model')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='learning rate')
    # model dataset
    parser.add_argument('--dataset_name', type=str, default='CricketX',
                        choices=['UCR--DATASETS',],
                        help='dataset')
    parser.add_argument('--nb_class', type=int, default=3,
                        help='class number')
    # Hyper-parameters for ts2vec model
    parser.add_argument('--irregular', type=float, default=0, help='The ratio of missing observations (defaults to 0)')
    # Hyper-parameters for vat model
    parser.add_argument('--n_power', type=int, default=4, metavar='N',
                        help='the iteration number of power iteration method in VAT')
    parser.add_argument('--xi', type=float, default=3, metavar='W', help='xi for VAT')
    parser.add_argument('--eps', type=float, default=1.0, metavar='W', help='epsilon for VAT')
    parser.add_argument('--ucr_path', type=str, default='./datasets',
                        help='Data root for dataset.')
    parser.add_argument('--device', default='cuda', type=str,
                        help='cpu or cuda')
    parser.add_argument('--expname', type=str, default='Ablation_exp', help='name of the experiment')
    parser.add_argument('--ckpt_dir', type=str, default='./ckpt/',
                        help='Data path for checkpoint.')
    parser.add_argument('--threshold', default=0.95, type=float,
                        help='pseudo label threshold')
    parser.add_argument('--T', default=0.5, type=float,
                        help='pseudo label temperature')
    parser.add_argument('--beta', default=0.5, type=float,
                        help='parameters for supmatch')
    parser.add_argument('--p_cutoff', type=float, default=0.95)
    parser.add_argument('--amp', type=str2bool, default=True, help='use mixed precision training or not')

    parser.add_argument('--weight_rampup', default=30, type=int, metavar='EPOCHS',
                        help='the length of rampup weight (default: 30)')
    parser.add_argument('--usp_weight', default=1.0, type=float, metavar='W',
                        help='the upper of unsuperivsed weight (default: 1.0)')
    # method
    parser.add_argument('--backbone', type=str, default='SimConv4')
    parser.add_argument('--model_name', type=str, default='supmatch',
                        choices=['SupCE','SemiTime','SemiTimev3','MTL','SemiTimev2','supmatch','contrastive',
                                 'Pi','Fixmatch','Pseudo','SupCE_con','Flexmatch','TFC'
                                 'SupCE_OOD','SemiTime_OOD','MTL_OOD','ts_tcc'
                                 'Pi_OOD','Fixmatch_OOD','Pseudo_OOD',
                                ],
                        help='choose  ')

    parser.add_argument('--config_dir', type=str, default='./config', help='The Configuration Dir')
    parser.add_argument('--label_ratio', type=float, default=0.1,
                        help='label ratio')
    #hyperparameter  for OOD classification
    parser.add_argument('--tot_class', type=int, default=7,
                        help='class number in distribution')
    #'ECG5000'              in distribution labels[1,2,3] out distribution labels [0,4]
    #'EpilepticSeizure',    in distribution labels[3,4],  out distribution labels [0,1,2]
    # because the classes is imbalance  tot_class is default

    #'UWaveGestureLibraryAll' and 'Mallat' tot_class=4
    #'XJTU'                 tot_class=7
    #'ElectricDevices'      tot_class=3
    parser.add_argument('--Mismatch_ratio', type=float, default=0.0,
                        help='class mismatch ratio')
    opt = parser.parse_args()
    return opt


def runexp(opt):
    import os
    import numpy as np
    import warnings
    warnings.filterwarnings("ignore")
    opt.ucr_path = './datasets'

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    exp = opt.expname


    Seeds =[0,1,2,3,4]
    # Seeds = [4]
    Runs = range(0, 5, 1)




    config_dict = get_config_from_json('{}/{}_config.json'.format(
        opt.config_dir, opt.dataset_name))       #从json获取配置

    opt.class_type = config_dict['class_type']
    opt.piece_size = config_dict['piece_size']

    if opt.model_name == 'SemiPF':
        model_paras='label{}_{}'.format(opt.label_ratio, opt.alpha)
    else:
        model_paras='label{}_{}'.format(opt.label_ratio,opt.Mismatch_ratio)
    aug1 = ['magnitude_warp']
    aug2 = ['time_warp']
    if aug1 == aug2:
        opt.aug_type = [aug1]
    elif type(aug1) is list:
        opt.aug_type = aug1 + aug2
    else:
        opt.aug_type = [aug1, aug2]

    log_dir = './results/{}/{}/{}/{}'.format(
        exp, opt.dataset_name, opt.model_name, model_paras)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    file2print_detail_train = open("{}/train_detail.log".format(log_dir), 'a+')
    print(datetime.datetime.now(), file=file2print_detail_train)
    print("Dataset\tTrain\tTest\tDimension\tClass\tSeed\tAcc_label\tAcc_unlabel\tEpoch_max\tw_aug_type\ts_aug_type"
          "\tbeta"
          , file=file2print_detail_train)
    file2print_detail_train.flush()

    file2print = open("{}/test.log".format(log_dir), 'a+')
    print(datetime.datetime.now(), file=file2print)
    print("Dataset\tAcc_mean\tAcc_std\tEpoch_max\tw_aug_type\ts_aug_type\tbeta",
          file=file2print)
    file2print.flush()

    file2print_detail = open("{}/test_detail.log".format(log_dir), 'a+')
    print(datetime.datetime.now(), file=file2print_detail)
    print("Dataset\tTrain\tTest\tDimension\tClass\tSeed\tAcc_max\tEpoch_max\tw_aug_type\ts_aug_type\tbeta",
          file=file2print_detail)
    file2print_detail.flush()

    ACCs = {}

    MAX_EPOCHs_seed = {}
    ACCs_seed = {}
    for seed in Seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        opt.ckpt_dir = './ckpt/{}/{}/{}/{}/{}/{}'.format(
            exp, opt.model_name, opt.dataset_name, '_'.join(opt.aug_type),
            model_paras, str(seed))

        if not os.path.exists(opt.ckpt_dir):
            os.makedirs(opt.ckpt_dir)

        print('[INFO] Running at:', opt.dataset_name)

        x_train, y_train, x_val, y_val, x_test, y_test, opt.nb_class, _ \
            = load_ucr2018(opt.ucr_path, opt.dataset_name)


        ACCs_run={}
        MAX_EPOCHs_run = {}
        for run in Runs:

            ################
            ## Train #######
            ################
            if opt.model_name == 'SupCE':
                acc_test, epoch_max = supervised_train(
                    x_train, y_train, x_val, y_val, x_test, y_test, opt)
                acc_unlabel=0


            elif 'SemiTime' == opt.model_name:
                acc_test, acc_unlabel, epoch_max = train_SemiTime(
                    x_train, y_train, x_val, y_val, x_test, y_test, opt)
            elif 'SemiTimev2' == opt.model_name:
                acc_test, acc_unlabel, epoch_max = train_SemiTimev2(
                    x_train, y_train, x_val, y_val, x_test, y_test, opt)
            elif 'SemiTimev3' == opt.model_name:
                acc_test, acc_unlabel, epoch_max = train_SemiTimev3(
                    x_train, y_train, x_val, y_val, x_test, y_test, opt)

            elif 'SemiTimev4' == opt.model_name:
                acc_test, acc_unlabel, epoch_max = train_SemiTimev4(
                    x_train, y_train, x_val, y_val, x_test, y_test, opt)

            elif 'SemiTimev5' == opt.model_name:
                acc_test, acc_unlabel, epoch_max = train_SemiTimev5(
                    x_train, y_train, x_val, y_val, x_test, y_test, opt)

            elif 'SemiTimev6' == opt.model_name:
                acc_test, acc_unlabel, epoch_max = train_SemiTimev6(
                    x_train, y_train, x_val, y_val, x_test, y_test, opt)
            elif 'time_bilinear' == opt.model_name:
                acc_test, acc_unlabel, epoch_max = train_time_bilinear(
                    x_train, y_train, x_val, y_val, x_test, y_test, opt)

            elif 'MTL' == opt.model_name:
                acc_test, acc_unlabel, epoch_max = train_Forecasting(
                    x_train, y_train, x_val, y_val, x_test, y_test,opt)
            elif 'SSTSC' == opt.model_name:
                acc_test, acc_unlabel, epoch_max=train_SSTSC(

                    x_train, y_train, x_val, y_val, x_test, y_test, opt)
            elif 'Fixmatch' == opt.model_name:
                acc_test, acc_unlabel,acc_ws, epoch_max=train_Fixmatch(
                    x_train, y_train, x_val, y_val, x_test, y_test, opt)

            elif 'Flexmatch' == opt.model_name:
                acc_test, acc_unlabel,acc_ws, epoch_max=train_Flexmatch(
                    x_train, y_train, x_val, y_val, x_test, y_test, opt)


            elif 'ts_tcc' == opt.model_name:
                acc_test, acc_unlabel, acc_ws, epoch_max =train_TS_TCC(
                    x_train, y_train, x_val, y_val, x_test, y_test, opt)

            elif 'TFC' in opt.model_name:
                acc_test, acc_unlabel, epoch_max = train_TFC(
                    x_train, y_train, x_val, y_val, x_test, y_test, opt)

            elif 'Fixmatch_CCA' == opt.model_name:
                acc_test, acc_unlabel,acc_ws, epoch_max=train_Fixmatch_CCA(
                    x_train, y_train, x_val, y_val, x_test, y_test, opt)

            elif 'supmatch' == opt.model_name:
                acc_test, acc_unlabel,acc_ws, epoch_max=train_supmatch(
                    x_train, y_train, x_val, y_val, x_test, y_test, opt)

            elif 'contrastive' == opt.model_name:
                acc_test, acc_unlabel,acc_ws, epoch_max=train_contrastive(
                    x_train, y_train, x_val, y_val, x_test, y_test, opt)
            elif 'wavelet_bilinear' == opt.model_name:
                acc_test, acc_unlabel,acc_ws, epoch_max=train_wavelet_bilinear(
                    x_train, y_train, x_val, y_val, x_test, y_test, opt)

            elif 'Fixmatch_bilinear' == opt.model_name:
                acc_test, acc_unlabel,acc_ws, epoch_max=train_Fixmatch_bilinear(
                    x_train, y_train, x_val, y_val, x_test, y_test, opt)
            elif 'Fixmatch_sup' == opt.model_name:
                acc_test, acc_unlabel,acc_ws, epoch_max=train_Fixmatch_sup(
                    x_train, y_train, x_val, y_val, x_test, y_test, opt)
            elif 'Fixmatch_sup_cca' == opt.model_name:
                acc_test, acc_unlabel,acc_ws, epoch_max=train_Fixmatch_sup_cca(
                    x_train, y_train, x_val, y_val, x_test, y_test, opt)

            elif 'Fixmatch_mix_cca' == opt.model_name:
                acc_test, acc_unlabel,acc_ws, epoch_max=train_Fixmatch_mix_cca(
                    x_train, y_train, x_val, y_val, x_test, y_test, opt)

            elif 'sup_con' == opt.model_name:
                acc_test, acc_unlabel,acc_ws, epoch_max=train_sup_con(
                    x_train, y_train, x_val, y_val, x_test, y_test, opt)

            elif 'Fixmatch_sup_bilinear' == opt.model_name:
                acc_test, acc_unlabel,acc_ws, epoch_max=train_Fixmatch_sup_bilinear(
                    x_train, y_train, x_val, y_val, x_test, y_test, opt)


            elif 'Fixmatch_sup_fft' == opt.model_name:
                acc_test, acc_unlabel,acc_ws, epoch_max=train_Fixmatch_sup_fft(
                    x_train, y_train, x_val, y_val, x_test, y_test, opt)
            elif 'Fixmatch_con_sup' == opt.model_name:
                acc_test, acc_unlabel,acc_ws, epoch_max=train_Fixmatch_con_sup(
                    x_train, y_train, x_val, y_val, x_test, y_test, opt)
            elif 'Pseudo' == opt.model_name:
                acc_test, acc_unlabel, acc_ws, epoch_max = train_pseudo(
                    x_train, y_train, x_val, y_val, x_test, y_test, opt)
            elif 'Pi' == opt.model_name:
                acc_test, acc_unlabel, acc_ws, epoch_max = train_pi(
                    x_train, y_train, x_val, y_val, x_test, y_test, opt)

            elif 'SupCE_con' == opt.model_name:
                acc_test, epoch_max = supervised_train_con(
                    x_train, y_train, x_val, y_val, x_test, y_test, opt)
                acc_unlabel = 0
            elif 'SupCE_OOD' == opt.model_name:
                acc_test, epoch_max = supervised_train_OOD(
                    x_train, y_train, x_val, y_val, x_test, y_test, opt)
                acc_unlabel=0
            elif 'SemiTime_OOD' == opt.model_name:
                acc_test, acc_unlabel, epoch_max = train_SemiTime_OOD(
                    x_train, y_train, x_val, y_val, x_test, y_test, opt)
            elif 'MTL_OOD' == opt.model_name:
                acc_test, acc_unlabel, epoch_max = train_Forecasting_OOD(
                    x_train, y_train, x_val, y_val, x_test, y_test,opt)
            elif 'Fixmatch_OOD' == opt.model_name:
                acc_test, acc_unlabel,acc_ws, epoch_max=train_Fixmatch_OOD(
                    x_train, y_train, x_val, y_val, x_test, y_test, opt)
            elif 'Pseudo_OOD' == opt.model_name:
                acc_test, acc_unlabel, acc_ws, epoch_max = train_pseudo_OOD(
                    x_train, y_train, x_val, y_val, x_test, y_test, opt)
            elif 'Pi_OOD' == opt.model_name:
                acc_test, acc_unlabel, acc_ws, epoch_max = train_pi_OOD(
                    x_train, y_train, x_val, y_val, x_test, y_test, opt)

            print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
                opt.dataset_name, x_train.shape[0], x_test.shape[0], x_train.shape[1], opt.nb_class,
                seed, round(acc_test, 2), round(acc_unlabel, 2), epoch_max,opt.beta),
                file=file2print_detail_train)
            file2print_detail_train.flush()
            ACCs_run[run] = acc_test
            MAX_EPOCHs_run[run] = epoch_max

        ACCs_seed[seed] = round(np.mean(list(ACCs_run.values())), 2)
        MAX_EPOCHs_seed[seed] = np.max(list(MAX_EPOCHs_run.values()))

        print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
            opt.dataset_name, x_train.shape[0], x_test.shape[0], x_train.shape[1], opt.nb_class,
            seed, ACCs_seed[seed], MAX_EPOCHs_seed[seed],opt.w_aug_type,opt.s_aug_type,opt.beta),
            file=file2print_detail)
        file2print_detail.flush()

    ACCs_seed_mean = round(np.mean(list(ACCs_seed.values())), 2)
    ACCs_seed_std = round(np.std(list(ACCs_seed.values())), 2)
    MAX_EPOCHs_seed_max = np.max(list(MAX_EPOCHs_seed.values()))

    print("{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
        opt.dataset_name, ACCs_seed_mean, ACCs_seed_std, MAX_EPOCHs_seed_max,opt.w_aug_type,opt.s_aug_type,opt.beta),
        file=file2print)
    file2print.flush()

if __name__ == "__main__":

    opt = parse_option()
    os.environ['CUDA_VISIBLE_DEVICES']=opt.gpu

    exp = 'exp-cls'
    from utils.expContent import expsEnd1 as exp
    q = queue.Queue()
    for e in exp:
        ds = e['ds']
        wt =e['w_transform']
        st =e['s_transform']
        # Mismatch_ratio=e['Mismatch_ratio']
        for data in ds:
            for w_trans in wt:
                for s_trans in st:
                # for ms in Mismatch_ratio:
                    paras = dict()
                    paras['data'] = data
                    paras['w_trans']=wt
                    paras['s_trans']=st
                    # paras['Mismatch_ratio']=ms
                    paras['exp_name'] = e['exp_name']
                    q.put(paras)

    while not q.empty():
        paras = q.get()
        data = paras['data']
        # ms = paras['Mismatch_ratio']
        w_trans=paras['w_trans']
        s_trans=paras['s_trans']
        opt.dataset_name = data
        opt.w_aug_type=w_trans
        opt.s_aug_type=s_trans
        # opt.Mismatch_ratio=ms
        opt.expname = paras['exp_name']
        runexp(opt)




