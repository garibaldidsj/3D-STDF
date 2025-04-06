import os
import time
import yaml
import argparse
import torch
import os.path as op
import numpy as np
from collections import OrderedDict
from tqdm import tqdm

import utils  # my tool box
import dataset
from net_stdf_temporalfusion import MFVQE

import torch.nn.functional as F

def ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, C1: float = 0.01**2, C2: float = 0.03**2):
    """
    Calcula o SSIM entre dois quadros em escala de cinza (1 canal) no PyTorch.

    Parâmetros:
    - img1 (torch.Tensor): Primeiro quadro (B, 1, H, W)
    - img2 (torch.Tensor): Segundo quadro (B, 1, H, W)
    - window_size (int): Tamanho da janela gaussiana (padrão: 11)
    - C1 (float): Constante para estabilidade (padrão: 0.01^2)
    - C2 (float): Constante para estabilidade (padrão: 0.03^2)

    Retorna:
    - SSIM médio (torch.Tensor): Valor médio de SSIM para cada batch.
    """
    def gaussian_window(window_size, sigma=1.5):
        kernel = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        kernel = torch.exp(-0.5 * (kernel / sigma).pow(2))
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, -1).repeat(1, 1, 1, 1)  # (1,1,window_size,1)
        return kernel * kernel.transpose(2, 3)  # Criar janela 2D (1,1,window_size,window_size)

    # Normalizar para faixa [0,1]
    img1 = img1.clamp(0, 1)
    img2 = img2.clamp(0, 1)

    # Criar janela gaussiana e mover para o mesmo dispositivo da imagem
    window = gaussian_window(window_size).to(img1.device)

    # Aplicar convolução para calcular médias locais
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=1)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=1)

    # Calcular variâncias e covariância
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=1) - mu1.pow(2)
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=1) - mu2.pow(2)
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=1) - mu1 * mu2

    # Fórmula do SSIM
    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1.pow(2) + mu2.pow(2) + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean()

def receive_arg():
    """Process all hyper-parameters and experiment settings.
    
    Record in opts_dict."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--opt_path', type=str, default='option.yml', 
        help='Path to option YAML file.'
        )
    args = parser.parse_args()
    
    with open(args.opt_path, 'r') as fp:
        opts_dict = yaml.load(fp, Loader=yaml.FullLoader)

    opts_dict['opt_path'] = args.opt_path

    if opts_dict['train']['exp_name'] == None:
        opts_dict['train']['exp_name'] = utils.get_timestr()

    opts_dict['train']['log_path'] = op.join(
        "exp", opts_dict['train']['exp_name'], "log_test.log"
        )
    opts_dict['train']['checkpoint_save_path_pre'] = op.join(
        "exp", opts_dict['train']['exp_name'], "ckp_"
        )
    opts_dict['test']['restore_iter'] = int(
        opts_dict['test']['restore_iter']
        )
    opts_dict['test']['checkpoint_save_path'] = (
        f"{opts_dict['train']['checkpoint_save_path_pre']}"
        f"{opts_dict['test']['restore_iter']}"
        '.pt'
        )

    return opts_dict


def main():
    # ==========
    # parameters
    # ==========

    opts_dict = receive_arg()
    unit = opts_dict['test']['criterion']['unit']

    # ==========
    # open logger
    # ==========

    log_fp = open(opts_dict['train']['log_path'], 'w')
    msg = (
        f"{'<' * 10} Test {'>' * 10}\n"
        f"Timestamp: [{utils.get_timestr()}]\n"
        f"\n{'<' * 10} Options {'>' * 10}\n"
        f"{utils.dict2str(opts_dict['test'])}"
        )
    print(msg)
    log_fp.write(msg + '\n')
    log_fp.flush()

    # ========== 
    # Ensure reproducibility or Speed up
    # ==========

    #torch.backends.cudnn.benchmark = False  # if reproduce
    #torch.backends.cudnn.deterministic = True  # if reproduce
    torch.backends.cudnn.benchmark = True  # speed up

    # ==========
    # create test data prefetchers
    # ==========
    
    # create datasets
    test_ds_type = opts_dict['dataset']['test']['type']
    radius = opts_dict['network']['radius']
    assert test_ds_type in dataset.__all__, \
        "Not implemented!"
    test_ds_cls = getattr(dataset, test_ds_type)
    test_ds = test_ds_cls(
        opts_dict=opts_dict['dataset']['test'], 
        radius=radius
        )

    test_num = len(test_ds)
    test_vid_num = test_ds.get_vid_num()

    # create datasamplers
    test_sampler = None  # no need to sample test data

    # create dataloaders
    test_loader = utils.create_dataloader(
        dataset=test_ds, 
        opts_dict=opts_dict, 
        sampler=test_sampler, 
        phase='val'
        )
    assert test_loader is not None

    # create dataloader prefetchers
    test_prefetcher = utils.CPUPrefetcher(test_loader)

    # ==========
    # create & load model
    # ==========

    model = MFVQE(opts_dict=opts_dict['network'])

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Número total de parâmetros treináveis: {num_params}')

    checkpoint_save_path = opts_dict['test']['checkpoint_save_path']
    msg = f'loading model {checkpoint_save_path}...'
    print(msg)
    log_fp.write(msg + '\n')

    checkpoint = torch.load(checkpoint_save_path)
    if 'module.' in list(checkpoint['state_dict'].keys())[0]:  # multi-gpu training
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            name = k[7:]  # remove module
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:  # single-gpu training
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    
    msg = f'> model {checkpoint_save_path} loaded.'
    print(msg)
    log_fp.write(msg + '\n')

    model = model.cuda()
    model.eval()

    # ==========
    # define criterion
    # ==========

    # define criterion
    assert opts_dict['test']['criterion'].pop('type') == \
        'PSNR', "Not implemented."
    criterion = utils.PSNR()

    # ==========
    # validation
    # ==========
                
    # create timer
    total_timer = utils.Timer()

    # create counters
    per_aver_dict = dict()
    ori_aver_dict = dict()
    per_aver_ssim = dict()
    ori_aver_ssim = dict()
    name_vid_dict = dict()
    for index_vid in range(test_vid_num):
        per_aver_dict[index_vid] = utils.Counter()
        ori_aver_dict[index_vid] = utils.Counter()
        per_aver_ssim[index_vid] = utils.Counter()
        ori_aver_ssim[index_vid] = utils.Counter()
        name_vid_dict[index_vid] = ""

    pbar = tqdm(
        total=test_num, 
        ncols=opts_dict['test']['pbar_len']
        )

    # fetch the first batch
    test_prefetcher.reset()
    val_data = test_prefetcher.next()

    with torch.no_grad():
        while val_data is not None:
            # get data
            gt_data = val_data['gt'].cuda()  # (B [RGB] H W)
            lq_data = val_data['lq'].cuda()  # (B T [RGB] H W)
            index_vid = val_data['index_vid'].item()
            name_vid = val_data['name_vid'][0]  # bs must be 1!
            
            b, _, c, h, w = lq_data.shape
            assert b == 1, "Not supported!"

            # Concatena e garante que o tensor seja contíguo
            input_data = torch.cat([lq_data[:, :, i, ...] for i in range(c)], dim=1).contiguous()

            def process_in_patches(model, input_data, patch_size=256):
                            """Processa a imagem em patches para evitar estouro de memória."""
                            _, channels, height, width = input_data.shape
                            output = torch.zeros((1, 1, height, width), device=input_data.device)
                            
                            for i in range(0, height, patch_size):
                                for j in range(0, width, patch_size):
                                    h_end = min(i + patch_size, height)
                                    w_end = min(j + patch_size, width)
                                    
                                    # Extrai o patch e torna-o contíguo
                                    patch = input_data[:, :, i:h_end, j:w_end].contiguous()
                                    with torch.no_grad():
                                        enhanced_patch = model(patch)
                                    output[:, :, i:h_end, j:w_end] = enhanced_patch
                                    
                            return output

                        # Se a altura for >= 1080, processa em patches; caso contrário, processa a imagem inteira.
            if h >= 1080:
                            enhanced_data = process_in_patches(model, input_data)
            else:
                            with torch.no_grad():
                                enhanced_data = model(input_data)

            # Avaliação
            torch.cuda.empty_cache()  # Limpa a memória da GPU
            batch_ori = criterion(lq_data[0, radius, ...], gt_data[0])
            batch_perf = criterion(enhanced_data[0], gt_data[0])

            ssim_ori = ssim(lq_data[0, radius, ...], gt_data[0])
            ssim_perf =ssim(enhanced_data[0], gt_data[0])

            ssim_ori = ssim_ori.item()
            ssim_perf = ssim_perf.item()

            # display
            pbar.set_description(
                "{:s}: PSNR: [{:.3f}] {:s} -> [{:.3f}] {:s} | SSIM: [{:.3f}] {:s} -> [{:.3f}] {:s}"
                .format(name_vid, batch_ori, unit, batch_perf, unit, ssim_ori, unit, ssim_perf, unit)
                )
            pbar.update()

            # log
            per_aver_dict[index_vid].accum(volume=batch_perf)
            ori_aver_dict[index_vid].accum(volume=batch_ori)

            per_aver_ssim[index_vid].accum(volume=ssim_perf)
            ori_aver_ssim[index_vid].accum(volume=ssim_ori)

            if name_vid_dict[index_vid] == "":
                name_vid_dict[index_vid] = name_vid
            else:
                assert name_vid_dict[index_vid] == name_vid, "Something wrong."

            # fetch next batch
            val_data = test_prefetcher.next()
        
    # end of val
    pbar.close()

    # log
    msg = '\n' + '<' * 10 + ' Results ' + '>' * 10
    print(msg)
    log_fp.write(msg + '\n')
    for index_vid in range(test_vid_num):
        per = per_aver_dict[index_vid].get_ave()
        ori = ori_aver_dict[index_vid].get_ave()

        per_ssim = per_aver_ssim[index_vid].get_ave()
        ori_ssim = ori_aver_ssim[index_vid].get_ave()

        name_vid = name_vid_dict[index_vid]
        msg = "{:s}: PSNR: [{:.3f}] {:s} -> [{:.3f}] {:s} SSIM: [{:.3f}] {:s} -> [{:.3f}] {:s}".format(
            name_vid, ori, unit, per, unit, ori_ssim, unit, per_ssim, unit
            )
        print(msg)
        log_fp.write(msg + '\n')
    ave_per = np.mean([
        per_aver_dict[index_vid].get_ave() for index_vid in range(test_vid_num)
        ])
    ave_ori = np.mean([
        ori_aver_dict[index_vid].get_ave() for index_vid in range(test_vid_num)
        ])
    ave_per_ssim = np.mean([
        per_aver_ssim[index_vid].get_ave() for index_vid in range(test_vid_num)
        ])
    ave_ori_ssim = np.mean([
        ori_aver_ssim[index_vid].get_ave() for index_vid in range(test_vid_num)
        ])
    msg = (
        f"{'> ori PSNR: [{:.3f}] {:s}'.format(ave_ori, unit)}\n"
        f"{'> ave PSNR: [{:.3f}] {:s}'.format(ave_per, unit)}\n"
        f"{'> delta PSNR: [{:.3f}] {:s}'.format(ave_per - ave_ori, unit)}\n"
        f"{'> ori SSIM: [{:.3f}] {:s}'.format(ave_ori_ssim, unit)}\n"
        f"{'> ave SSIM: [{:.3f}] {:s}'.format(ave_per_ssim, unit)}\n"
        f"{'> delta SSIM: [{:.3f}] {:s}'.format(ave_per_ssim - ave_ori_ssim, unit)}"
        )
    print(msg)
    log_fp.write(msg + '\n')
    log_fp.flush()

    # ==========
    # final log & close logger
    # ==========

    total_time = total_timer.get_interval() / 3600
    msg = "TOTAL TIME: [{:.1f}] h".format(total_time)
    print(msg)
    log_fp.write(msg + '\n')
    
    msg = (
        f"\n{'<' * 10} Goodbye {'>' * 10}\n"
        f"Timestamp: [{utils.get_timestr()}]"
        )
    print(msg)
    log_fp.write(msg + '\n')
    
    log_fp.close()


if __name__ == '__main__':
    main()
    