import torch
import numpy as np
from collections import OrderedDict
from net_stdf import MFVQE
import utils
from tqdm import tqdm

import os
from PIL import Image
import cv2

ckp_path = 'exp/MFQEv2_R3/ckp_290000.pt'  # trained at QP37, LDP, HM16.5

raw_yuv_path = '/home/pc-darwin/Garibaldi/stdf-pytorch/data/MFQEv2/test_18/raw/KristenAndSara_1280x720_600.yuv'
lq_yuv_path = '/home/pc-darwin/Garibaldi/stdf-pytorch/data/MFQEv2/test_18/QP37/KristenAndSara_1280x720_600.yuv'

#BasketballPass_416x240_500.yuv
#FourPeople_1280x720_600.yuv
#KristenAndSara_1280x720_600.yuv


h, w, nfs = 720, 1280, 300

torch.cuda.empty_cache()

video_name = os.path.splitext(os.path.basename(raw_yuv_path))[0]


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

def yuv420_to_rgb(Y, U, V):
    """
    Converte uma imagem YUV420p para RGB.

    :param Y: Matriz do canal Y com shape (H, W).
    :param U: Matriz do canal U com shape (H/2, W/2).
    :param V: Matriz do canal V com shape (H/2, W/2).
    :return: Imagem RGB.
    """
    H, W = Y.shape

    # Redimensiona U e V para o tamanho de Y
    U_resized = cv2.resize(U, (W // 2, H // 2), interpolation=cv2.INTER_LINEAR)
    V_resized = cv2.resize(V, (W // 2, H // 2), interpolation=cv2.INTER_LINEAR)

    # Converte para uint8 antes de empilhar
    Y = Y.astype(np.uint8)
    U_resized = U_resized.astype(np.uint8)
    V_resized = V_resized.astype(np.uint8)

    # Criar um único array YUV no formato correto
    yuv420p = np.concatenate((Y.flatten(), U_resized.flatten(), V_resized.flatten())).astype(np.uint8)

    # Reformata para o shape esperado pelo OpenCV (H * 1.5, W)
    yuv420p = yuv420p.reshape((H + H // 2, W))

    # Converte de YUV para RGB usando OpenCV
    rgb = cv2.cvtColor(yuv420p, cv2.COLOR_YUV2RGB_I420)

    return rgb

def save_rgb_image(Y, U, V, output_path):
    """
    Converte uma imagem YUV420p para RGB e a salva.

    :param Y: Matriz do canal Y com shape (H, W).
    :param U: Matriz do canal U com shape (H/2, W/2).
    :param V: Matriz do canal V com shape (H/2, W/2).
    :param output_path: Caminho onde a imagem será salva.
    """
    rgb_image = yuv420_to_rgb(Y, U, V)
    cv2.imwrite(output_path, cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))



def load_yuv420p(yuv_path, width, height, num_frames):
    """
    Carrega um arquivo YUV 420p e separa os canais Y, U e V em arrays NumPy distintos.

    :param yuv_path: Caminho do arquivo YUV.
    :param width: Largura do vídeo.
    :param height: Altura do vídeo.
    :param num_frames: Número total de quadros no arquivo.
    :return: Tupla (Y_frames, U_frames, V_frames), onde cada elemento é uma lista de arrays NumPy.
    """
    y_size = width * height
    uv_size = (width // 2) * (height // 2)  # 1/4 do tamanho de Y devido ao subsampling 4:2:0

    Y_frames = []
    U_frames = []
    V_frames = []

    with open(yuv_path, 'rb') as f:
        for _ in range(num_frames):
            # Ler canal Y
            Y = np.frombuffer(f.read(y_size), dtype=np.uint8).reshape((height, width))
            # Ler canal U
            U = np.frombuffer(f.read(uv_size), dtype=np.uint8).reshape((height // 2, width // 2))
            # Ler canal V
            V = np.frombuffer(f.read(uv_size), dtype=np.uint8).reshape((height // 2, width // 2))

            Y_frames.append(Y)
            U_frames.append(U)
            V_frames.append(V)

    return np.array(Y_frames), np.array(U_frames), np.array(V_frames)



def save_grayscale_frame(enhanced_frm, output_dir='saida_quadros', frame_name='frame'):
    """
    Salva um tensor 2D como imagem em escala de cinza (.png).
    
    Parâmetros:
    - enhanced_frm: Tensor 2D no formato (H, W).
    - output_dir: Diretório onde a imagem será salva.
    - frame_name: Nome do arquivo para a imagem de saída.
    """
    # Garantir que a pasta de saída exista
    os.makedirs(output_dir, exist_ok=True)
    
    # A entrada enhanced_frm já está no formato (H, W), então podemos simplesmente desanexá-lo e converter
    grayscale_frame = enhanced_frm.detach().cpu().numpy()  # Converte o tensor para um array numpy
    
    # Denormaliza a imagem, se necessário
    grayscale_frame = (grayscale_frame * 255).clip(0, 255)  # Multiplica por 255 e clipa para garantir que esteja no intervalo correto
    
    # Cria o caminho para salvar a imagem com o nome dado
    output_path = os.path.join(output_dir, f"{frame_name}.png")
    
    # Convertendo para imagem em escala de cinza e salvando
    img = Image.fromarray(grayscale_frame.astype('uint8'))  # Converte para uint8 para ser compatível com a imagem
    img.save(output_path)
    print(f"Quadro salvo em: {output_path}")
    return grayscale_frame

def main():
    # ==========
    # Load pre-trained model
    # ==========
    opts_dict = {
        'radius': 3, 
        'stdf': {
            'in_nc': 1, 
            'out_nc': 64, 
            'nf': 32, 
            'nb': 3, 
            'base_ks': 3, 
            'deform_ks': 3, 
            },
        'qenet': {
            'in_nc': 64, 
            'out_nc': 1, 
            'nf': 48, 
            'nb': 8, 
            'base_ks': 3, 
            },
        }   
    model = MFVQE(opts_dict=opts_dict)
    msg = f'loading model {ckp_path}...'
    print(msg)
    checkpoint = torch.load(ckp_path)
    if 'module.' in list(checkpoint['state_dict'].keys())[0]:  # multi-gpu training
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            name = k[7:]  # remove module
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:  # single-gpu training
        model.load_state_dict(checkpoint['state_dict'])

    msg = f'> model {ckp_path} loaded.'
    print(msg)
    model = model.cuda()
    model.eval()

    # ==========
    # Load entire video
    # ==========
    msg = f'loading raw and low-quality yuv...'
    print(msg)
    raw_y = utils.import_yuv(
        seq_path=raw_yuv_path, h=h, w=w, tot_frm=nfs, start_frm=0, only_y=False
        )
    lq_y = utils.import_yuv(
        seq_path=lq_yuv_path, h=h, w=w, tot_frm=nfs, start_frm=0, only_y=True
        )
    
    raw_y = raw_y[0].astype(np.float32) / 255.
    lq_y = lq_y.astype(np.float32) / 255.
    msg = '> yuv loaded.'
    print(msg)

    # ==========
    # Define criterion
    # ==========
    criterion = utils.PSNR()
    unit = 'dB'

    # ==========
    # Test
    # ==========
    pbar = tqdm(total=nfs, ncols=80)
    ori_psnr_counter = utils.Counter()
    enh_psnr_counter = utils.Counter()
    for idx in range(nfs):
        # load lq
        idx_list = list(range(idx-3,idx+4))
        idx_list = np.clip(idx_list, 0, nfs-1)
        input_data = []
        for idx_ in idx_list:
            input_data.append(lq_y[idx_])
        input_data = torch.from_numpy(np.array(input_data))
        input_data = torch.unsqueeze(input_data, 0).cuda()

        # enhance

        if h >= 720:
            enhanced_frm = process_in_patches(model, input_data)
        else:
            with torch.no_grad():
                enhanced_frm = model(input_data)


        #enhanced_frm = model(input_data)

        # eval
        gt_frm = torch.from_numpy(raw_y[idx]).cuda()
        batch_ori = criterion(input_data[0, 3, ...], gt_frm)
        batch_perf = criterion(enhanced_frm[0, 0, ...], gt_frm)
        

        if idx == 100 or idx == 50 or idx == 20:
            enframe = save_grayscale_frame(enhanced_frm[0, 0, ...],"saida_quadros",f"{video_name}_{idx}_enhanced")
            inpframe = save_grayscale_frame(input_data[0, 3, ...],"saida_quadros",f"{video_name}_{idx}_input")
            gtframe = save_grayscale_frame(gt_frm,"saida_quadros",f"{video_name}_{idx}_gt")

            Y, U, V = load_yuv420p(raw_yuv_path, w, h, nfs)

            U = U[idx]
            V = V[idx]

            save_rgb_image(enframe, U, V, f"saida_quadros/{video_name}_{idx}_color_enhanced.png")
            save_rgb_image(inpframe, U, V, f"saida_quadros/{video_name}_{idx}_color_input.png")
            save_rgb_image(gtframe, U, V, f"saida_quadros/{video_name}_{idx}_color_gt.png")


        ori_psnr_counter.accum(volume=batch_ori)
        enh_psnr_counter.accum(volume=batch_perf)

        # display
        pbar.set_description(
            "[{:.3f}] {:s} -> [{:.3f}] {:s}"
            .format(batch_ori, unit, batch_perf, unit)
            )
        pbar.update()
    
    pbar.close()
    ori_ = ori_psnr_counter.get_ave()
    enh_ = enh_psnr_counter.get_ave()
    print('ave ori [{:.3f}] {:s}, enh [{:.3f}] {:s}, delta [{:.3f}] {:s}'.format(
        ori_, unit, enh_, unit, (enh_ - ori_) , unit
        ))
    print('> done.')


if __name__ == '__main__':
    main()
