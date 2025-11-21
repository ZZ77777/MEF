import os
import json
import time
import datetime
import argparse

import numpy as np

import torch
import torch.optim
import torchvision
from torchvision import datasets, transforms

from optim.adam import Adam
from optim.adamax import Adamax
from models.nets import Model
from models.utils import preprocess, postprocess


"""
Funcionalidades:
    
    utils.py: 
        - expm, series: Cálculos de exponenciales de matrices
        - squeeze2d, unsqueeze2d: Transformaciones espaciales de tensores
        - preprocess, postprocess: imágenes
    
    flows.py: Capas de normalizing flows
        - ActNorm: Normalización con parámetros aprendibles
        - Conv1x1: Convoluciones invertibles 1x1
        - CouplingLayer: Capas de acoplamiento o transformaciones invertibles
    
    nets.py: Arquitectura del modelo
        - Model: Red neuronal de torch
    
    optim/:
        - adam.py, adamax.py: Variantes de optimizadores con promedios de Polyak
    
    main.py: 
        - Entrenamiento y generación de muestras
"""

"""
                                 utils:
                                    expm, series
     _______________________________|   squeeze2d, unsqueeze2d, split2d, unsplit2d
    |                                     |  preprocess, postprocess
    V                                     |      |
    flows:                                |      |
    ActNorm, Conv1x1, CouplingLayer       |      |                   
    |                                     |      |
    |   __________________________________|      |
    |  |                                         |
    V  V                                         |
    nets:    adam.py + adamax.py                 |
    Model           |                            |
    |_______________|____________________________|
    |
    V
    main
"""





def main(args):
    
    """
    Flujo de ejecución del modelo.
    
    Args:
        mode: train, test, sample
    """
    
    
    if args.mode == 'train':
        ### Configurar dispositivo (GPU o CPU)
        device = torch.device(args.device) 
        
        ### Directorio para resultados
        save_dir = get_save_dir(args)
        # file_name = os.path.join(save_dir, args.dataset + '.json')
        # with open(file_name, 'w') as f_obj:
        #    json.dump(args.__dict__, f_obj)
        
        ### Crear modelo de normalizing flow

        model = get_model(args)         
        
        # model = torch.nn.DataParallel(model)
        model.to(device) ### Mover a CPU/GPU
        
        ## Esas cuatro lineas se repiten en las tres condiciones
        ## ¿Por que no se han puesto antes del if?
        

        ### Configurar optimizador y scheduler de learning rate
        optimizer = get_optimizer(args, model)
        ### MultiStepLR reduce el learning rate en epochs específicos
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.step_size, args.lr_decay)
        ### Cargar datasets
        train_data, test_data = get_dataset(args)
        
        ### Crear DataLoaders para iterar sobre los datos en batches ??
        train_loader = torch.utils.data.DataLoader(
            train_data, 
            batch_size=args.batch_size,
            num_workers=args.workers, ### Procesos paralelos al cargar datos
            shuffle=True)
        
        test_loader = torch.utils.data.DataLoader(
            test_data, 
            batch_size=args.batch_size,
            num_workers=args.workers, 
            shuffle=False)
        
        
        ### Obtener batch inicial para inicializar parámetros del modelo
        init_data = get_init_data(args, train_data)
        # param_num = sum(p.numel() for p in model.parameters())
        # print('parameter number  ', param_num)
        
        ### Entrenamiento
        train(args, device, save_dir, model, optimizer, scheduler, train_loader, test_loader, init_data)




    elif args.mode == 'test':
        ### Evaluar el modelo
        
        ### Lo mismo para configurar el dispositivo y crear el modelo 
        device = torch.device(args.device)
        save_dir = get_save_dir(args)
        model = get_model(args)
        model.to(device)
        # model = torch.nn.DataParallel(model) 
        optimizer = get_optimizer(args, model)
        
        ### Carga solo datos de pruaba
        _, test_data = get_dataset(args)
        
        
        
        test_loader = torch.utils.data.DataLoader(
            test_data, 
            batch_size=args.batch_size,
            num_workers=args.workers, shuffle=False)
        
        model_dir = os.path.join(save_dir, 'models')
        
        file_name = 'epoch_{}.pth'.format(args.test_epoch)
        state_file = os.path.join(model_dir, file_name)
        if not os.path.isfile(state_file):
            raise RuntimeError('file {} is not found'.format(state_file))
        print('load checkpoint {}'.format(state_file))
        checkpoint = torch.load(state_file, device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        optimizer.swap()
        test(args, device, model, test_loader, args.test_epoch)
        optimizer.swap()
        
        
        
        
        

    elif args.mode == 'sample':
        ### Generación de muestras
        
        ### Lo mismo para configurar el dispositivo y crear el modelo 
        device = torch.device(args.device)
        save_dir = get_save_dir(args)
        model = get_model(args)
        model.to(device)
        
        model = torch.nn.DataParallel(model)
        optimizer = get_optimizer(args, model)
        model_dir = os.path.join(save_dir, 'models')
        file_name = 'epoch_{}.pth'.format(args.sample_epoch)
        state_file = os.path.join(model_dir, file_name)
        if not os.path.isfile(state_file):
            raise RuntimeError('file {} is not found'.format(state_file))
        print('load checkpoint {}'.format(state_file))
        checkpoint = torch.load(state_file, device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        optimizer.swap()
        sample(args, device, save_dir, model, args.sample_epoch)
        optimizer.swap()

    else:
        raise ValueError('wrong mode')
    
    
    
    
    
    
    
    
    
    
    
    
    
    


def train(args, device, save_dir, model, optimizer, scheduler, train_loader, test_loader, init_data):
    """
    Entrenamiento principal.
    
    Args:
        args: Configuración
        device: Dispositivo (CPU/GPU)
        save_dir: Directorio para guardar checkpoints
        model: Modelo de normalizing flow
        optimizer: Optimizador
        scheduler: Scheduler de learning rate
        train_loader: DataLoader de entrenamiento
        test_loader: DataLoader de prueba
        init_data: Batch inicial para inicialización
    """
    
    ### Para registrar métricas durante el entrenamiento
    train_log = {'train_loss': [], 'epoch_loss': [], 'epoch_time': [], 'test_loss': []}
    
    start_epoch = 1
    best_loss = 1e8 ### Mejor loss encontrado para guardar mejor modelo
    
    ### Warmup scheduler incrementa gradualmente el learning rate al inicio
    lr_lambda = lambda step: (step + 1) / (len(train_loader) * args.warmup_epoch)
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


    ### Reanudar entrenamiento desde checkpoint si se especifica  
    if args.resume_epoch is not None:
        state_file = os.path.join(save_dir, 'models', 'epoch_' + str(args.resume_epoch) + '.pth')
        if not os.path.isfile(state_file):
            raise RuntimeError('file {} is not found'.format(state_file))
        
        print('load checkpoint {}'.format(state_file))
        checkpoint = torch.load(state_file, device)
        start_epoch = checkpoint['epoch'] + 1
        train_log = checkpoint['train_log']
        model.module.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
    else:
        ### INICIALIZACIÓN DEL MODELO
        ### Los normalizing flows requieren inicialización data-dependent
        ### para que las activaciones tengan media 0 y varianza 1
        model.eval()
        with torch.no_grad():
            init_data = init_data.to(device)
            ### Añadir ruido uniforme para dequantización
            z = torch.rand_like(init_data)
            init_data = preprocess(init_data, args.bits, z)
            ### Forward pass de inicialización
            _, _ = model(init_data, init=True)

    print('start training') ### BUCLE PRINCIPAL DE ENTRENAMIENTO
    for epoch in range(start_epoch, args.epochs + 1):
        model.train() ### Modo entrenamiento para habilitar dropout, batch norm, etc.
        total_loss = 0.
        number = 0 ### Total de muestras procesadas
        t0 = time.time()

        ### Iteración sobre batches 
        for data, _ in train_loader: 
            data = data.to(device)
            
            ### DEQUANTIZACIÓN
            ### Añadir ruido uniforme a las imágenes discretas para hacerlas continuas
            ### Importante para modelar distribuciones continuas bien
            z = torch.rand_like(data)
            data = preprocess(data, args.bits, z)
            
            ### FORWARD PASS
            ### output: variables latentes de distribución normal
            ### log_det: log-determinante del Jacobiano (cambio de volumen)
            output, log_det = model(data)
            
            ### CALCULAR LOSS
            ### Loss = -log p(x) donde p(x) se calcula usando change of variables
            loss = compute_loss(args, output, log_det)
            
            ### Convertir a bits por dimensión (métrica estándar en normalizing flows)
            train_log['train_loss'].append(loss.item() / (np.log(2) * args.dimension))
            
            total_loss += loss.item() * data.size(0)
            number += data.size(0)
            
            ### BACKWARD PASS Y ACTUALIZACIÓN
            ### Limpiar gradientes previos, calcular nuevos y actualizar parámetros
            optimizer.zero_grad()
            loss.backward()         
            optimizer.step()        
            
            ### Aplicar warmup scheduler durante las primeras épocas
            if epoch <= args.warmup_epoch:
                warmup_scheduler.step()

        ### MÉTRICAS DE LA ÉPOCA
        bits_per_dim = total_loss / number / (np.log(2) * args.dimension)
        train_log['epoch_loss'].append((epoch, bits_per_dim))
        
        t1 = time.time()
        train_log['epoch_time'].append((epoch, t1 - t0))
        print('[train:epoch {}]. loss: {:.8f},time:{:.1f}s '.format(epoch, bits_per_dim, t1 - t0))
        
        ### Actualizar learning rate según el scheduler
        scheduler.step()
        
        ### EVALUACIÓN PERIÓDICA
        if not (epoch % args.test_interval):
            ### Evaluar con parámetros promediados con Polyak averaging
            optimizer.swap()
            test_loss = test(args, device, model, test_loader, epoch)
            optimizer.swap()
            
            ### Evaluar con parámetros actuales
            test_loss1 = test(args, device, model, test_loader, epoch)
            train_log['test_loss'].append((epoch, test_loss, test_loss1))
            
            ### Guardar mejor modelo
            if test_loss < best_loss:
                best_loss = test_loss
                save(save_dir, epoch, train_log, model, optimizer, scheduler, is_best=True)
                
        ### GUARDAR CHECKPOINT PERIÓDICAMENTE
        if not (epoch % args.save_interval):
            save(save_dir, epoch, train_log, model, optimizer, scheduler)
    return
















def test(args, device, model, test_loader, epoch):
    """
    Evaluar el modelo en el conjunto de prueba.
    
    Args:
        args: Configuración
        device: Dispositivo (CPU/GPU)
        model: Modelo a evaluar
        test_loader: DataLoader de prueba
        epoch: Número de época actual
    
    Returns:
        bits_per_dim: Loss promedio en bits por dimensión
    """
    total_loss = 0.
    number = 0
    
    model.eval() ### Modo evaluación (deshabilita dropout, etc.)
    
    with torch.no_grad(): ### No calcular gradientes para ahorrar tiempo y memoria
        for data, _ in test_loader:
            data = data.to(device)
            
            ### Dequantización (igual que en entrenamiento)
            z = torch.rand_like(data)
            data = preprocess(data, args.bits, z)
            
            ### Forward pass
            output, log_det = model(data)
            
            ### Loss
            loss = compute_loss(args, output, log_det)
            total_loss += loss.item() * data.size(0)
            number += data.size(0)
            
    ### Pasar a bits por dimensión
    bits_per_dim = total_loss / number / (np.log(2) * args.dimension)
    print('[test:epoch {}]. loss: {:.8f} '.format(epoch, bits_per_dim))
    return bits_per_dim














def sample(args, device, save_dir, model, epoch): 
    """
    Generar muestras con el modelo entrenado.
    
    Con normalizing flows se puede muestrear z ~ N(0,I)
    y aplicar la transformación inversa para obtener muestras x.
    
    Args:
        args: Configuración
        device: Dispositivo (CPU/GPU)
        save_dir: Directorio donde guardar las muestras
        model: Modelo entrenado
        epoch: Época del modelo
    """
    ### Muestrear del espacio latente de distribución normal estándar
    z = torch.randn(args.sample_size, 3, args.image_size, args.image_size).to(device)
    
    
    model.eval()
    with torch.no_grad():
        ### Aplicar transformación inversa: z -> x
        output, _ = model(z, reverse=True)
        
        ### Postprocesar para obtener imágenes válidas [0, 255]
        output = postprocess(output, args.bits)
        
    ### Crear directorio de muestras si no existe
    sample_dir = os.path.join(save_dir, 'samples')
    if not os.path.isdir(sample_dir):
        os.makedirs(sample_dir)
        
    ### Guardar grid de imágenes
    torchvision.utils.save_image(output, os.path.join(sample_dir, 'epoch_{}.png'.format(epoch)),
                                 nrow=int(args.sample_size ** 0.5), pad_value=1)
    return








def get_save_dir(args):
    """
    Crear y devolver directorio para guardar resultados.
    """
    if args.save_dir:
        save_dir = args.save_dir
    else:
        ### Crear nombre único con dataset y timestamp
        name = args.dataset + '_' + str(datetime.datetime.now())[:-7].replace(' ', '-').replace(':', '-')
        save_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'save', name)
    
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
        
    return save_dir







def get_model(args):
    """
    Crear instancia del modelo de flujo.
    """
    model = Model(args.levels,          ### Número de niveles jerárquicos en el modelo
                  args.num_flows,       ### Lista con número de flows por nivel
                  args.conv_type,       ### Tipo de convolución 1x1 invertible
                  args.flow_type,       ### Tipo de coupling layer
                  args.num_blocks,      ### Número de bloques en las redes de acoplamiento
                  args.hidden_channels, ### Canales ocultos en las redes
                  args.image_size)      ### Tamaño de las imágenes
    return model









def get_dataset(args):
    """
    Cargar y preparar datasets de entrenamiento y prueba.
    """
    if args.dataset == 'cifar10':
        ### CIFAR-10 (32x32 imágenes RGB): 50k train, 10k test
        train_data = datasets.CIFAR10(args.dataset_dir, train=True, download=True, #### Añadido download para prueba
                                      transform=transforms.Compose([
                                          transforms.RandomHorizontalFlip(), ### Data augmentation
                                          transforms.ToTensor() ### Convertir a tensor [0, 1]
                                      ]))
        test_data = datasets.CIFAR10(args.dataset_dir, train=False,  download=True, #### Añadido download para prueba
                                     transform=transforms.Compose([
                                         transforms.ToTensor()
                                     ]))
        assert args.image_size == 32
        assert args.dimension == 3072 ### 32 * 32 * 3
        
    elif args.dataset == 'imagenet32':
        ### ImageNet 32x32: versión downsampled de ImageNet
        
        train_data = datasets.ImageFolder(os.path.join(args.dataset_dir, 'train_32x32'),
                                          transform=transforms.Compose([
                                              transforms.ToTensor()
                                          ]))
        test_data = datasets.ImageFolder(os.path.join(args.dataset_dir, 'valid_32x32'),
                                         transform=transforms.Compose([
                                             transforms.ToTensor()
                                         ]))
        assert args.image_size == 32
        assert args.dimension == 3072
        
    elif args.dataset == 'imagenet64':
        ### ImageNet 64x64: versión downsampled de ImageNet
        train_data = datasets.ImageFolder(os.path.join(args.dataset_dir, 'train_64x64'),
                                          transform=transforms.Compose([
                                              transforms.ToTensor()
                                          ]))
        test_data = datasets.ImageFolder(os.path.join(args.dataset_dir, 'valid_64x64'),
                                         transform=transforms.Compose([
                                             transforms.ToTensor()
                                         ]))
        assert args.image_size == 64
        assert args.dimension == 12288 ### 64 * 64 * 3
    else:
        raise ValueError('wrong dataset')
    return train_data, test_data









def get_init_data(args, train_data):
    """
    Obtener batch aleatorio para inicialización del modelo.
    Se necesita un forward pass inicial con datos reales para inicializar bien las capas de normalización.
    """
    train_index = np.arange(len(train_data))
    np.random.shuffle(train_index)
    
    ### Seleccionar muestras aleatorias
    init_index = np.random.choice(train_index, args.init_batch_size, replace=False)
    
    images = []
    for index in init_index:
        image, _ = train_data[index]
        images.append(image)
        
    return torch.stack(images, dim=0)








def get_optimizer(args, model):
    """
    Crear optimizador con Polyak averaging.
    
    Polyak averaging mantiene un promedio exponencial móvil de los parámetros
    para mejorar estabilidad y rendimiento
    """
    if args.optimizer == 'adam':
        optimizer = Adam(
            [{'params': model.parameters()}],
            lr=args.lr,
            weight_decay=args.weight_decay, ### Regularización L2 creo??
            polyak=args.polyak)  ### Factor para el promedio móvil
        

    elif args.optimizer == 'adamax':
        optimizer = Adamax(
            [{'params': model.parameters()}], 
            lr=args.lr,
            weight_decay=args.weight_decay, 
            polyak=args.polyak)
        
    else:
        raise ValueError('wrong optimizer')
    return optimizer









def compute_loss(args, output, log_det):
    """
    Calcular la negative log-likelihood del modelo.
    
    Según chatgpt
        La perdida en modelos de flujo se basa en el cambio de variables:
            log p(x) = log p(z) + log|det(dz/dx)|
        
        donde:
            - p(z) es la distribución latente (normal estándar)
            - log|det(dz/dx)| es el log-determinante del Jacobiano
    
    Args:
        args: Configuración (bits y dimension)
        output: Variables latentes z = f(x)
        log_det: Log-determinante del Jacobiano acumulado
    
    Returns:
        loss: Negative log-likelihood
    """
    ### Log-probabilidad de z bajo distribución normal estándar
    log_p = torch.distributions.Normal(torch.zeros_like(output), torch.ones_like(output)).log_prob(output).view(
        output.size(0), -1).sum(-1)
    
    ### Loss = -log p(x)
    ### El término np.log((2.0 ** args.bits) / 2.0) * args.dimension
    ### corresponde a la corrección por dequantización
    loss = -(log_p + log_det - np.log((2.0 ** args.bits) / 2.0) * args.dimension).mean()
    return loss









def save(save_dir, epoch, train_log, model, optimizer, scheduler, is_best=False):
    """
    Guardar checkpoint del modelo.
    
    Args:
        save_dir: Directorio de guardado
        epoch: Época actual
        train_log: Historial de entrenamiento
        model: Modelo a guardar
        optimizer: Optimizador (incluye parámetros de Polyak)
        scheduler: Scheduler de learning rate
        is_best: Si True, guarda como 'epoch_best.pth'
    """
    
    model_dir = os.path.join(save_dir, 'models')
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
        
        
    file_name = 'epoch_best.pth' if is_best else 'epoch_{}.pth'.format(epoch)
    state_path = os.path.join(model_dir, file_name)
    
    ### Guardar estado completo
    state = {
        'epoch': epoch,
        'train_log': train_log,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }
    torch.save(state, state_path)











### - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MEF')
    ### CONFIGURACIÓN DE ARGUMENTOS
    ## Siento que podría haber una otra forma de hacer esto
    
    ### Modo de operación
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test', 'sample'],
                        help='mode')
    
    ### Hardware
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='device')
    
    
    ### Dataset
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'imagenet32', 'imagenet64'],
                        help='dataset')
    parser.add_argument('--dataset_dir', type=str, default='/datasets',
                        help='dataset directory.')
    parser.add_argument('--save_dir', type=str, default='',
                        help='save directory')
    
    
    ### Configuración de imágenes
    parser.add_argument('--image_size', type=int, default=32,
                        choices=[32, 64],
                        help='image size')
    parser.add_argument('--dimension', type=int, default=3072,
                        choices=[3072, 12288],
                        help='image dimension')
    parser.add_argument('--bits', type=int, default=8,
                        help='number of bits per pixel.')
    
    
    ### Configuración de ejecución
    parser.add_argument('--workers', type=int, default=8,
                        help='number of data loader workers')
    parser.add_argument('--save_interval', type=int, default=1,
                        help='number of batches to save')
    parser.add_argument('--test_interval', type=int, default=1,
                        help='number of batches to test')
    
    ### Control de checkpoints
    parser.add_argument('--resume_epoch', type=int, default=None,
                        help='resume training epoch')
    parser.add_argument('--test_epoch', type=int, default=None,
                        help='test epoch')
    parser.add_argument('--sample_epoch', type=int, default=None,
                        help='sample epoch')
    parser.add_argument('--sample_size', type=int, default=64,
                        help='sample size')
    
    
    
    ### ARQUITECTURA DEL MODELO
    parser.add_argument('--levels', type=int, default=3,
                        help='number of flow levels')
    parser.add_argument('--num_flows', type=list, default=[8, 4, 2],
                        help='number of flows per level')
    parser.add_argument('--conv_type', type=str, default='matrixexp',
                        choices=['standard', 'decomposition', 'matrixexp'],
                        help='invertible 1x1 convolution')
    parser.add_argument('--flow_type', type=str, default='matrixexp',
                        choices=['additive', 'affine', 'matrixexp'],
                        help='flow type')
    parser.add_argument('--num_blocks', type=int, default=8,
                        help='number of blocks of coupling layers')
    parser.add_argument('--hidden_channels', type=int, default=128,
                        help='hidden channels of coupling layers')
    
    
    ### HIPERPARÁMETROS DE ENTRENAMIENTO
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs to train ')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--init_batch_size', type=int, default=512,
                        help='batch size for initialization')
    
    
    ### Optimización
    parser.add_argument('--optimizer', type=str, default='adamax',
                        choices=['adam', 'adamax'],
                        help='optimizer')
    parser.add_argument('--warmup_epoch', type=int, default=1,
                        help='warmup epoch')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--step_size', type=list, default=[40],
                        help='multi step learning rate decay')
    parser.add_argument('--lr_decay', type=float, default=0.5,
                        help='factor of learning rate decay')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='weight decay')
    parser.add_argument('--polyak', type=float, default=0.999,
                        help='polyak average')
    
    
    
    ### Reproducibilidad
    parser.add_argument('--seed', type=int, default=0,
                        help='seed')
    
    
    
    
    parse_args = parser.parse_args()

    """
    param_dir = ''
    if param_dir:
        with open(param_dir) as f_obj:
            parse_args.__dict__ = json.load(f_obj)
    """

    ### Establecer semillas 
    np.random.seed(parse_args.seed)
    torch.manual_seed(parse_args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(parse_args.seed)
        
        
        
        
    main(parse_args) ### Ejecutar programa principal
