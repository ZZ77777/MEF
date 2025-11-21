import scipy
from scipy.linalg import logm
from scipy.stats import ortho_group, special_ortho_group

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils import expm, series

eps = 1e-8
### Epsilon para evitar divisiones por cero
### Primera vez que lo veo


"""
Respaso de modelos de flujo:

consisten en transformaciones que se pueden revertir 
que mapean datos complejos a distribuciones simples 
normalmente gaussianas. 

Cada transformación debe ser:
 invertible: Podemos recuperar x desde z sin perder información
 diferenciable: Podemos calcular gradientes
    la transformación tiene derivadas y podemos calcularlas suavemente.
    lo cual es necesario para hacer backpropagation
 con jacobian eficiente: Podemos calcular log|det(df/dx)|
    convirtiendo datos x en un latente z con una transformación invertible f,
    la probabilidad cambia según el determinante del jacobiano
    Este determinante describe cómo se estira o encoge
    el espacio al aplicar la transformación.
    Pero para una función general, el determinante del jacobiano 
    es caro de calcular y se hace millones de veces en el entrenamiento.
    Para hacer el determinante facil de calcular
    las transformaciones están diseñadas para que su jacobiano 
    tenga una forma especial (triangular, diagonal, etc.)



Aqui se implementan tres tipos de capas fundamentales:
- ActNorm/Norm: Normalización adaptativa
- Conv1x1: Mezcla de canales invertible
- CouplingLayer: Transformaciones complejas mediante acoplamiento
"""















class ActNorm(nn.Module):
    """
    Actnorm layers: y = softplus(scale) * x + shift which are data dependent, initialized
    such that the distribution of activations has zero mean and unit variance
    given an initial mini-batch of data.
    Normalize all activations independently instead of normalizing per channel.


    ActNorm tiene parámetros APRENDIBLES (scale y shift), cosa que BatchNorm no,
    Normaliza cada activación independientemente y no por canal
    
    Inicialización: Con las estadísticas del primer batch, 
    ajusta scale y shift para que las salidas tengan media 0 y varianza 1.
    
    La normalización ayuda a stabilizar el entrenamiento,
    el modelo sea menos sensible a la escala de los datos
    y hace fácil calcular el log-determinante
    """

    def __init__(self, in_channels, image_size):

        super(ActNorm, self).__init__()
        self.in_channels = in_channels
        self.image_size = image_size
        self.scale = nn.Parameter(torch.ones(in_channels, image_size, image_size))
        self.shift = nn.Parameter(torch.zeros(in_channels, image_size, image_size))

    def forward(self, x, reverse=False, init=False, init_scale=1.0):
        """
        Forward pass o inverso si reverse=True
        
        Args:
            x: Tensor de entrada [batch, channels, height, width]
            reverse: Si True, aplica transformación inversa
            init: Si True, inicializa parámetros usando estadísticas de x
            init_scale: Factor de escala para la inicialización
            
        Returns:
            x: Tensor transformado
            log_det: Log-determinante del Jacobiano (escalar)
        """
        if init:
            ### Inicialización dependiente de datos
            ### Media y desviación estándar del batch
            mean = x.mean(dim=0)
            std = x.std(dim=0)

            ### Invertir std para normalizar
            inv_std = init_scale / (std + eps)

            ### Configurar scale: 
            ### se necesita la inversa de softplus invera porque 
            ### forward usa softplus y softplus(x) = log(1 + exp(x))
            self.scale.data.copy_(torch.log(-1 + torch.exp(inv_std)))
            self.shift.data.copy_(-mean * inv_std) ### Configurar shift para centrar en cero

        if not reverse:
            ### FORWARD: y = scale * x + shift
            scale = F.softplus(self.scale) ### Asegurar que scale > 0
            x = scale * x + self.shift

            ### Log-determinante: sum(log(scale))
            ### Para transformación afín y = ax + b, |det(J)| = a^n donde n = dim
            log_det = torch.log(scale).sum()
        else:
            ### INVERSE: x = (y - shift) / scale
            scale = F.softplus(self.scale)
            x = (x - self.shift) / scale
            ### Log-determinante inverso tiene signo opuesto
            log_det = torch.log(scale).sum().mul(-1)
        return x, log_det

    def extra_repr(self):
        """Representación para print(model)"""
        return 'in_channels={}, image_size={}'.format(self.in_channels, self.image_size)














class Norm(nn.Module):
    """
    Norm layers: y = scale * x +shift which are data dependent, initialized
    such that the distribution of activations per channel has zero mean and unit variance
    given an initial mini-batch of data.


    Normalización por canal
    
    Similar a ActNorm pero normaliza POR CANAL en lugar de por cada activación.
    Es decir, cada canal tiene un único scale y shift que se aplica a todas
    las posiciones espaciales.
    
    Transformación: y = scale * x + shift
    donde scale y shift tienen forma [channels, 1, 1]
    
    Se usa en las redes internas de los coupling layers.
    """

    def __init__(self, in_channels):
        """
        Args:
            in_channels: Número de canales
        """
        super(Norm, self).__init__()
        self.in_channels = in_channels

        ### Un parámetro por canal (broadcasting a dimensiones espaciales)
        self.scale = nn.Parameter(torch.ones(in_channels, 1, 1))
        self.shift = nn.Parameter(torch.zeros(in_channels, 1, 1))

    def forward(self, x, init=False, init_scale=1.0):
        """
        Args:
            x: Tensor [batch, channels, height, width]
            init: Si True, inicializa usando estadísticas de x
            init_scale: Factor de escala inicial
            
        Returns:
            x: Tensor normalizado
        """
        if init:
            ### INICIALIZACIÓN DATA-DEPENDENT
            ### Reorganizar para calcular estadísticas por canal
            out = x.transpose(0, 1).contiguous().view(self.in_channels, -1)
            ### out tiene forma [channels, batch*height*width]

            ### Media y std por canal
            mean = out.mean(dim=1).view(self.in_channels, 1, 1)
            std = out.std(dim=1).view(self.in_channels, 1, 1)

            ### parámetros para normalizar
            inv_std = init_scale / (std + eps)
            self.scale.data.copy_(inv_std)
            self.shift.data.copy_(-mean * inv_std)

        x = self.scale * x + self.shift ### Aplicar transformación afín
        return x

    def extra_repr(self):
        return 'in_channels={}'.format(self.in_channels)

















class Conv1x1(nn.Module):
    """
    1x1 convolutions have three types.
    Standard convolutions:  y = Wx
    PLU decomposition convolutions: y = PLUx
    Matrix exp convolutions: y = e^{W}x

    


    Convolución 1x1 invertible

    Mezclan información sin cambiar las dimensiones.
    1. hacen que la información fluya entre canales
    2. Son invertibles si la matriz de pesos lo es
    3. Su determinante es calculable eficientemente
    
    Tres implementaciones disponibles:
    
    1. STANDARD: y = Wx
       - W es una matriz aprendible inicializada ortogonalmente
       - Requiere calcular det(W) o invertir W en el paso inverso
       
    2. DECOMPOSITION (PLU): y = PLUx
       - W = PLU donde P es permutación, L lower triangular, U upper triangular
       - det(PLU) = det(P)*prod(diag(L))*prod(diag(U)) = ±prod(diag(U))
       - Más eficiente que standard
       
    3. MATRIXEXP: y = exp(W)x
       - Usa exponencial de matrices: exp(W) = Σ(W^k/k!)
       - det(exp(W)) = exp(tr(W)) (muy eficiente!)
       - Siempre invertible: exp(W)⁻¹ = exp(-W)
       - Es la implementación preferida en este código
    """

    def __init__(self, in_channels, conv_type):
        """
        Args:
            in_channels: Número de canales (entrada = salida para 1x1 conv)
            conv_type: 'standard', 'decomposition', o 'matrixexp'
        """
        super(Conv1x1, self).__init__()
        self.in_channels = in_channels
        self.conv_type = conv_type
        if not conv_type == 'decomposition':
            ### Para standard y matrixexp: una matriz de pesos
            self.weight = nn.Parameter(torch.rand(in_channels, in_channels))
        else:
            ### Para PLU decomposition: tres matrices
            self.l = nn.Parameter(torch.rand(in_channels, in_channels)) ### Lower
            self.u = nn.Parameter(torch.rand(in_channels, in_channels)) ### Upper
            p = torch.rand(in_channels, in_channels) ### Permutación 

            ### Máscaras para forzar estructura triangular
            l_mask = torch.tril(torch.ones(in_channels, in_channels), diagonal=-1)
            identity = torch.eye(in_channels)
            u_mask = torch.tril(torch.ones(in_channels, in_channels), diagonal=0).t()

            ### Registrar buffers, no son parámetros aprendibles
            self.register_buffer('p', p)
            self.register_buffer('l_mask', l_mask)
            self.register_buffer('identity', identity)
            self.register_buffer('u_mask', u_mask)

    def forward(self, x, reverse=False, init=False):
        """
        Args:
            x: Tensor [batch, channels, height, width]
            reverse: Si True, aplica transformación inversa
            init: Si True, inicializa pesos
            
        Returns:
            x: Tensor transformado
            log_det: Log-determinante multiplicado por (height * width)
        """
        if init:
            ### Inicialización
            if self.conv_type == 'matrixexp':
                ### Inicializar con logaritmo de matriz ortogonal
                ### exp(logm(Q)) = Q donde Q es ortogonal
                rand = special_ortho_group.rvs(self.in_channels) ### Matriz ortogonal especial det = 1
                rand = logm(rand) ### Logaritmo de matriz
                ### Parte real (puede tener parte imaginaria por errores)
                rand = torch.from_numpy(rand.real) 
                self.weight.data.copy_(rand)
            elif self.conv_type == 'standard':
                nn.init.orthogonal_(self.weight)
            elif self.conv_type == 'decomposition':
                ### Inicializar con descomposición PLU de matriz ortogonal
                w = ortho_group.rvs(self.in_channels)
                p, l, u = scipy.linalg.lu(w)
                self.p.copy_(torch.from_numpy(p))
                self.l.data.copy_(torch.from_numpy(l))
                self.u.data.copy_(torch.from_numpy(u))
            else:
                raise ValueError('wrong 1x1 conlution type')

        if not reverse:
            ### ============ FORWARD PASS ============
            if self.conv_type == 'matrixexp':
                ### Calcular exp(W) usando serie de Taylor <<------------------
                weight = expm(self.weight)
                ### Aplicar como convolución 1x1 (equivalente a matmul por canal)
                x = F.conv2d(x, weight.view(self.in_channels, self.in_channels, 1, 1))
                
                ### Log-det: log|det(exp(W))| = tr(W)
                ### Multiplicar por número de píxeles (height * width)
                log_det = torch.diagonal(self.weight).sum().mul(x.size(2) * x.size(3))
            elif self.conv_type == 'standard':
                ### Aplicar W directamente
                x = F.conv2d(x, self.weight.view(self.in_channels, self.in_channels, 1, 1))
                
                ### Calcular log|det(w)| usando slogdet
                _, log_det = torch.slogdet(self.weight)
                log_det = log_det.mul(x.size(2) * x.size(3))

            elif self.conv_type == 'decomposition':
                ### Reconstruir W = PLU
                ### Lower triangular con 1s en diagonal
                l = self.l * self.l_mask + self.identity 
                u = self.u * self.u_mask ### Upper triangular
                weight = torch.matmul(self.p, torch.matmul(l, u))

                x = F.conv2d(x, weight.view(self.in_channels, self.in_channels, 1, 1))
                
                ### Log-det: log|det(PLU)| = log|prod(diag(U))|
                log_det = torch.diagonal(self.u).abs().log().sum().mul(x.size(2) * x.size(3))
            else:
                raise ValueError('wrong 1x1 conlution type')
            
        else:
            # ============ INVERSE PASS ============

            if self.conv_type == 'matrixexp':
                ### Inversa: exp(-W)
                weight = expm(-self.weight)
                x = F.conv2d(x, weight.view(self.in_channels, self.in_channels, 1, 1))

                ### Log-det inverso tiene signo opuesto
                log_det = torch.diagonal(self.weight).sum().mul(x.size(2) * x.size(3)).mul(-1)
            
            elif self.conv_type == 'standard': ### inversa de W
                x = F.conv2d(x, torch.inverse(self.weight).view(self.in_channels, self.in_channels, 1, 1))
                _, log_det = torch.slogdet(self.weight)
                log_det = log_det.mul(x.size(2) * x.size(3)).mul(-1)
            elif self.conv_type == 'decomposition': ### Inversa (PLU) = Inv(U) inv(L) inv(p)
                l = self.l * self.l_mask + self.identity
                u = self.u * self.u_mask
                weight = torch.matmul(self.p, torch.matmul(l, u))
                x = F.conv2d(x, torch.inverse(weight).view(self.in_channels, self.in_channels, 1, 1))
                log_det = torch.diagonal(self.u).sum().mul(x.size(2) * x.size(3)).mul(-1)
            else:
                raise ValueError('wrong 1x1 conlution type')
        return x, log_det

    def extra_repr(self):
        return 'in_channels={}, conv_type={}'.format(self.in_channels, self.conv_type)




















class CouplingLayer(nn.Module):
    """
    Coupling layers have three types.
    Additive coupling layers: y2 = x2 + b(x1)
    Affine coupling layers: y2 = s(x1) * x2 + b(x1)
    Matrix exp coupling layers: y2 = e^{s(x1)}x2 + b(x1)

    
    Capa de Acoplamiento

    1. Divide la entrada x en dos partes: x1 y x2
    2. x1 pasa sin modificar
    3. x2 se transforma usando una función de x1
    
    Esto garantiza invertibilidad porque siempre tenemos x1 disponible 
    para la inversa y La transformación de x2 puede ser compleja 
    
    Tipos de transformaciones:
    
    1. ADDITIVE: y2 = x2 + NN(x1)
       - Más simple
       - Jacobiano = identidad (log-det = 0)
       
    2. AFFINE: y₂ = exp(s(x1)) * x2 + t(x1)
       - s(x1) = escala, t(x1) = traslación
       - Más expresivo que additive
       - Usado en RealNVP
       
    3. MATRIXEXP: y₂ = exp(S(x1)) * x2 + t(x1)
       - S(x1) es una MATRIZ (no escalar como en affine)
       - Cada elemento de x2 puede interactuar con todos los demás
       - Más expresivo pero más costoso computacionalmente
       - Usa aproximaciones para matrices grandes
    """

    def __init__(self, flow_type, num_blocks, in_channels, hidden_channels):
        """
        Args:
            flow_type: 'additive', 'affine', o 'matrixexp'
            num_blocks: Número de bloques en la red neuronal
            in_channels: Canales totales de entrada
            hidden_channels: Canales ocultos en la red
        """

        super(CouplingLayer, self).__init__()
        self.flow_type = flow_type
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        ### Dividir canales en 2 grupos
        self.x2_channels = in_channels // 2 ### Segunda mitad
        self.x1_channels = in_channels - self.x2_channels ### Primera mitad, puede ser +1 si impar
        ### Configurar según tipo de flow
        if flow_type == 'additive':
            ### Solo se necesita predecir traslación
            self.num_out = 1
        elif flow_type == 'affine':
            # Se necesita predecir escala y traslación
            # Parámetros para estabilizar el entrenamiento
            self.scale = nn.Parameter(torch.ones(1) / 8) ### Escala pequeña inicial
            self.shift = nn.Parameter(torch.zeros(1))
            self.rescale = nn.Parameter(torch.ones(1))
            self.reshift = nn.Parameter(torch.zeros(1))
            self.num_out = 2 ### scale y shift
        elif flow_type == 'matrixexp':
            ### Predecir matriz + traslación
            self.scale = nn.Parameter(torch.ones(1) / 8)
            self.shift = nn.Parameter(torch.zeros(1))
            self.rescale = nn.Parameter(torch.ones(1) / self.x2_channels)
            self.reshift = nn.Parameter(torch.zeros(1))
            ### Para matrices pequeñas: parametrización directa
            ### Para matrices grandes: factorización de bajo rango
            self.max_out = 24
            if self.x2_channels <= self.max_out:
                ### Parametrización completa: x2_channels x x2_channels matriz
                self.num_out = (self.x2_channels + 1)
            else:
                ### Factorización: W = I + AB^T donde A,B ∈ R^{x2_channels x k}
                self.k = 3 ### Rango de la aproximación
                self.num_out = 2 * self.k + 1 ### k para A, k para B, 1 para traslación
        else:
            raise ValueError('wrong flow type')
        
        ### Red neuronal que predice los parámetros de transformación
        ### Entrada: x1, Salida: parámetros de transformación para x2
        self.net = ConvBlock(num_blocks, self.x1_channels, self.hidden_channels, self.x2_channels * self.num_out)

    def forward(self, x, reverse=False, init=False):
        """
        Args:
            x: Tensor [batch, channels, height, width]
            reverse: Si True, aplica transformación inversa
            init: Si True, inicializa la red neuronal
            
        Returns:
            out: Tensor transformado
            log_det: Log-determinante del Jacobiano
        """
        ### Dividir entrada en dos partes
        x1 = x[:, :self.x1_channels]  ### Primera parte no se modifica
        x2 = x[:, self.x1_channels:]  ### Segunda parte se transforma
        if self.flow_type == 'additive':
            ### ============ ADDITIVE COUPLING ============
            if not reverse:
                ### Forward: y2 = x2 + NN(x1)
                x2 = x2 + self.net(x1, init=init)
                out = torch.cat([x1, x2], dim=1)
                ### Log-det = 0, la transformación por traslación no cambia volumen
                log_det = x.new_zeros(x.size(0))
            else:
                ### Inverse: x2 = y2 - NN(x1)
                x2 = x2 - self.net(x1)
                out = torch.cat([x1, x2], dim=1)
                log_det = x.new_zeros(x.size(0))
        elif self.flow_type == 'affine':
            ### ============ AFFINE COUPLING ============
            if not reverse:
                ### Predecir parámetros de transformación
                out = self.net(x1, init=init)
                outs = out.chunk(2, dim=1)  ### Dividir en shift y log_scale
                shift = outs[0]

                ### Estabilizar log_scale usando tanh y parámetros aprendibles
                ### Esto mantiene los valores en un rango razonable
                log_scale = self.rescale * torch.tanh(self.scale * outs[1] + self.shift) + self.reshift
                
                ### Forward: y2 = exp(log_scale) * x2 + shift
                x2 = torch.exp(log_scale) * x2 + shift

                ### Log-det: sum(log_scale)
                out = torch.cat([x1, x2], dim=1)
                log_det = log_scale.sum([1, 2, 3])
            else:
                ### Inverse: x₂ = (y₂ - shift) / exp(log_scale)
                out = self.net(x1)
                outs = out.chunk(2, dim=1)
                shift = outs[0]
                log_scale = self.rescale * torch.tanh(self.scale * outs[1] + self.shift) + self.reshift
                
                x2 = torch.exp(-log_scale) * (x2 - shift)
                out = torch.cat([x1, x2], dim=1)
                log_det = log_scale.sum([1, 2, 3]).mul(-1)

        elif self.flow_type == 'matrixexp':
            ### ============ MATRIX EXPONENTIAL COUPLING ============
            if not reverse:
                if self.x2_channels <= self.max_out:
                    ### CASO 1: Parametrización completa para dimensiones pequeñas
                    ### Predecir parámetros
                    out = self.net(x1, init=init).unsqueeze(2)
                    outs = out.chunk(self.num_out, dim=1)
                    shift = outs[0].squeeze(2)

                    ### Construir matriz de pesos
                    ### weight tiene forma [batch, height, width, x2_channels, x2_channels]
                    weight = torch.cat(outs[1:], dim=2).permute(0, 3, 4, 1, 2)
                    weight = self.rescale * torch.tanh(self.scale * weight + self.shift) + self.reshift
                    
                    ### Aplicar exp(W) a x2
                    x2 = x2.unsqueeze(2).permute(0, 3, 4, 1, 2) ### Reorganizar para matmul
                    x2 = torch.matmul(expm(weight), x2).permute(0, 3, 4, 1, 2).squeeze(2) + shift
                    out = torch.cat([x1, x2], dim=1)
                    
                    ### Log-det: tr(W)
                    log_det = torch.diagonal(weight, dim1=-2, dim2=-1).sum([1, 2, 3])
                else:
                    ### CASO 2: Aproximación de bajo rango para dimensiones grandes
                    ### Usar factorización: exp(W) = I + exp(AB^T)B^T A
                    ### donde A, B son R^{x2_channels x k}

                    out = self.net(x1, init=init).unsqueeze(2)
                    outs = out.chunk(self.num_out, dim=1)
                    shift = outs[0].squeeze(2)

                    ### Extraer factores A y B
                    weight1 = torch.cat(outs[1:self.k + 1], dim=2).permute(0, 3, 4, 2, 1)
                    weight2 = torch.cat(outs[self.k + 1:2 * self.k + 1], dim=2).permute(0, 3, 4, 1, 2)
                    
                    ### Estabilizar
                    weight1 = self.rescale * torch.tanh(self.scale * weight1 + self.shift) + self.reshift + eps
                    weight2 = self.rescale * torch.tanh(self.scale * weight2 + self.shift) + self.reshift + eps
                    
                    ### Calcular AB^T
                    weight3 = torch.matmul(weight1, weight2)

                    ### Calcular W = I + B * series(AB^T) * A^T
                    ### series(X) = suma X^k/(k+1)! es una aproximación más estable
                    weight = torch.eye(self.x2_channels, device=x.device) + torch.matmul(
                        torch.matmul(weight2, series(weight3)), weight1)
                    
                    ### Aplicar transformación
                    x2 = x2.unsqueeze(2).permute(0, 3, 4, 1, 2)
                    x2 = torch.matmul(weight, x2).permute(0, 3, 4, 1, 2).squeeze(2) + shift
                    out = torch.cat([x1, x2], dim=1)
                    ### Log-det aproximado usando tr(AB^T)
                    log_det = torch.diagonal(weight3, dim1=-2, dim2=-1).sum([1, 2, 3])
            else:
                ### INVERSE PASS (similar estructura pero con signos invertidos)
                if self.x2_channels <= self.max_out:
                    out = self.net(x1).unsqueeze(2)
                    outs = out.chunk(self.num_out, dim=1)
                    shift = outs[0].squeeze(2)
                    weight = torch.cat(outs[1:], dim=2).permute(0, 3, 4, 1, 2)
                    weight = self.rescale * torch.tanh(self.scale * weight + self.shift) + self.reshift
                    
                    ### Aplicar exp(-W)
                    x2 = (x2 - shift).unsqueeze(2).permute(0, 3, 4, 1, 2)
                    x2 = torch.matmul(expm(-weight), x2).permute(0, 3, 4, 1, 2).squeeze(2)
                    out = torch.cat([x1, x2], dim=1)
                    log_det = torch.diagonal(weight, dim1=-2, dim2=-1).sum([1, 2, 3]).mul(-1)
                else:
                    out = self.net(x1).unsqueeze(2)
                    outs = out.chunk(self.num_out, dim=1)
                    shift = outs[0].squeeze(2)
                    weight1 = torch.cat(outs[1:self.k + 1], dim=2).permute(0, 3, 4, 2, 1)
                    weight2 = torch.cat(outs[self.k + 1:2 * self.k + 1], dim=2).permute(0, 3, 4, 1, 2)
                    weight1 = self.rescale * torch.tanh(self.scale * weight1 + self.shift) + self.reshift + eps
                    weight2 = self.rescale * torch.tanh(self.scale * weight2 + self.shift) + self.reshift + eps
                    weight3 = torch.matmul(weight1, weight2)

                    # Inversa: W-1 = I - B * series(-AB^T) * A^T
                    weight = torch.eye(self.x2_channels, device=x.device) - torch.matmul(
                        torch.matmul(weight2, series(-weight3)), weight1)
                    
                    x2 = (x2 - shift).unsqueeze(2).permute(0, 3, 4, 1, 2)
                    x2 = torch.matmul(weight, x2).permute(0, 3, 4, 1, 2).squeeze(2)
                    out = torch.cat([x1, x2], dim=1)
                    log_det = torch.diagonal(weight3, dim1=-2, dim2=-1).sum([1, 2, 3]).mul(-1)
        else:
            raise ValueError('wrong flow type')

        return out, log_det

    def extra_repr(self):
        return 'in_channels={}, hidden_channels={}, out_channels={},flow_type={}'.format(self.in_channels,
                                                                                         self.hidden_channels,
                                                                                         self.in_channels,
                                                                                         self.flow_type)










class ConvBlock(nn.Module):
    """
    Bloque convolucional para predecir parámetros de transformación.
    
    Arquitectura:
    1. Capa convolucional inicial (in_channels -> hidden_channels)
    2. Varios bloques residuales (Block)
    3. Capa convolucional final (hidden_channels -> out_channels)
    
    La capa final se inicializa con pesos cero para que al inicio
    del entrenamiento la transformación sea la identidad.
    """
    def __init__(self, num_blocks, in_channels, hidden_channels, out_channels):
        """
        Args:
            num_blocks: Número de bloques residuales
            in_channels: Canales de entrada
            hidden_channels: Canales ocultos
            out_channels: Canales de salida
        """
        super(ConvBlock, self).__init__()
        layers = list()

        ### Primera capa: expandir a hidden_channels con activación
        layers.append(Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, activ=True))
        
        ### Bloques residuales
        for _ in range(num_blocks):
            layers.append(Block(hidden_channels))

        self.layers = nn.ModuleList(layers)

        # Capa de salida: reducir a out_channels
        self.out_layer = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)
        
        # INICIALIZACIÓN CERO: La transformación inicial es identidad
        # Esto es importante para estabilidad en el inicio del entrenamiento
        nn.init.constant_(self.out_layer.weight, 0.0)
        nn.init.constant_(self.out_layer.bias, 0.0)

    def forward(self, x, init=False):
        """
        Args:
            x: Tensor de entrada
            init: Si True, inicializa las capas de normalización
            
        Returns:
            x: Predicciones de parámetros para la transformación
        """
        for layer in self.layers:
            x = layer(x, init=init)
        x = self.out_layer(x)
        return x












class Block(nn.Module):
    """
    Bloque residual con arquitectura bottleneck.
    
    Estructura:
    x -> Conv3x3 -> ELU -> Conv1x1 -> ELU -> Conv3x3 -> (+) -> ELU
    |___________________________________________________|
    
    - La primera conv3x3 extrae características locales
    - La conv1x1 mezcla información entre canales (bottleneck)
    - La segunda conv3x3 refina las características
    - La conexión residual (+) ayuda con el flujo de gradientes
    - La última conv3x3 se inicializa con scale=0 para comenzar como identidad
    """
    def __init__(self, channels):
        """
        Args:
            channels: Número de canales (entrada = salida)
        """
        super(Block, self).__init__()
        self.conv1 = Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = Conv2d(channels, channels, kernel_size=1, padding=0)
        self.conv3 = Conv2d(channels, channels, kernel_size=3, padding=1)
        self.elu = nn.ELU() ### ELU: max(0, x) para x>=0, α(exp(x)-1) para x<0

    def forward(self, x, init=False):
        """
        Args:
            x: Tensor de entrada
            init: Si True, inicializa normalizaciones
            
        Returns:
            out: Tensor de salida (misma forma que entrada)
        """
        identity = x ### Guardar para conexión residual

        ### Path principal
        out = self.elu(self.conv1(x, init=init))
        out = self.elu(self.conv2(out, init=init))
        ### conv3 se inicializa con scale=0 para que al inicio out ≈ 0
        out = self.conv3(out, init=init, init_scale=0.0)

        ### Conexión residual
        out += identity
        out = self.elu(out)
        return out












class Conv2d(nn.Module):
    """
    Capa convolucional 2D con normalización data-dependent.
    
    Características:
    - Convolución estándar sin bias (bias se maneja con normalización)
    - Normalización por canal (Norm layer)
    - Activación ELU opcional
    - Inicialización data-dependent para estabilidad
    
    La inicialización data-dependent ajusta los pesos para que las
    activaciones tengan media 0 y varianza 1 en el primer batch.
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding, activ=False):
        """
        Args:
            in_channels: Canales de entrada
            out_channels: Canales de salida
            kernel_size: Tamaño del kernel
            padding: Padding para mantener dimensiones espaciales
            activ: Si True, aplica ELU después de normalización
        """
        super(Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activ = activ

        ### Convolución sin bias (el bias se incorpora en la normalización)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        
        ### Normalización por canal
        self.norm = Norm(out_channels)

        ### Inicialización Kaiming: apropiada para ReLU/ELU
        nn.init.kaiming_uniform_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0.0)
        if activ: ### Activación opcional
            self.elu = nn.ELU(inplace=True)

    def forward(self, x, init=False, init_scale=1.0):
        """
        Args:
            x: Tensor de entrada [batch, in_channels, height, width]
            init: Si True, inicializa pesos y normalización
            init_scale: Factor de escala para inicialización
            
        Returns:
            x: Tensor de salida [batch, out_channels, height, width]
        """
        if init:
            ### INICIALIZACIÓN DATA-DEPENDENT
            ### 1. Hacer un forward pass para ver las estadísticas de salida
            out = self.forward(x)
            n_channels = out.size(1)

            ### 2. Calcular media y std de las activaciones
            out = out.transpose(0, 1).contiguous().view(n_channels, -1)
            mean = out.mean(dim=1)
            std = out.std(dim=1)

            ### 3. Ajustar pesos de convolución para normalizar salidas
            inv_std = 1.0 / (std + eps)
            ### Escalar pesos para que std(salida) = 1
            self.conv.weight.data.mul_(inv_std.view(n_channels, 1, 1, 1))
            ### 4. Ajustar bias si existe para que mean(salida) = 0
            if self.conv.bias is not None:
                self.conv.bias.data.add_(-mean).mul_(inv_std)
                
        ### Forward pass normal
        x = self.conv(x)
        x = self.norm(x, init=init, init_scale=init_scale)
        if self.activ:
            x = self.elu(x)
        return x

    def extra_repr(self):
        return 'in_channels={}, out_channels={}'.format(self.in_channels, self.out_channels)
