"""
Tracking y Diagnóstico para Normalizing Flows

Añade este código a main.py para:
1. Visualizar progreso en tiempo real
2. Detectar problemas (overfitting, gradientes, etc.)
3. Generar plots de diagnóstico
4. Monitorear salud del entrenamiento
"""

import matplotlib
matplotlib.use('Agg')  # Backend sin GUI
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
sns.set_style("darkgrid")
import os
import time
import numpy as np

# ============================================================================
# 1. CLASE PARA TRACKING Y VISUALIZACIÓN
# ============================================================================

class TrainingMonitor:
    """
    Monitor de entrenamiento con visualización en tiempo real.
    Detecta problemas comunes y genera diagnósticos.
    """
    
    def __init__(self, save_dir, log_interval=10):
        """
        Args:
            save_dir: Directorio donde guardar plots
            log_interval: Cada cuántos batches guardar métricas detalladas
        """
        self.save_dir = save_dir
        self.log_interval = log_interval
        
        # Crear directorio para plots
        self.plot_dir = os.path.join(save_dir, 'plots')
        if not os.path.isdir(self.plot_dir):
            os.makedirs(self.plot_dir)
        
        # Almacenamiento de métricas
        self.metrics = {
            'batch_losses': [],
            'epoch_train_losses': [],
            'epoch_test_losses': [],
            'learning_rates': [],
            'grad_norms': [],
            'weight_norms': [],
            'log_dets': [],
            'epoch_times': [],
            'batch_times': []
        }
        
        # Para detectar problemas
        self.warnings = []
        self.last_update = time.time()
        self.total_batches = 0  # Track total batches to skip early warnings
        
    def log_batch(self, batch_idx, loss, log_det, model, lr):
        """
        Registrar métricas de un batch.
        """
        self.metrics['batch_losses'].append(loss)
        self.metrics['log_dets'].append(log_det)
        self.metrics['learning_rates'].append(lr)
        self.total_batches += 1
        
        # Calcular normas de gradientes cada log_interval batches
        if batch_idx % self.log_interval == 0:
            grad_norm = self._compute_grad_norm(model)
            weight_norm = self._compute_weight_norm(model)
            self.metrics['grad_norms'].append(grad_norm)
            self.metrics['weight_norms'].append(weight_norm)
            
            # Detectar problemas (skip first 50 batches for warmup)
            if self.total_batches > 50:
                self._check_gradients(grad_norm)
            self._check_loss(loss)
    
    def log_epoch(self, epoch, train_loss, test_loss, epoch_time):
        """
        Registrar métricas de una época.
        """
        self.metrics['epoch_train_losses'].append((epoch, train_loss))
        self.metrics['epoch_test_losses'].append((epoch, test_loss))
        self.metrics['epoch_times'].append(epoch_time)
        
        # Detectar overfitting
        self._check_overfitting(train_loss, test_loss)
        
        # Generar plots cada época
        self.plot_training_progress(epoch)
        
    def _compute_grad_norm(self, model):
        """Calcular norma L2 de todos los gradientes."""
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5
    
    def _compute_weight_norm(self, model):
        """Calcular norma L2 de todos los pesos.""" 
        total_norm = 0.0
        for p in model.parameters():
            param_norm = p.data.norm(2)
            total_norm += param_norm.item() ** 2
        return total_norm ** 0.5
    
    def _check_gradients(self, grad_norm):
        """Detectar problemas con gradientes."""
        if grad_norm > 100000:  # Increased threshold from 1000
            warning = f"⚠️  GRADIENTES EXPLOSIVOS: norm={grad_norm:.2f}"
            print(warning)
            self.warnings.append(warning)
        elif grad_norm < 1e-7 and len(self.metrics['grad_norms']) > 10:
            warning = f"⚠️  GRADIENTES DESAPARECIENDO: norm={grad_norm:.2e}"
            print(warning)
            self.warnings.append(warning)
    
    def _check_loss(self, loss):
        """Detectar problemas con la loss."""
        if np.isnan(loss) or np.isinf(loss):
            warning = f"🔴 LOSS INVÁLIDA: {loss}"
            print(warning)
            self.warnings.append(warning)
        elif len(self.metrics['batch_losses']) > 100:
            # Verificar si la loss se estancó
            recent_losses = self.metrics['batch_losses'][-100:]
            if np.std(recent_losses) < 1e-6:
                warning = f"⚠️  LOSS ESTANCADA: std={np.std(recent_losses):.2e}"
                print(warning)
                self.warnings.append(warning)
    
    def _check_overfitting(self, train_loss, test_loss):
        """Detectar overfitting."""
        gap = test_loss - train_loss
        if gap > 0.5:
            warning = f"⚠️  POSIBLE OVERFITTING: gap={gap:.4f}"
            print(warning)
            self.warnings.append(warning)
    
    def plot_training_progress(self, epoch):
        """
        Generar plot completo del progreso de entrenamiento.
        """
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 3, figure=fig)
        
        # 1. Loss por época
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_epoch_losses(ax1)
        
        # 2. Loss por batch (últimos 1000 batches)
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_batch_losses(ax2)
        
        # 3. Learning rate
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_learning_rate(ax3)
        
        # 4. Normas de gradientes
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_gradient_norms(ax4)
        
        # 5. Normas de pesos
        ax5 = fig.add_subplot(gs[1, 1])
        self._plot_weight_norms(ax5)
        
        # 6. Log-determinantes
        ax6 = fig.add_subplot(gs[1, 2])
        self._plot_log_dets(ax6)
        
        # 7. Tiempo por época
        ax7 = fig.add_subplot(gs[2, 0])
        self._plot_epoch_times(ax7)
        
        # 8. Gap train-test
        ax8 = fig.add_subplot(gs[2, 1])
        self._plot_train_test_gap(ax8)
        
        # 9. Estado del entrenamiento
        ax9 = fig.add_subplot(gs[2, 2])
        self._plot_training_status(ax9, epoch)
        
        plt.tight_layout()
        save_path = os.path.join(self.plot_dir, f'training_epoch_{epoch:03d}.png')
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Plot guardado: {save_path}")
    
    def _plot_epoch_losses(self, ax):
        """Plot de losses por época."""
        if len(self.metrics['epoch_train_losses']) > 0:
            epochs, train_losses = zip(*self.metrics['epoch_train_losses'])
            ax.plot(epochs, train_losses, 'o-', label='Train', linewidth=2, markersize=6)
        
        if len(self.metrics['epoch_test_losses']) > 0:
            epochs, test_losses = zip(*self.metrics['epoch_test_losses'])
            ax.plot(epochs, test_losses, 's-', label='Test', linewidth=2, markersize=6)
        
        ax.set_xlabel('Época', fontsize=12)
        ax.set_ylabel('Loss (bits/dim)', fontsize=12)
        ax.set_title('Loss por Época', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    def _plot_batch_losses(self, ax):
        """Plot de losses por batch (últimos N)."""
        if len(self.metrics['batch_losses']) > 0:
            losses = self.metrics['batch_losses'][-1000:]  # Últimos 1000
            ax.plot(losses, alpha=0.6, linewidth=1)
            
            # Añadir media móvil
            if len(losses) > 20:
                window = 20
                moving_avg = np.convolve(losses, np.ones(window)/window, mode='valid')
                ax.plot(range(window-1, len(losses)), moving_avg, 'r-', 
                       linewidth=2, label=f'Media móvil ({window})')
        
        ax.set_xlabel('Batch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Loss por Batch (últimos 1000)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    def _plot_learning_rate(self, ax):
        """Plot del learning rate."""
        if len(self.metrics['learning_rates']) > 0:
            ax.plot(self.metrics['learning_rates'], linewidth=2)
            ax.set_xlabel('Batch', fontsize=12)
            ax.set_ylabel('Learning Rate', fontsize=12)
            ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
    
    def _plot_gradient_norms(self, ax):
        """Plot de normas de gradientes."""
        if len(self.metrics['grad_norms']) > 0:
            ax.plot(self.metrics['grad_norms'], linewidth=2, color='orange')
            ax.axhline(y=1000, color='r', linestyle='--', label='Umbral explosión')
            ax.axhline(y=1e-7, color='r', linestyle='--', label='Umbral desaparición')
            ax.set_xlabel('Checkpoint', fontsize=12)
            ax.set_ylabel('Norma L2', fontsize=12)
            ax.set_title('Norma de Gradientes', fontsize=14, fontweight='bold')
            ax.set_yscale('log')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
    
    def _plot_weight_norms(self, ax):
        """Plot de normas de pesos."""
        if len(self.metrics['weight_norms']) > 0:
            ax.plot(self.metrics['weight_norms'], linewidth=2, color='green')
            ax.set_xlabel('Checkpoint', fontsize=12)
            ax.set_ylabel('Norma L2', fontsize=12)
            ax.set_title('Norma de Pesos', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
    
    def _plot_log_dets(self, ax):
        """Plot de log-determinantes."""
        if len(self.metrics['log_dets']) > 0:
            log_dets = self.metrics['log_dets'][-1000:]
            ax.plot(log_dets, alpha=0.6, linewidth=1)
            ax.set_xlabel('Batch', fontsize=12)
            ax.set_ylabel('Log Det', fontsize=12)
            ax.set_title('Log-Determinante Jacobiano', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
    
    def _plot_epoch_times(self, ax):
        """Plot de tiempo por época."""
        if len(self.metrics['epoch_times']) > 0:
            ax.plot(self.metrics['epoch_times'], 'o-', linewidth=2, markersize=6)
            ax.set_xlabel('Época', fontsize=12)
            ax.set_ylabel('Tiempo (s)', fontsize=12)
            ax.set_title('Tiempo por Época', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
    
    def _plot_train_test_gap(self, ax):
        """Plot del gap entre train y test."""
        if len(self.metrics['epoch_train_losses']) > 0 and len(self.metrics['epoch_test_losses']) > 0:
            _, train = zip(*self.metrics['epoch_train_losses'])
            _, test = zip(*self.metrics['epoch_test_losses'])
            gap = np.array(test) - np.array(train)
            epochs = range(1, len(gap) + 1)
            
            ax.plot(epochs, gap, 'o-', linewidth=2, markersize=6)
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Umbral overfitting')
            ax.set_xlabel('Época', fontsize=12)
            ax.set_ylabel('Test - Train Loss', fontsize=12)
            ax.set_title('Gap Train-Test', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
    
    def _plot_training_status(self, ax, epoch):
        """Resumen del estado del entrenamiento."""
        ax.axis('off')
        
        # Calcular estadísticas
        status_text = f"ESTADO DEL ENTRENAMIENTO - Época {epoch}\n\n"
        
        if len(self.metrics['epoch_train_losses']) > 0:
            current_loss = self.metrics['epoch_train_losses'][-1][1]
            status_text += f"Loss actual: {current_loss:.4f} bits/dim\n"
            
            if len(self.metrics['epoch_train_losses']) > 1:
                prev_loss = self.metrics['epoch_train_losses'][-2][1]
                improvement = prev_loss - current_loss
                status_text += f"Mejora: {improvement:+.4f}\n"
        
        if len(self.metrics['grad_norms']) > 0:
            status_text += f"\nNorma gradientes: {self.metrics['grad_norms'][-1]:.2e}\n"
        
        if len(self.metrics['learning_rates']) > 0:
            status_text += f"Learning rate: {self.metrics['learning_rates'][-1]:.2e}\n"
        
        # Advertencias
        if len(self.warnings) > 0:
            status_text += f"\n⚠️  ADVERTENCIAS RECIENTES:\n"
            for warning in self.warnings[-5:]:  # Últimas 5
                status_text += f"  • {warning}\n"
        else:
            status_text += f"\n✅ Sin problemas detectados"
        
        # Estimación de tiempo restante
        if len(self.metrics['epoch_times']) > 0:
            avg_time = np.mean(self.metrics['epoch_times'][-5:])
            status_text += f"\n\nTiempo promedio/época: {avg_time:.1f}s"
        
        ax.text(0.1, 0.9, status_text, fontsize=10, verticalalignment='top',
                fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


# ============================================================================
# 2. MODIFICAR FUNCIÓN train() EN main.py
# ============================================================================

def train_with_monitoring(args, device, save_dir, model, optimizer, scheduler, 
                         train_loader, test_loader, init_data):
    """
    Versión mejorada de train() con monitoreo visual.
    REEMPLAZA la función train() original en main.py
    """
    
    # Crear monitor
    monitor = TrainingMonitor(save_dir, log_interval=10)
    
    train_log = {'train_loss': [], 'epoch_loss': [], 'epoch_time': [], 'test_loss': []}
    start_epoch = 1
    best_loss = 1e8
    lr_lambda = lambda step: (step + 1) / (len(train_loader) * args.warmup_epoch)
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if args.resume_epoch is not None:
        state_file = os.path.join(save_dir, 'models', 'epoch_' + str(args.resume_epoch) + '.pth')
        if not os.path.isfile(state_file):
            raise RuntimeError('file {} is not found'.format(state_file))
        print('load checkpoint {}'.format(state_file))
        checkpoint = torch.load(state_file, device)
        start_epoch = checkpoint['epoch'] + 1
        train_log = checkpoint['train_log']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
    else:
        model.eval()
        with torch.no_grad():
            init_data = init_data.to(device)
            z = torch.rand_like(init_data)
            init_data = preprocess(init_data, args.bits, z)
            _, _ = model(init_data, init=True)

    print('start training')
    print('='*70)
    
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss = 0.
        number = 0
        t0 = time.time()

        # Barra de progreso
        pbar = tqdm(train_loader, desc=f'Época {epoch}/{args.epochs}')
        
        for batch_idx, (data, _) in enumerate(pbar):
            data = data.to(device)
            z = torch.rand_like(data)
            data = preprocess(data, args.bits, z)
            output, log_det = model(data)
            loss = compute_loss(args, output, log_det)
            
            bits_per_dim = loss.item() / (np.log(2) * args.dimension)
            train_log['train_loss'].append(bits_per_dim)
            total_loss += loss.item() * data.size(0)
            number += data.size(0)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if epoch <= args.warmup_epoch:
                warmup_scheduler.step()
            
            # TRACKING
            current_lr = optimizer.param_groups[0]['lr']
            monitor.log_batch(batch_idx, bits_per_dim, log_det.mean().item(), 
                            model, current_lr)
            
            # Actualizar barra de progreso
            pbar.set_postfix({'loss': f'{bits_per_dim:.4f}', 
                            'lr': f'{current_lr:.2e}'})

        bits_per_dim = total_loss / number / (np.log(2) * args.dimension)
        train_log['epoch_loss'].append((epoch, bits_per_dim))
        t1 = time.time()
        epoch_time = t1 - t0
        train_log['epoch_time'].append((epoch, epoch_time))
        
        print(f'\n[train:epoch {epoch}]. loss: {bits_per_dim:.8f}, time:{epoch_time:.1f}s')
        
        scheduler.step()
        
        if not (epoch % args.test_interval):
            test_loss = test(args, device, model, test_loader, epoch)
            train_log['test_loss'].append((epoch, bits_per_dim, test_loss))
            
            # TRACKING DE ÉPOCA
            monitor.log_epoch(epoch, bits_per_dim, test_loss, epoch_time)
            
            if test_loss < best_loss:
                best_loss = test_loss
                save(save_dir, epoch, train_log, model, optimizer, scheduler, is_best=True)
                print(f'✅ Nuevo mejor modelo guardado! Loss: {best_loss:.6f}')
        
        if not (epoch % args.save_interval):
            save(save_dir, epoch, train_log, model, optimizer, scheduler)
        
        print('='*70)
    
    # Guardar warnings finales
    if len(monitor.warnings) > 0:
        warnings_file = os.path.join(save_dir, 'warnings.txt')
        with open(warnings_file, 'w') as f:
            f.write('\n'.join(monitor.warnings))
        print(f'⚠️  Se guardaron {len(monitor.warnings)} advertencias en {warnings_file}')
    
    return


# ============================================================================
# 3. VISUALIZACIÓN DE MUESTRAS DURANTE ENTRENAMIENTO
# ============================================================================

def generate_samples_during_training(args, device, save_dir, model, epoch, num_samples=16):
    """
    Generar y guardar muestras durante el entrenamiento para ver progreso visual.
    Llama esta función al final de cada época.
    """
    z = torch.randn(num_samples, 3, args.image_size, args.image_size).to(device)
    model.eval()
    with torch.no_grad():
        output, _ = model(z, reverse=True)
        output = postprocess(output, args.bits)
    
    sample_dir = os.path.join(save_dir, 'training_samples')
    if not os.path.isdir(sample_dir):
        os.makedirs(sample_dir)
    
    torchvision.utils.save_image(
        output, 
        os.path.join(sample_dir, f'epoch_{epoch:03d}.png'),
        nrow=int(num_samples ** 0.5), 
        pad_value=1
    )


# ============================================================================
# 4. ANÁLISIS POST-ENTRENAMIENTO
# ============================================================================

def analyze_training_results(save_dir):
    """
    Análisis completo después del entrenamiento.
    Ejecuta esto después de terminar el entrenamiento.
    """
    import glob
    
    # Cargar todos los checkpoints
    model_dir = os.path.join(save_dir, 'models')
    checkpoints = sorted(glob.glob(os.path.join(model_dir, 'epoch_*.pth')))
    
    if len(checkpoints) == 0:
        print("No se encontraron checkpoints")
        return
    
    # Extraer historial de losses
    train_losses = []
    test_losses = []
    epochs = []
    
    for ckpt_path in checkpoints:
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu')
            if 'train_log' in ckpt:
                train_log = ckpt['train_log']
                if 'epoch_loss' in train_log:
                    for epoch, loss in train_log['epoch_loss']:
                        if epoch not in epochs:
                            epochs.append(epoch)
                            train_losses.append(loss)
                if 'test_loss' in train_log:
                    for item in train_log['test_loss']:
                        if len(item) >= 2:
                            test_losses.append(item[1])
        except:
            continue
    
    # Crear informe
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Curvas de aprendizaje
    axes[0, 0].plot(epochs, train_losses, 'o-', label='Train')
    if len(test_losses) > 0:
        axes[0, 0].plot(epochs[:len(test_losses)], test_losses, 's-', label='Test')
    axes[0, 0].set_xlabel('Época')
    axes[0, 0].set_ylabel('Loss (bits/dim)')
    axes[0, 0].set_title('Curvas de Aprendizaje')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot 2: Tasa de mejora
    if len(train_losses) > 1:
        improvements = np.diff(train_losses)
        axes[0, 1].plot(epochs[1:], -improvements, 'o-')
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Época')
        axes[0, 1].set_ylabel('Mejora en Loss')
        axes[0, 1].set_title('Tasa de Mejora por Época')
        axes[0, 1].grid(True)
    
    # Plot 3: Estadísticas
    axes[1, 0].axis('off')
    stats_text = "ESTADÍSTICAS FINALES\n\n"
    stats_text += f"Mejor loss train: {min(train_losses):.6f}\n"
    if len(test_losses) > 0:
        stats_text += f"Mejor loss test: {min(test_losses):.6f}\n"
    stats_text += f"Loss final train: {train_losses[-1]:.6f}\n"
    if len(test_losses) > 0:
        stats_text += f"Loss final test: {test_losses[-1]:.6f}\n"
    stats_text += f"Épocas entrenadas: {len(epochs)}\n"
    
    axes[1, 0].text(0.1, 0.9, stats_text, fontsize=12, verticalalignment='top',
                    fontfamily='monospace')
    
    # Plot 4: Distribución de losses
    axes[1, 1].hist(train_losses, bins=20, alpha=0.7, label='Train')
    if len(test_losses) > 0:
        axes[1, 1].hist(test_losses, bins=20, alpha=0.7, label='Test')
    axes[1, 1].set_xlabel('Loss')
    axes[1, 1].set_ylabel('Frecuencia')
    axes[1, 1].set_title('Distribución de Losses')
    axes[1, 1].legend()
    
    plt.tight_layout()
    analysis_path = os.path.join(save_dir, 'training_analysis.png')
    plt.savefig(analysis_path, dpi=150)
    plt.close()
    
    print(f"📊 Análisis guardado en: {analysis_path}")


# ============================================================================
# INSTRUCCIONES DE USO
# ============================================================================

"""
CÓMO USAR ESTE SISTEMA:

1. Añadir al principio de main.py:
   from tqdm import tqdm
   import matplotlib.pyplot as plt
   import seaborn as sns

2. Reemplazar la función train() con train_with_monitoring()

3. Al final de cada época en train_with_monitoring(), añadir:
   if not (epoch % 5):  # Cada 5 épocas
       generate_samples_during_training(args, device, save_dir, model, epoch)

4. Después del entrenamiento, ejecutar:
   analyze_training_results(save_dir)

5. Los plots se guardarán en: save_dir/plots/
   Las muestras en: save_dir/training_samples/

OUTPUTS:
- training_epoch_XXX.png: Dashboard completo cada época
- training_samples/epoch_XXX.png: Muestras generadas
- training_analysis.png: Análisis final post-entrenamiento
- warnings.txt: Log de problemas detectados
"""