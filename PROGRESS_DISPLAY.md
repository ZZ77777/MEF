# Sistema de Visualización de Progreso

Se ha implementado un sistema mejorado de visualización de progreso en tiempo real usando `tqdm`. Ahora verás:

## 🔄 Barra de Progreso por Época

Durante el entrenamiento de cada época:
```
Época 1/50: 45%|████▌     | 360/800 [00:45<01:05, 6.55 batch/s, loss=3.2145, lr=1.00e-03]
```

**Información mostrada:**
- ✅ Porcentaje completado
- ✅ Número de batches procesados / total
- ✅ Tiempo transcurrido / tiempo estimado restante
- ✅ Velocidad de procesamiento (batches/segundo)
- ✅ **Loss actual en tiempo real** (bits/dim)
- ✅ **Learning rate actual**

## 📊 Resumen de Época

Después de cada época:
```
╔════════════════════════════════════════════╗
║ Época 1/50 - Loss: 3.214562 bits/dim | Tiempo: 110.2s
╚════════════════════════════════════════════╝
```

## 🧪 Evaluación en Validación

Cuando se evalúa el conjunto de test:
```
Evaluando en conjunto de validación...

┌─ TEST LOSS (Polyak avg): 3.102541 bits/dim | Gap: -0.112021 ✓ Muy bueno
└─ TEST LOSS (Current):   3.115234 bits/dim
```

**Información mostrada:**
- Loss con parámetros promediados (Polyak)
- Loss con parámetros actuales
- Gap entre train y test (indicador de overfitting)
  - ✓ Muy bueno: gap < 0
  - ✓ Normal: 0 ≤ gap ≤ 0.5
  - 📈 Overfitting: gap > 0.5

## 🎨 Características Adicionales

### Barras de Validación Silenciosas
Las barras de test se muestran pero se limpian automáticamente sin contaminar la salida.

### Estimación de Tiempo
- Tiempo restante se actualiza en tiempo real
- Te da idea de cuándo terminará el entrenamiento

### Monitoreo de Hiperparámetros
- Puedes ver cómo cambia el learning rate
- Útil para debuggear problemas de convergencia

## 💡 Cómo Interpretar los Datos

### Loss (bits/dim)
- Métrica estándar en normalizing flows
- Más bajo es mejor
- Típicamente debería decrecer en las primeras épocas

### Learning Rate (lr)
- Comienza en el valor configurado (e.g., 0.001)
- Decrece según el scheduler cada época
- Si es muy alto → inestabilidad, si es muy bajo → convergencia lenta

### Gap Train-Test
- **Negativo o cercano a 0**: Modelo generaliza bien
- **0 a 0.5**: Normal, algo de overfitting
- **Mayor a 0.5**: Posible overfitting severo

## 🚀 Mejoras Implementadas

1. ✅ Barra de progreso con tqdm
2. ✅ Actualización en tiempo real del loss
3. ✅ Visualización del learning rate
4. ✅ Resumen elegante de épocas
5. ✅ Barras de test sin contaminar output
6. ✅ Indicadores visuales para diagnóstico

Ahora puedes monitorear el entrenamiento cómodamente sin necesidad de parsear logs!
