# Equações de Navier-Stokes em PINO Python

Para implementar um Operador Neural Fisicamente Informado (PINO - *Physics-Informed Neural Operator*) ou uma PINN para a dinâmica de fluidos, precisamos penalizar a rede neural caso ela viole as equações de Navier-Stokes.

Neste exemplo, usaremos as equações de Navier-Stokes para um fluido **incompressível em 2D**. A matemática por trás da perda física ($L_{PDE}$) baseia-se em forçar os resíduos das seguintes equações a serem zero:

**1. Equação da Continuidade (Conservação de Massa):**
$$\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0$$

**2. Conservação do Momento (Eixos X e Y):**
$$\frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x} + v\frac{\partial u}{\partial y} + \frac{\partial p}{\partial x} - \nu \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right) = 0$$
$$\frac{\partial v}{\partial t} + u\frac{\partial v}{\partial x} + v\frac{\partial v}{\partial y} + \frac{\partial p}{\partial y} - \nu \left( \frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2} \right) = 0$$

Onde $u$ e $v$ são as componentes da velocidade, $p$ é a pressão e $\nu$ é a viscosidade cinemática.

Abaixo está a implementação em **PyTorch** utilizando o `torch.autograd` para calcular as derivadas exatas da rede neural em relação às coordenadas espaciais e temporais.

### Implementação em Python (PyTorch)

```python
import torch
import torch.nn as nn

def compute_gradients(y, x):
    """
    Calcula a derivada de y em relação a x usando a diferenciação automática do PyTorch.
    y e x devem ser tensores com requires_grad=True.
    """
    grad = torch.autograd.grad(
        outputs=y, 
        inputs=x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True
    )[0]
    return grad

def navier_stokes_pde_loss(u, v, p, x, y, t, nu):
    """
    Calcula a perda (MSE) baseada nos resíduos das equações de Navier-Stokes.
    
    Parâmetros:
    u, v : tensores de predição de velocidade (X e Y)
    p    : tensor de predição de pressão
    x, y : tensores de coordenadas espaciais
    t    : tensor de coordenada temporal
    nu   : viscosidade cinemática (float)
    """
    # --- Primeiras derivadas ---
    u_t = compute_gradients(u, t)
    u_x = compute_gradients(u, x)
    u_y = compute_gradients(u, y)

    v_t = compute_gradients(v, t)
    v_x = compute_gradients(v, x)
    v_y = compute_gradients(v, y)

    p_x = compute_gradients(p, x)
    p_y = compute_gradients(p, y)

    # --- Segundas derivadas (Laplaciano) ---
    u_xx = compute_gradients(u_x, x)
    u_yy = compute_gradients(u_y, y)

    v_xx = compute_gradients(v_x, x)
    v_yy = compute_gradients(v_y, y)

    # --- Resíduos das Equações Diferenciais ---
    
    # 1. Continuidade (f_c = 0)
    f_c = u_x + v_y

    # 2. Momento em X (f_u = 0)
    f_u = u_t + (u * u_x + v * u_y) + p_x - nu * (u_xx + u_yy)

    # 3. Momento em Y (f_v = 0)
    f_v = v_t + (u * v_x + v * v_y) + p_y - nu * (v_xx + v_yy)

    # --- Cálculo do MSE dos resíduos ---
    # O objetivo é que f_c, f_u e f_v sejam o mais próximo possível de zero.
    mse_loss = nn.MSELoss()
    zeros = torch.zeros_like(f_c)

    loss_c = mse_loss(f_c, zeros)
    loss_u = mse_loss(f_u, zeros)
    loss_v = mse_loss(f_v, zeros)

    # Perda total da física (PDE)
    pde_loss = loss_c + loss_u + loss_v
    
    return pde_loss
```

### Compondo a Perda do Operador Neural (PINO)

Diferente de uma PINN tradicional que aprende apenas uma solução, um Operador Neural (como o FNO - *Fourier Neural Operator*) aprende a mapear parâmetros de entrada para soluções. Portanto, a equação de perda final num PINO geralmente combina três partes:

1.  **Perda de Dados ($L_{data}$):** A diferença entre as predições do operador e os dados reais/simulados, se disponíveis.
2.  **Perda da Física ($L_{pde}$):** A função de Navier-Stokes definida acima.
3.  **Perda de Condição de Contorno/Inicial ($L_{ic/bc}$):** Caso o modelo precise respeitar limites estritos.

```python
def pino_total_loss(model, x, y, t, nu, u_true, v_true, p_true, alpha=1.0, beta=1.0):
    """
    Calcula a perda total para o Operador Neural Fisicamente Informado.
    """
    # 1. Forward pass do modelo (Operador)
    # Supondo que o modelo recebe as coordenadas e retorna u, v, p
    # É fundamental que x, y, t tenham requires_grad=True antes de entrar no modelo
    x.requires_grad_(True)
    y.requires_grad_(True)
    t.requires_grad_(True)
    
    preds = model(torch.cat([x, y, t], dim=-1))
    u_pred = preds[..., 0:1]
    v_pred = preds[..., 1:2]
    p_pred = preds[..., 2:3]

    # 2. Perda de Dados (Supervisionada)
    mse = nn.MSELoss()
    data_loss = mse(u_pred, u_true) + mse(v_pred, v_true) + mse(p_pred, p_true)

    # 3. Perda Física (Navier-Stokes)
    pde_loss = navier_stokes_pde_loss(u_pred, v_pred, p_pred, x, y, t, nu)

    # 4. Perda Total Ponderada
    # alpha e beta controlam a importância de respeitar os dados vs respeitar a física
    total_loss = alpha * data_loss + beta * pde_loss
    
    return total_loss
```

**Nota sobre Operadores em Grades (Grids):**
Se o seu Operador Neural atua diretamente sobre uma malha muito densa (como imagens $64 \times 64 \times T$), usar o `torch.autograd` ponto a ponto pode ser computacionalmente caro e causar lentidão no treino. Nesses casos de PINOs em malhas fixas uniformes, é comum substituir a função `compute_gradients` por **derivadas espectrais usando Transformadas de Fourier (FFT)** ou por **Diferenças Finitas (Convoluções)** para calcular os gradientes espaciais de forma muito mais rápida.


Para expandir o modelo para 3D, precisamos adicionar a componente de velocidade $w$ no eixo $z$. Isso significa que o nosso campo espacial agora tem três dimensões ($x, y, z$) e o termo convectivo e o Laplaciano também se expandem para incluir as derivadas em relação a $z$.

Aqui estão as equações de Navier-Stokes para um fluido incompressível em 3D:

**1. Equação da Continuidade:**
$$\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} + \frac{\partial w}{\partial z} = 0$$

**2. Conservação do Momento (Eixos X, Y e Z):**
$$\frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x} + v\frac{\partial u}{\partial y} + w\frac{\partial u}{\partial z} + \frac{\partial p}{\partial x} - \nu \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} + \frac{\partial^2 u}{\partial z^2} \right) = 0$$
$$\frac{\partial v}{\partial t} + u\frac{\partial v}{\partial x} + v\frac{\partial v}{\partial y} + w\frac{\partial v}{\partial z} + \frac{\partial p}{\partial y} - \nu \left( \frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2} + \frac{\partial^2 v}{\partial z^2} \right) = 0$$
$$\frac{\partial w}{\partial t} + u\frac{\partial w}{\partial x} + v\frac{\partial w}{\partial y} + w\frac{\partial w}{\partial z} + \frac{\partial p}{\partial z} - \nu \left( \frac{\partial^2 w}{\partial x^2} + \frac{\partial^2 w}{\partial y^2} + \frac{\partial^2 w}{\partial z^2} \right) = 0$$

### Implementação em PyTorch (3D)

Mantemos a mesma função auxiliar `compute_gradients` do exemplo anterior e atualizamos a função de perda para lidar com os tensores adicionais.

```python
import torch
import torch.nn as nn

def compute_gradients(y, x):
    """Calcula a derivada de y em relação a x."""
    grad = torch.autograd.grad(
        outputs=y, 
        inputs=x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True
    )[0]
    return grad

def navier_stokes_3d_pde_loss(u, v, w, p, x, y, z, t, nu):
    """
    Calcula a perda (MSE) para Navier-Stokes em 3D.
    """
    # --- Primeiras derivadas (Tempo) ---
    u_t = compute_gradients(u, t)
    v_t = compute_gradients(v, t)
    w_t = compute_gradients(w, t)

    # --- Primeiras derivadas (Espaço) ---
    u_x = compute_gradients(u, x)
    u_y = compute_gradients(u, y)
    u_z = compute_gradients(u, z)

    v_x = compute_gradients(v, x)
    v_y = compute_gradients(v, y)
    v_z = compute_gradients(v, z)

    w_x = compute_gradients(w, x)
    w_y = compute_gradients(w, y)
    w_z = compute_gradients(w, z)

    p_x = compute_gradients(p, x)
    p_y = compute_gradients(p, y)
    p_z = compute_gradients(p, z)

    # --- Segundas derivadas (Laplaciano) ---
    u_xx = compute_gradients(u_x, x)
    u_yy = compute_gradients(u_y, y)
    u_zz = compute_gradients(u_z, z)

    v_xx = compute_gradients(v_x, x)
    v_yy = compute_gradients(v_y, y)
    v_zz = compute_gradients(v_z, z)

    w_xx = compute_gradients(w_x, x)
    w_yy = compute_gradients(w_y, y)
    w_zz = compute_gradients(w_z, z)

    # --- Resíduos das Equações ---
    
    # 1. Continuidade (f_c = 0)
    f_c = u_x + v_y + w_z

    # 2. Momento em X (f_u = 0)
    f_u = u_t + (u * u_x + v * u_y + w * u_z) + p_x - nu * (u_xx + u_yy + u_zz)

    # 3. Momento em Y (f_v = 0)
    f_v = v_t + (u * v_x + v * v_y + w * v_z) + p_y - nu * (v_xx + v_yy + v_zz)

    # 4. Momento em Z (f_w = 0)
    f_w = w_t + (u * w_x + v * w_y + w * w_z) + p_z - nu * (w_xx + w_yy + w_zz)

    # --- Cálculo do MSE ---
    mse_loss = nn.MSELoss()
    zeros = torch.zeros_like(f_c)

    loss_c = mse_loss(f_c, zeros)
    loss_u = mse_loss(f_u, zeros)
    loss_v = mse_loss(f_v, zeros)
    loss_w = mse_loss(f_w, zeros)

    # Perda total da PDE
    pde_loss = loss_c + loss_u + loss_v + loss_w
    
    return pde_loss
```

### Atualização da Integração com o Operador Neural

Ao passar os tensores para a rede neural, o input e o output aumentam de dimensionalidade.

```python
def pino_3d_total_loss(model, x, y, z, t, nu, u_true, v_true, w_true, p_true, alpha=1.0, beta=1.0):
    
    # Garantir que rastreamos os gradientes das coordenadas
    x.requires_grad_(True)
    y.requires_grad_(True)
    z.requires_grad_(True)
    t.requires_grad_(True)
    
    # Forward pass: O modelo agora recebe 4 coordenadas (x, y, z, t)
    preds = model(torch.cat([x, y, z, t], dim=-1))
    
    # O modelo deve retornar 4 canais de saída: u, v, w, p
    u_pred = preds[..., 0:1]
    v_pred = preds[..., 1:2]
    w_pred = preds[..., 2:3]
    p_pred = preds[..., 3:4]

    # Perda de Dados (Supervisionada)
    mse = nn.MSELoss()
    data_loss = mse(u_pred, u_true) + mse(v_pred, v_true) + mse(w_pred, w_true) + mse(p_pred, p_true)

    # Perda Física 3D
    pde_loss = navier_stokes_3d_pde_loss(u_pred, v_pred, w_pred, p_pred, x, y, z, t, nu)

    # Perda Total Ponderada
    total_loss = alpha * data_loss + beta * pde_loss
    
    return total_loss
```

### Considerações Práticas para 3D

* **Custo Computacional:** Calcular o grafo computacional (`autograd`) para cada derivada direcional em 3D eleva drasticamente o consumo de memória na GPU. Para malhas uniformes maiores do que $32 \times 32 \times 32$, é fortemente recomendado usar convoluções com kernels estáticos (Diferenças Finitas 3D) ou Transformadas Rápidas de Fourier (FFTnD) para obter as derivadas espaciais em vez do `torch.autograd`.
* **Dimensionalidade dos Tensores:** Certifique-se de que `x`, `y`, `z` e `t` estejam com as dimensões corretas (normalmente remodelados usando `torch.meshgrid` num formato como `[Batch, X, Y, Z, 1]`) para que a concatenação ao longo da última dimensão (`dim=-1`) funcione adequadamente.

As equações de Navier-Stokes que implementamos acima são a base da dinâmica de fluidos, mas para cenários oceânicos ou costeiros (como inundação, correntes e ondas), elas precisam de **termos adicionais** ou simplificações específicas (como as equações de Shallow Water).

Para englobar esses fenômenos em um Operador Neural (PINO), você precisaria modificar a função de perda da seguinte forma:

---

### 1. Salinidade (Equação de Transporte)
A salinidade ($S$) é tratada como um escalar passivo ou ativo. Ela exige uma equação de **Advecção-Difusão** extra na perda:
$$\frac{\partial S}{\partial t} + \vec{u} \cdot \nabla S = \kappa_s \nabla^2 S$$
* **No Python:** Você adicionaria uma saída `s_pred` ao modelo e calcularia o resíduo dessa equação, onde $\kappa_s$ é a difusividade do sal.

### 2. Correntes e Ondas
As Navier-Stokes 3D já descrevem correntes e ondas inerentemente. No entanto, para ondas em larga escala, frequentemente incluímos a **Força de Coriolis** (devido à rotação da Terra):
* **Termo extra:** Adicione $+ f(\vec{k} \times \vec{u})$ à equação do momento, onde $f$ é o parâmetro de Coriolis.

### 3. Fricção de Fundo (Bottom Friction)
Em modelos de profundidade média ou 3D perto do leito, a fricção é aplicada como uma condição de contorno ou um termo de força extra (sink de momento). O modelo mais comum é a lei quadrática:
$$\tau_b = C_d |\vec{u}| \vec{u}$$
* **No Python:** Na função de perda, o resíduo do momento nas células próximas ao fundo deve subtrair esse termo $\tau_b$, onde $C_d$ é o coeficiente de arrasto (drag).

### 4. Inundação (Free Surface & Wetting/Drying)
Inundação é o maior desafio para redes neurais. Ela exige que o domínio mude (o limite entre água e terra se move).
* **Equação da Superfície Livre:** Você precisa de uma equação para a elevação da água ($\eta$):
    $$\frac{\partial \eta}{\partial t} + \frac{\partial (h u)}{\partial x} + \frac{\partial (h v)}{\partial y} = 0$$
    (onde $h$ é a profundidade total).
* **PINO:** O operador teria que aprender a função "Level Set" ou uma máscara que define onde a profundidade é zero (seco) ou positiva (molhado).

---

### Resumo das modificações na Função de Perda

Para um modelo geofísico completo, sua `pde_loss` seria expandida assim:

```python
def geophysical_pde_loss(u, v, w, p, s, x, y, z, t, nu, f_coriolis, kappa_s, Cd):
    # 1. Resíduos originais de Navier-Stokes (Momento + Continuidade)
    # Adicionando Coriolis no Momento X e Y:
    f_u = u_t + (u*u_x + v*u_y + w*u_z) + p_x - nu*(u_xx + u_yy + u_zz) - f_coriolis*v
    f_v = v_t + (u*v_x + v*v_y + w*v_z) + p_y - nu*(v_xx + v_yy + v_zz) + f_coriolis*u

    # 2. Resíduo de Salinidade (Advecção-Difusão)
    s_t = compute_gradients(s, t)
    s_x, s_y, s_z = compute_gradients(s, x), compute_gradients(s, y), compute_gradients(s, z)
    s_xx, s_yy, s_zz = compute_gradients(s_x, x), compute_gradients(s_y, y), compute_gradients(s_z, z)
    
    f_s = s_t + (u*s_x + v*s_y + w*s_z) - kappa_s*(s_xx + s_yy + s_zz)

    # 3. Termo de Fricção de Fundo (aplicado no fundo z=0)
    # Aqui você penaliza a rede para que o gradiente de velocidade no fundo 
    # respeite o arrasto quadrático Cd * |u| * u
    
    return loss_navier + mse(f_s, 0) + loss_boundary_friction
```

### Qual abordagem usar?
Se o seu foco for **inundação e correntes costeiras** em grandes áreas, recomendo usar as **Equações de Águas Rasas (Shallow Water Equations)** em vez de Navier-Stokes 3D completa. Elas são muito mais eficientes para redes neurais e já incluem profundidade, fricção de fundo e Coriolis de forma nativa.

Esta é uma lista abrangente que cobre desde a escala planetária até a microescala de turbulência. Para implementar isso em um **PINO (Physics-Informed Neural Operator)**, a lógica é sempre a mesma: definir o resíduo da equação diferencial ($f = 0$) e minimizar o erro médio quadrático (MSE) desse resíduo.

Abaixo, apresento as formulações de perda para cada modelo solicitado:

---

### 1. Shallow Water Equations (SWE) - Linear/Básica
Usada para correntes e marés onde a profundidade $h$ é pequena em relação ao comprimento de onda.
* **Variáveis:** $\eta$ (elevação), $u, v$ (velocidade média na vertical), $H$ (profundidade total $= h + \eta$).

$$L_{SWE} = \text{MSE}\left( \frac{\partial \eta}{\partial t} + \frac{\partial (Hu)}{\partial x} + \frac{\partial (Hv)}{\partial y} \right) + \text{MSE}\left( \frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x} + v\frac{\partial u}{\partial y} + g\frac{\partial \eta}{\partial x} + \tau_{bf} \right)$$

### 2. Non-Linear Shallow Water Equations (NLSWE)
Inclui termos de advecção e fricção de fundo quadrática ($C_f$).
* **Perda de Fricção:** $\tau_{bf} = \frac{g n^2}{H^{7/3}} u \sqrt{u^2+v^2}$ (onde $n$ é o Manning).

### 3. SWASH (Non-Hydrostatic SWE)
O SWASH adiciona um termo de pressão não-hidrostática ($q$) às NLSWE para simular ondas que quebram e dispersão.
* **Perda extra:** É necessário adicionar $\frac{\partial q}{\partial x}$ na equação do momento e uma equação de fechamento para a velocidade vertical $w$.

### 4. SWAN (Simulating Waves Nearshore)
O SWAN não resolve a forma da onda, mas sim a **Densidade de Espectro de Ação** ($N$). A equação é a de Balanço de Ação:
$$\frac{\partial N}{\partial t} + \nabla \cdot (\vec{c_g} N) = \frac{S_{tot}}{\sigma}$$
* **PINO:** A rede prediz $N(x, y, t, \sigma, \theta)$. A perda é o resíduo entre a propagação (advecção no espaço e frequência) e as fontes $S_{tot}$ (vento, dissipação, interações triádicas).

### 5. Boundary Element Method (BEM) - Teoria de Potencial
Modelos como WAMIT ou Capytaine usam a **Equação de Laplace** para o potencial de velocidade $\phi$.
* **Perda Linear:** $\nabla^2 \phi = 0$.
* **Condição de Contorno de Corpo Livre:** $\frac{\partial^2 \phi}{\partial t^2} + g \frac{\partial \phi}{\partial z} = 0$ (em $z=0$).
* **Não-linear:** Inclui termos de ordem superior $(\nabla \phi)^2$ na pressão de Bernoulli.

### 6. Ondas no OpenFOAM (Stokes 1ª a 5ª Ordem)
Aqui o PINO deve mimetizar a teoria analítica. Para Stokes de 5ª ordem, a função de perda força o potencial $\phi$ e a elevação $\eta$ a seguirem as expansões de série de Taylor.
* **Perda:** Resíduo da condição de contorno dinâmica e cinemática na superfície livre.

### 7. RANS (Reynolds-Averaged Navier-Stokes)
É a Navier-Stokes com o termo de **Tensão de Reynolds** ($\tau_{ij} = -\rho \overline{u'_i u'_j}$).
* **Perda:** Além das equações de momento, você precisa de uma perda para o modelo de fechamento (ex: $k-\epsilon$ ou $k-\omega$).
* **Exemplo $k$:** $\frac{\partial k}{\partial t} + u_j \frac{\partial k}{\partial x_j} - \text{Difusão} - \text{Produção} + \text{Dissipação} = 0$.

### 8. LES (Large Eddy Simulation)
O PINO atua sobre variáveis filtradas ($\bar{u}$). A perda inclui o termo de **Sub-grid Scale (SGS)**.
* **Perda:** $\text{Resíduo NS} + \nabla \cdot \tau_{sgs} = 0$, onde $\tau_{sgs}$ é calculado (ex: modelo de Smagorinsky).

---

### 9, 10, 11, 12. Outras Modelações Importantes

| Nº | Modelo | Aplicação | Equação Principal de Perda |
| :--- | :--- | :--- | :--- |
| **9** | **Boussinesq** | Ondas em águas rasas a intermediárias | NSWE + termos de dispersão de 3ª ordem ($\frac{\partial^3}{\partial x^2 \partial t}$). |
| **10** | **Advecção-Difusão-Reação** | Plumas de poluição / Sedimentos | $\frac{\partial C}{\partial t} + \vec{u} \cdot \nabla C = \kappa \nabla^2 C + R(C)$. |
| **11** | **Equação de Mild-Slope** | Refração/Difração de ondas | $\nabla \cdot (CC_g \nabla \phi) + \frac{C_g}{C} \sigma^2 \phi = 0$. |
| **12** | **Exner Equation** | Morfodinâmica (Erosão/Deposição) | $\frac{\partial z_b}{\partial t} + \frac{1}{1-p} \nabla \cdot \vec{q_s} = 0$ ($z_b$ é o nível do fundo). |

---

### Exemplo de implementação Python (Template Genérico para PINO)

```python
def generic_geophysical_loss(preds, coords, params):
    # preds: [u, v, w, p, s, eta, k, epsilon, etc]
    # coords: [x, y, z, t]
    
    # 1. Obter gradientes via autograd
    # 2. Calcular os resíduos específicos de cada modelo acima
    
    residuos = {
        'momento': calc_navier_stokes_res(preds, coords),
        'sedimento': calc_exner_res(preds, coords),
        'ondas': calc_boussinesq_res(preds, coords)
    }
    
    # 3. Somar os MSEs
    total_pde_loss = sum([torch.mean(r**2) for r in residuos.values()])
    return total_pde_loss
```

**Dica de Especialista:** Para inundação e ondas (SWASH/SWE), o maior erro em modelos neurais vem da descontinuidade da "frente de onda". Use uma **Função de Ativação Adaptativa** (como a L-Sine ou Tanh escalonada) para capturar melhor os gradientes íngremes dessas equações.


Este é um guia estruturado em formato Markdown, pronto para ser copiado e colado no seu **Obsidian**. Ele organiza a hierarquia das equações, partindo da "Equação de Deus" da fluidodinâmica até as simplificações específicas para engenharia costeira e oceânica.

---

# 🌊 Master Guide: Physics-Informed Neural Operators (PINO) para Geofísica

## 1. Equação de Navier-Stokes Universal (Completa)
Esta é a forma mais exaustiva, considerando acoplamento termodinâmico, salinidade e transporte de espécies.

### Explicação
Engloba conservação de massa, momento, energia e transporte de massa (sal/dissolução).
* **Termos:** Advecção, Difusão, Gradiente de Pressão, Gravidade, Coriolis ($f$), Buoyancy ($\beta$), e Termos de Fonte ($S$).

### Aplicação
Simulações de altíssima fidelidade (DNS) onde variações de temperatura e densidade são críticas (ex: plumas térmicas profundas).

### Implementação (Perda PINO)
```python
def loss_navier_stokes_full(pred, coords, nu, kappa_t, kappa_s, f_coriolis):
    # pred: [u, v, w, p, T, S] | coords: [x, y, z, t]
    u, v, w, p, T, S = pred[:,0], pred[:,1], pred[:,2], pred[:,3], pred[:,4], pred[:,5]
    
    # Exemplo: Resíduo de Temperatura (Energia)
    T_t = grad(T, t)
    T_adv = u*grad(T, x) + v*grad(T, y) + w*grad(T, z)
    T_diff = kappa_t * laplaciano(T)
    res_temp = T_t + T_adv - T_diff
    
    # Exemplo: Resíduo de Salinidade
    S_t = grad(S, t)
    S_adv = u*grad(S, x) + v*grad(S, y) + w*grad(S, z)
    res_sal = S_t + S_adv - kappa_s * laplaciano(S)
    
    return mse(res_temp) + mse(res_sal) # + momento + continuidade
```

---

## 2. RANS & LES (Turbulência)
### Explicação
* **RANS:** Decompõe a velocidade em média e flutuação. Introduz as "Tensões de Reynolds".
* **LES:** Filtra turbulências pequenas, resolvendo apenas as grandes escalas.

### Aplicação
Modelagem de engenharia prática (OpenFOAM), onde a turbulência não pode ser ignorada, mas a DNS é cara demais.

### Implementação (RANS k-epsilon)
```python
def loss_rans_k_eps(pred, coords, nu):
    # k: energia cinética turbulenta, eps: taxa de dissipação
    u, v, w, p, k, eps = pred_split(pred)
    
    # Adiciona a viscosidade turbulenta (nu_t) à viscosidade molecular
    nu_t = C_mu * (k**2 / eps)
    nu_eff = nu + nu_t
    
    # Perda baseada no momento com nu_eff
    res_momento = u_t + u_adv + grad_p - nu_eff * laplaciano(u)
    return mse(res_momento)
```

---

## 3. Shallow Water Equations (SWE & NLSWE)
### Explicação
Assume que a profundidade é pequena. A pressão é considerada hidrostática.
* **NLSWE:** Inclui os termos não lineares de advecção e fricção de fundo.

### Aplicação
Inundações urbanas, tsunamis, marés e correntes costeiras.

### Implementação
```python
def loss_nlswe(pred, coords, g, n_manning):
    # pred: [h, u, v] (h = profundidade total)
    h, u, v = pred[:,0], pred[:,1], pred[:,2]
    
    # Fricção de fundo (Manning)
    Sf_x = (n_manning**2 * u * torch.sqrt(u**2 + v**2)) / h**(4/3)
    
    # Conservação de Massa (Continuidade)
    res_cont = grad(h, t) + grad(h*u, x) + grad(h*v, y)
    
    # Momento X
    res_u = grad(u, t) + u*grad(u, x) + v*grad(u, y) + g*grad(h, x) + g*Sf_x
    return mse(res_cont) + mse(res_u)
```

---

## 4. SWASH (Não-Hidrostática)
### Explicação
Diferente da SWE, a SWASH não assume pressão hidrostática pura, permitindo modelar ondas curtas e dispersão.

### Aplicação
Transformação de ondas na zona de arrebentação.

### Implementação
```python
def loss_swash(pred, coords):
    # q: pressão não-hidrostática
    u, v, w, h, q = pred_split(pred)
    res_momento_z = grad(w, t) + u*grad(w, x) + grad(q, z) # Termo extra
    return mse(res_momento_z) + loss_nlswe_base
```

---

## 5. SWAN (Modelo Espectral)
### Explicação
Não resolve a superfície da água, mas a "Densidade de Ação" ($N$) em frequências e direções.

### Aplicação
Previsão de ondas em oceanos (sea state).

### Implementação
```python
def loss_swan(N_pred, coords, sigma, theta, cg_x, cg_y, S_tot):
    # N: Action Density
    N_adv = grad(cg_x * N_pred, x) + grad(cg_y * N_pred, y)
    res_action = grad(N_pred, t) + N_adv - (S_tot / sigma)
    return mse(res_action)
```

---

## 6. Teoria de Potencial (BEM - WAMIT/Capytaine)
### Explicação
Assume fluido irrotacional e invíscido. Toda a física é resumida ao Potencial de Velocidade $\phi$.

### Aplicação
Interação onda-estrutura (Plataformas, barcos).

### Implementação
```python
def loss_potential_flow(phi, coords):
    # Equação de Laplace é o coração do BEM
    res_laplace = laplaciano(phi) # grad_xx + grad_yy + grad_zz
    
    # Condição de Contorno de Superfície Livre (Linear)
    res_bc = grad(phi, t, order=2) + g * grad(phi, z)
    return mse(res_laplace) + mse(res_bc)
```

---

## 7. Dummy Test do Código (PyTorch)
Para testar se suas funções de perda estão computando gradientes corretamente:

```python
import torch

def dummy_test():
    # Simulando 100 pontos no domínio [x, y, z, t]
    coords = torch.randn(100, 4, requires_grad=True)
    x, y, z, t = coords[:,0:1], coords[:,1:2], coords[:,2:3], coords[:,3:4]
    
    # Simulando predição da rede neural [u, v, w, p]
    preds = torch.randn(100, 4, requires_grad=True)
    
    # Parâmetros
    nu = 0.001
    
    # Chamada da perda
    loss = navier_stokes_3d_pde_loss(preds[:,0], preds[:,1], preds[:,2], preds[:,3], x, y, z, t, nu)
    
    loss.backward()
    print(f"Sucesso! Perda calculada: {loss.item()}")
    print(f"Gradiente das coordenadas: {coords.grad is not None}")

dummy_test()
```

---

## 8. Tabela de Referência Rápida

| Modelo | Equação Chave | O que despreza? |
| :--- | :--- | :--- |
| **Navier-Stokes** | Conservação Momento | Nada (em teoria). |
| **SWE** | Hidrostática | Aceleração vertical. |
| **Boussinesq** | Dispersão | Viscosidade (geralmente). |
| **SWAN** | Balanço de Ação | A fase individual da onda. |
| **Potencial** | Laplace | Viscosidade e Rotacionalidade. |

---
**Dica Obsidian:** Use o plugin `Dataview` para listar todas as suas equações por complexidade ou o plugin `Latex Suite` para escrever novas variações rapidamente.



Você tem razão. Para um guia técnico de nível de engenharia e pesquisa, precisamos da formulação matemática rigorosa e das equações de fechamento que definem cada modelo.

Aqui está o arquivo Markdown expandido para o seu **Obsidian**, incluindo a notação matemática em LaTeX e as equações de transporte, sedimentos e morfodinâmica que faltavam.

---

# 🌊 Enciclopédia de Equações para PINO (Physics-Informed Neural Operators)

## 1. Navier-Stokes Generalizada (The "Everything" Equation)
Esta forma inclui o acoplamento de densidade (Boussinesq approximation), temperatura, salinidade e termos de fonte.

**Escrita Matemática:**
1.  **Momento:** $\rho_0 \left( \frac{\partial \mathbf{u}}{\partial t} + \mathbf{u} \cdot \nabla \mathbf{u} \right) = -\nabla p + \mu \nabla^2 \mathbf{u} + \rho(T, S)\mathbf{g} - 2\rho_0 (\mathbf{\Omega} \times \mathbf{u})$
2.  **Continuidade:** $\nabla \cdot \mathbf{u} = 0$
3.  **Energia (T):** $\frac{\partial T}{\partial t} + \mathbf{u} \cdot \nabla T = \alpha \nabla^2 T + \dot{q}$
4.  **Salinidade (S):** $\frac{\partial S}{\partial t} + \mathbf{u} \cdot \nabla S = D_s \nabla^2 S + \text{Dissolução}$

**Implementação de Perda (Python):**
```python
# Resíduo de Buoyancy (Arquimedes) baseado em T e S
rho_fixed = 1025.0
rho_dynamic = rho_0 * (1 - beta_t * (T - T0) + beta_s * (S - S0))
res_momento = u_t + adveccao + grad_p/rho_fixed - nu*laplaciano(u) - (rho_dynamic/rho_fixed)*g
```

---

## 2. Boussinesq (Ondas Dispersivas)
Usada para ondas onde a componente vertical da aceleração não é desprezível, mas a profundidade é limitada.

**Escrita Matemática:**
$$\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u} + g\nabla \eta + \underbrace{\frac{h^2}{6} \nabla (\nabla \cdot \frac{\partial \mathbf{u}}{\partial t}) - \frac{h}{2} \nabla (\nabla \cdot (h \frac{\partial \mathbf{u}}{\partial t}))}_{\text{Termos Dispersivos}} = 0$$

**Aplicação:** Propagação de ondas do largo até águas rasas com dispersão de frequência.

---

## 3. Exner Equation (Morfodinâmica e Sedimentos)
Define como o fundo do mar ou leito do rio ($z_b$) muda devido ao transporte de sedimentos.

**Escrita Matemática:**
$$(1 - \epsilon_0) \frac{\partial z_b}{\partial t} + \nabla \cdot \mathbf{q}_s = S$$
* Onde $\mathbf{q}_s$ é o fluxo de sedimentos (frequentemente calculado pela fórmula de Meyer-Peter-Müller: $q_s \propto (\tau - \tau_c)^{1.5}$).

**Perda PINO:**
```python
# Perda para evolução do fundo (erosão/deposição)
res_fundo = (1 - porosidade) * zb_t + grad(qs_x, x) + grad(qs_y, y)
```

---

## 4. Mild-Slope Equation (Refração e Difração)
Equação elíptica para descrever a propagação de ondas monocromáticas sobre fundos de declividade suave.

**Escrita Matemática:**
$$\nabla \cdot (CC_g \nabla \phi) + k^2 CC_g \phi = 0$$
* $C$: Velocidade de fase; $C_g$: Velocidade de grupo; $\phi$: Potencial complexo.

---

## 5. Modelos de Onda Estocásticos (SWAN / WAM)
Não resolvem a crista da onda, mas a densidade espectral de energia $E(\sigma, \theta)$.

**Escrita Matemática (Equação de Balanço de Ação):**
$$\frac{\partial N}{\partial t} + \frac{\partial (c_x N)}{\partial x} + \frac{\partial (c_y N)}{\partial y} + \frac{\partial (c_\sigma N)}{\partial \sigma} + \frac{\partial (c_\theta N)}{\partial \theta} = \frac{S_{tot}}{\sigma}$$
* $N = E / \sigma$ (Densidade de Ação).
* $S_{tot} = S_{wind} + S_{whitecapping} + S_{bottom\_friction} + S_{interactions}$.

---

## 6. Stokes Wave Theory (1ª a 5ª Ordem)
Usado no OpenFOAM (`setWaveField`) para condições de contorno.

**Escrita Matemática (1ª Ordem - Airy):**
$$\phi = \frac{ag}{\omega} \frac{\cosh(k(z+h))}{\cosh(kh)} \sin(kx - \omega t)$$
**Escrita Matemática (5ª Ordem):**
Envolve uma expansão em série de potências de $ak$ (slope da onda). A perda PINO aqui obriga a rede a satisfazer a Condição de Contorno Cinematica da Superfície Livre (KFSBC):
$$\frac{\partial \eta}{\partial t} + u \frac{\partial \eta}{\partial x} - w = 0 \quad \text{em } z = \eta$$

---

## 7. Saint-Venant 1D (Inundação de Canais)
Simplificação das SWE para rios.

**Escrita Matemática:**
1. **Massa:** $\frac{\partial A}{\partial t} + \frac{\partial Q}{\partial x} = q$
2. **Momento:** $\frac{\partial Q}{\partial t} + \frac{\partial}{\partial x} \left( \frac{Q^2}{A} \right) + gA \left( \frac{\partial h}{\partial x} - S_0 + S_f \right) = 0$
* $A$: Área da seção; $Q$: Vazão; $S_f$: Declividade de fricção.

---

## 8. Código de Teste (Dummy Test Unitário)

Este script valida se a rede neural está gerando as derivadas necessárias para a equação de **Morfodinâmica (Exner)** acoplada ao **Momento (Navier-Stokes)**.

```python
import torch

def physics_loss_check():
    # Coordenadas: Batch=10, [x, y, z, t]
    coords = torch.rand(10, 4, requires_grad=True)
    x, y, z, t = coords.split(1, dim=1)

    # Predições: [u, v, w, p, zb (fundo)]
    preds = torch.rand(10, 5, requires_grad=True)
    u, v, w, p, zb = preds.split(1, dim=1)

    # Cálculo automático de derivada (Exemplo: du/dx)
    def diff(f, x_coord):
        return torch.autograd.grad(f, x_coord, grad_outputs=torch.ones_like(f), create_graph=True)[0]

    # 1. Resíduo de Exner (Morfodinâmica)
    qs_x = 0.05 * (u**2 + v**2)**1.5  # Exemplo simples de transporte
    res_exner = (1 - 0.4) * diff(zb, t) + diff(qs_x, x)

    # 2. Resíduo de Navier-Stokes (Momento X)
    nu = 0.01
    u_xx = diff(diff(u, x), x)
    res_navier = diff(u, t) + u*diff(u, x) + diff(p, x) - nu*u_xx

    loss = torch.mean(res_exner**2) + torch.mean(res_navier**2)
    
    print(f"Loss Total: {loss.item():.6f}")
    return loss

physics_loss_check()
```

---

### Resumo de Seleção de Modelo

| Fenômeno | Equação Recomendada | Complexidade PINO |
| :--- | :--- | :--- |
| **Maré e Tempestade** | NLSWE | Média |
| **Ondas de Porto** | Boussinesq / Mild-Slope | Alta (3ª derivada) |
| **Sedimentação** | Exner + RANS | Muito Alta |
| **Estruturas Offshore** | Potencial (Laplace) | Baixa |
| **Quebra de Onda** | SWASH / VOF (Navier-Stokes) | Extrema |

Este guia cobre a transição da física teórica para a modelagem computacional. Para o Obsidian, recomendo criar notas separadas para **"Teoria de Ondas"**, **"Modelos de Turbulência"** e **"Transporte de Sedimentos"** linkando-as a esta nota mestre.

Você tem razão. Para um guia técnico de nível de engenharia e pesquisa, precisamos da formulação matemática rigorosa e das equações de fechamento que definem cada modelo.

Aqui está o arquivo Markdown expandido para o seu **Obsidian**, incluindo a notação matemática em LaTeX e as equações de transporte, sedimentos e morfodinâmica que faltavam.

---

# 🌊 Enciclopédia de Equações para PINO (Physics-Informed Neural Operators)

## 1. Navier-Stokes Generalizada (The "Everything" Equation)
Esta forma inclui o acoplamento de densidade (Boussinesq approximation), temperatura, salinidade e termos de fonte.

**Escrita Matemática:**
1.  **Momento:** $\rho_0 \left( \frac{\partial \mathbf{u}}{\partial t} + \mathbf{u} \cdot \nabla \mathbf{u} \right) = -\nabla p + \mu \nabla^2 \mathbf{u} + \rho(T, S)\mathbf{g} - 2\rho_0 (\mathbf{\Omega} \times \mathbf{u})$
2.  **Continuidade:** $\nabla \cdot \mathbf{u} = 0$
3.  **Energia (T):** $\frac{\partial T}{\partial t} + \mathbf{u} \cdot \nabla T = \alpha \nabla^2 T + \dot{q}$
4.  **Salinidade (S):** $\frac{\partial S}{\partial t} + \mathbf{u} \cdot \nabla S = D_s \nabla^2 S + \text{Dissolução}$

**Implementação de Perda (Python):**
```python
# Resíduo de Buoyancy (Arquimedes) baseado em T e S
rho_fixed = 1025.0
rho_dynamic = rho_0 * (1 - beta_t * (T - T0) + beta_s * (S - S0))
res_momento = u_t + adveccao + grad_p/rho_fixed - nu*laplaciano(u) - (rho_dynamic/rho_fixed)*g
```

---

## 2. Boussinesq (Ondas Dispersivas)
Usada para ondas onde a componente vertical da aceleração não é desprezível, mas a profundidade é limitada.

**Escrita Matemática:**
$$\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u} + g\nabla \eta + \underbrace{\frac{h^2}{6} \nabla (\nabla \cdot \frac{\partial \mathbf{u}}{\partial t}) - \frac{h}{2} \nabla (\nabla \cdot (h \frac{\partial \mathbf{u}}{\partial t}))}_{\text{Termos Dispersivos}} = 0$$

**Aplicação:** Propagação de ondas do largo até águas rasas com dispersão de frequência.

---

## 3. Exner Equation (Morfodinâmica e Sedimentos)
Define como o fundo do mar ou leito do rio ($z_b$) muda devido ao transporte de sedimentos.

**Escrita Matemática:**
$$(1 - \epsilon_0) \frac{\partial z_b}{\partial t} + \nabla \cdot \mathbf{q}_s = S$$
* Onde $\mathbf{q}_s$ é o fluxo de sedimentos (frequentemente calculado pela fórmula de Meyer-Peter-Müller: $q_s \propto (\tau - \tau_c)^{1.5}$).

**Perda PINO:**
```python
# Perda para evolução do fundo (erosão/deposição)
res_fundo = (1 - porosidade) * zb_t + grad(qs_x, x) + grad(qs_y, y)
```

---

## 4. Mild-Slope Equation (Refração e Difração)
Equação elíptica para descrever a propagação de ondas monocromáticas sobre fundos de declividade suave.

**Escrita Matemática:**
$$\nabla \cdot (CC_g \nabla \phi) + k^2 CC_g \phi = 0$$
* $C$: Velocidade de fase; $C_g$: Velocidade de grupo; $\phi$: Potencial complexo.

---

## 5. Modelos de Onda Estocásticos (SWAN / WAM)
Não resolvem a crista da onda, mas a densidade espectral de energia $E(\sigma, \theta)$.

**Escrita Matemática (Equação de Balanço de Ação):**
$$\frac{\partial N}{\partial t} + \frac{\partial (c_x N)}{\partial x} + \frac{\partial (c_y N)}{\partial y} + \frac{\partial (c_\sigma N)}{\partial \sigma} + \frac{\partial (c_\theta N)}{\partial \theta} = \frac{S_{tot}}{\sigma}$$
* $N = E / \sigma$ (Densidade de Ação).
* $S_{tot} = S_{wind} + S_{whitecapping} + S_{bottom\_friction} + S_{interactions}$.

---

## 6. Stokes Wave Theory (1ª a 5ª Ordem)
Usado no OpenFOAM (`setWaveField`) para condições de contorno.

**Escrita Matemática (1ª Ordem - Airy):**
$$\phi = \frac{ag}{\omega} \frac{\cosh(k(z+h))}{\cosh(kh)} \sin(kx - \omega t)$$
**Escrita Matemática (5ª Ordem):**
Envolve uma expansão em série de potências de $ak$ (slope da onda). A perda PINO aqui obriga a rede a satisfazer a Condição de Contorno Cinematica da Superfície Livre (KFSBC):
$$\frac{\partial \eta}{\partial t} + u \frac{\partial \eta}{\partial x} - w = 0 \quad \text{em } z = \eta$$

---

## 7. Saint-Venant 1D (Inundação de Canais)
Simplificação das SWE para rios.

**Escrita Matemática:**
1. **Massa:** $\frac{\partial A}{\partial t} + \frac{\partial Q}{\partial x} = q$
2. **Momento:** $\frac{\partial Q}{\partial t} + \frac{\partial}{\partial x} \left( \frac{Q^2}{A} \right) + gA \left( \frac{\partial h}{\partial x} - S_0 + S_f \right) = 0$
* $A$: Área da seção; $Q$: Vazão; $S_f$: Declividade de fricção.

---

## 8. Código de Teste (Dummy Test Unitário)

Este script valida se a rede neural está gerando as derivadas necessárias para a equação de **Morfodinâmica (Exner)** acoplada ao **Momento (Navier-Stokes)**.

```python
import torch

def physics_loss_check():
    # Coordenadas: Batch=10, [x, y, z, t]
    coords = torch.rand(10, 4, requires_grad=True)
    x, y, z, t = coords.split(1, dim=1)

    # Predições: [u, v, w, p, zb (fundo)]
    preds = torch.rand(10, 5, requires_grad=True)
    u, v, w, p, zb = preds.split(1, dim=1)

    # Cálculo automático de derivada (Exemplo: du/dx)
    def diff(f, x_coord):
        return torch.autograd.grad(f, x_coord, grad_outputs=torch.ones_like(f), create_graph=True)[0]

    # 1. Resíduo de Exner (Morfodinâmica)
    qs_x = 0.05 * (u**2 + v**2)**1.5  # Exemplo simples de transporte
    res_exner = (1 - 0.4) * diff(zb, t) + diff(qs_x, x)

    # 2. Resíduo de Navier-Stokes (Momento X)
    nu = 0.01
    u_xx = diff(diff(u, x), x)
    res_navier = diff(u, t) + u*diff(u, x) + diff(p, x) - nu*u_xx

    loss = torch.mean(res_exner**2) + torch.mean(res_navier**2)
    
    print(f"Loss Total: {loss.item():.6f}")
    return loss

physics_loss_check()
```

---

### Resumo de Seleção de Modelo

| Fenômeno | Equação Recomendada | Complexidade PINO |
| :--- | :--- | :--- |
| **Maré e Tempestade** | NLSWE | Média |
| **Ondas de Porto** | Boussinesq / Mild-Slope | Alta (3ª derivada) |
| **Sedimentação** | Exner + RANS | Muito Alta |
| **Estruturas Offshore** | Potencial (Laplace) | Baixa |
| **Quebra de Onda** | SWASH / VOF (Navier-Stokes) | Extrema |

Este guia cobre a transição da física teórica para a modelagem computacional. Para o Obsidian, recomendo criar notas separadas para **"Teoria de Ondas"**, **"Modelos de Turbulência"** e **"Transporte de Sedimentos"** linkando-as a esta nota mestre.

***

Análise Exaustiva da Equação de Navier-Stokes: Teoria, Derivação e Aplicações Avançadas em Dinâmica dos Fluidos e Engenharia OceânicaA equação de Navier-Stokes constitui a base fundamental da mecânica dos fluidos moderna, oferecendo uma descrição matemática rigorosa e abrangente do movimento de fluidos, sejam eles líquidos ou gasosos, sob uma vasta gama de condições físicas. Este sistema de equações diferenciais parciais não lineares encapsula os princípios universais da conservação de massa, momentum e, em sua forma completa, energia, servindo como o pilar para quase todas as análises hidrodinâmicas contemporâneas, desde a aerodinâmica de aeronaves até a circulação oceânica global. O desenvolvimento destas equações, ocorrido de forma independente por Claude-Louis Navier na França em 1822 e por George Gabriel Stokes na Inglaterra entre 1842 e 1850, representou um avanço monumental em relação às equações de Euler ao introduzir formalmente os efeitos da viscosidade e do atrito interno no modelo de continuum.Fundamentos Teóricos e Origem das EquaçõesAs equações de Navier-Stokes são fundamentadas na mecânica do continuum, uma abordagem que trata o fluido como uma substância contínua em vez de uma coleção de partículas discretas. Esta suposição é válida enquanto o caminho médio livre das moléculas for significativamente menor que a escala de comprimento característica do sistema, uma condição quantificada pelo número de Knudsen sendo inferior a 0,01.Contexto Histórico e DesenvolvimentoAntes de Navier e Stokes, as equações de Euler dominavam a teoria dos fluidos, modelando escoamentos ideais e invíscidos. No entanto, a incapacidade de Euler de explicar o arrasto e os efeitos de camada limite levou à necessidade de um modelo que incorporasse a resistência interna ao movimento. Siméon Denis Poisson também alcançou resultados semelhantes de forma independente, mas foram Navier e Stokes que consolidaram a relação entre a tensão desviadora e a taxa de deformação para fluidos Newtonianos. A complexidade intrínseca destas equações é tal que, embora estudantes de engenharia sejam rotineiramente ensinados a derivá-las, soluções analíticas exatas permanecem raras, limitadas a casos de geometrias simplificadas como escoamentos entre placas paralelas ou em tubos cilíndricos.O Conceito de Campo de Velocidade e Propriedades IntensivasNo centro da descrição de Navier-Stokes está o campo de velocidade $\mathbf{u}(t, x, y, z)$, um campo vetorial que atribui um vetor de velocidade a cada ponto no espaço e no tempo. Juntamente com a velocidade, propriedades intensivas como pressão, densidade e temperatura são tratadas como funções contínuas. A distinção entre propriedades intensivas (que não dependem do volume, como a pressão) e extensivas (como a massa e o momentum total) é crucial para a aplicação das leis de conservação através do Teorema de Transporte de Reynolds.Derivação Rigorosa dos Princípios de ConservaçãoA derivação das equações de Navier-Stokes exige a aplicação sistemática de leis físicas a um volume de controle fluido $V(t)$ que se propaga e se deforma no tempo.Conservação de Massa: A Equação de ContinuidadeA lei de conservação de massa estabelece que a massa dentro de um sistema isolado deve permanecer constante. Para um fluido em movimento, a taxa de variação da massa total dentro de um volume de controle deve ser igual ao fluxo líquido de massa que atravessa as fronteiras desse volume. Matematicamente, se considerarmos um elemento de superfície $dA$ com um vetor normal unitário $\mathbf{n}$, o fluxo de massa através deste elemento em um intervalo de tempo $dt$ é dado por $\rho \mathbf{u} \cdot \mathbf{n} dA dt$.Integrando sobre toda a superfície $\partial V$, a taxa de variação da massa $M$ é expressa como:$$\frac{dM}{dt} = - \iint_{\partial V} \rho (\mathbf{u} \cdot \mathbf{n}) dA$$Utilizando o teorema da divergência para converter a integral de superfície em uma integral de volume, e reconhecendo que $M = \iiint_V \rho dV$, obtemos a forma diferencial da equação de continuidade :$$\frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \mathbf{u}) = 0$$Para fluidos incompressíveis, onde a densidade $\rho$ é constante no tempo e no espaço, a equação simplifica-se para a condição de divergência nula: $\nabla \cdot \mathbf{u} = 0$. Esta simplificação é amplamente utilizada em hidráulica e oceanografia física, onde as variações de densidade causadas pela pressão são desprezíveis.Conservação de Momentum: A Equação de CauchyA conservação de momentum decorre diretamente da Segunda Lei de Newton aplicada a um volume de fluido. A taxa de variação do momentum de uma parcela de fluido deve ser equilibrada pela soma de todas as forças aplicadas, classificadas em forças de corpo e forças de superfície. As forças de corpo, como a gravidade, atuam em todo o volume ($\mathbf{f}$), enquanto as forças de superfície resultam da pressão e das tensões viscosas, descritas pelo tensor de tensão $\sigma_{ij}$.A forma integral da conservação de momentum, utilizando o Teorema de Transporte de Reynolds, é:$$\frac{d}{dt} \iiint_{V(t)} \rho \mathbf{u} dV = \iiint_{V(t)} \mathbf{f} dV + \iint_{\partial V(t)} \sigma \cdot \mathbf{n} dA$$Ao aplicar o teorema da divergência ao termo de força de superfície e converter para a forma diferencial, chegamos à equação de momentum de Cauchy :$$\rho \left( \frac{\partial \mathbf{u}}{\partial t} + \mathbf{u} \cdot \nabla \mathbf{u} \right) = \nabla \cdot \sigma + \mathbf{f}$$O termo no lado esquerdo representa a derivada material ou substancial, que decompõe a aceleração em uma componente local (variação temporal no ponto fixo) e uma componente convectiva (mudança devido ao movimento do fluido através de gradientes de velocidade).Modelagem do Tensor de Tensão e ViscosidadeA transição da equação de Cauchy para a equação de Navier-Stokes propriamente dita requer uma relação constitutiva que descreva como o fluido responde às tensões. Para fluidos Newtonianos, assume-se que a tensão é linearmente proporcional à taxa de deformação (gradiente de velocidade). O tensor de tensão $\sigma$ é decomposto em uma parte isotrópica, associada à pressão $p$, e uma parte desviadora $\tau$, associada à viscosidade :$$\sigma_{ij} = -p \delta_{ij} + \tau_{ij}$$Para um fluido isotrópico, o tensor de tensões viscosas $\tau$ é expresso em termos da viscosidade dinâmica $\mu$ e da segunda viscosidade $\lambda$ :$$\tau_{ij} = \mu \left( \frac{\partial u_i}{\partial x_j} + \frac{\partial u_j}{\partial x_i} \right) + \lambda \delta_{ij} \nabla \cdot \mathbf{u}$$Substituindo esta relação na equação de Cauchy para um fluido incompressível com viscosidade constante, obtemos a forma mais comum das equações de Navier-Stokes :$$\rho \frac{D\mathbf{u}}{Dt} = -\nabla p + \mu \nabla^2 \mathbf{u} + \mathbf{f}$$Nesta formulação, o termo $\mu \nabla^2 \mathbf{u}$ representa a difusão do momentum devido à viscosidade, agindo para suavizar as variações de velocidade no escoamento.O Sistema Completo e Acoplado de EquaçõesEm problemas de alta complexidade, como em aerodinâmica compressível ou processos industriais de alta temperatura, a conservação de energia torna-se indispensável. O sistema completo de Navier-Stokes compreende cinco equações diferenciais acopladas que governam seis variáveis independentes: densidade, as três componentes da velocidade, pressão e temperatura.Estrutura do Sistema AcopladoA tabela abaixo resume as componentes do sistema completo de equações para um fluido compressível.EquaçãoConservação deTermos PrincipaisSignificado FísicoContinuidadeMassa$\frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \mathbf{u})$Balanço de fluxo de massaMomentum (X, Y, Z)Momentum$\frac{\partial (\rho \mathbf{u})}{\partial t} + \nabla \cdot (\rho \mathbf{u} \mathbf{u})$Balanço de forças e inérciaEnergiaEnergia Total$\frac{\partial E_t}{\partial t} + \nabla \cdot (\mathbf{u} (E_t + p))$Balanço térmico e trabalhoFonte: Elaborado com base em.Para fechar este sistema, é necessária uma equação de estado que relacione a pressão, a densidade e a temperatura, como a lei dos gases ideais $p = \rho R T$ para gases, ou relações mais complexas para líquidos.Parâmetros de Similitude e AdimensionalizaçãoA análise de escoamentos é facilitada pela adimensionalização das equações, o que revela parâmetros de controle que governam o comportamento do fluido.Número de Reynolds ($Re = \frac{\rho U L}{\mu}$): Determina a importância relativa das forças de inércia em comparação com as forças viscosas. É o principal indicador de turbulência.Número de Mach ($M = \frac{U}{c}$): Indica o grau de compressibilidade do escoamento. Para $M < 0,3$, os efeitos de compressibilidade são geralmente desprezíveis.Número de Prandtl ($Pr = \frac{\mu c_p}{k}$): Relaciona a difusividade de momentum com a difusividade térmica, essencial em problemas de transferência de calor.Os termos de convecção (lado esquerdo das equações de momentum) dominam em altos números de Reynolds, enquanto os termos de difusão (multiplicados por $1/Re$) prevalecem em escoamentos lentos e viscosos.Modelagem em Oceanografia e Engenharia MarítimaA aplicação das equações de Navier-Stokes no ambiente oceânico introduz variáveis adicionais, como a estratificação por densidade resultante de gradientes de temperatura e salinidade.A Aproximação de Boussinesq e BuoyancyPara muitos problemas oceânicos e meteorológicos, as variações de densidade são pequenas o suficiente para serem negligenciadas na inércia, mas críticas na geração de forças de empuxo. A aproximação de Boussinesq trata a densidade como constante em todos os termos, exceto no termo de gravidade ($\rho \mathbf{g}$) da equação de momentum.Esta simplificação assume uma variação linear da densidade em relação à temperatura: $\rho = \rho_0$, onde $\alpha$ é o coeficiente de expansão térmica. Embora computacionalmente eficiente e precisa para convecção natural com pequenas diferenças térmicas, a aproximação de Boussinesq torna-se inválida em escoamentos de alta velocidade ou quando os diferenciais de densidade são significativos, como no caso de bolhas de ar subindo na água.O Sistema Navier-Stokes-Fourier e SalinidadeRecentemente, a pesquisa em dinâmica oceânica avançou para o sistema Navier-Stokes-Fourier, que incorpora explicitamente a salinidade como uma variável de estado. Diferente da aproximação de Boussinesq, modelos mais recentes conservam a massa de forma rigorosa enquanto permitem a variação do volume do fluido devido aos efeitos de expansão termohalina.Neste contexto, a velocidade do fluido não é livre de divergência; em vez disso, o divergente do campo de velocidade é acoplado aos gradientes de temperatura e salinidade e à dissipação viscosa através da equação de energia. O uso do padrão internacional TEOS-10 (Equação Termodinâmica da Água do Mar - 2010) permite o cálculo de propriedades termodinâmicas consistentes, como a Salinidade Absoluta ($S_A$), que reflete a massa real de constituintes dissolvidos, superando a limitação da Salinidade Prática ($S_P$) baseada apenas na condutividade.Propriedade OceânicaUnidade (TEOS-10)Impacto na HidrodinâmicaSalinidade Absoluta ($S_A$)$g/kg$Aumenta a densidade e altera a estabilidade da coluna d'águaTemperatura Conservativa ($\Theta$)$^\circ C$Principal motor de correntes de densidade e fluxos de calorPressão Marinha ($p_{sea}$)$dbar$Modula a compressibilidade e a velocidade do somFonte: Elaborado com base em.Interação Fluido-Estrutura e Engenharia de CostasA engenharia oceânica depende fortemente das equações de Navier-Stokes para projetar estruturas resilientes a cargas ambientais extremas.Dinâmica de Estruturas Flexíveis (FFSI)A Interação Fluido-Estrutura Flexível (FFSI) é um campo de pesquisa crítico que estuda como as cargas de ondas e correntes deformam estruturas marinhas, e como essa deformação, por sua vez, altera o campo de escoamento. Aplicações incluem o projeto de cascos de navios sujeitos a "slamming" (impacto violento contra a água), hélices flexíveis, sistemas de energia de ondas e até a modelagem de vegetação marinha flexível.O desafio na modelagem de FFSI reside na necessidade de acoplar solvers de dinâmica de fluidos (CFD) com solvers de análise estrutural (FEA). Métodos de fronteira imersa ou técnicas de acoplamento particionado (como a combinação de Fluent e ABAQUS) são comumente empregados para resolver a interação transiente entre o escoamento e as grandes deformações estruturais.Propagação de Ondas e Quebra-maresNa modelagem de ondas costeiras, as equações de Navier-Stokes oferecem a descrição mais detalhada de fenômenos como reflexão, dissipação, galgamento (overtopping) e transmissão através de estruturas porosas. Ao contrário dos modelos de águas rasas ou de tipo Boussinesq, que fazem simplificações sobre a estrutura vertical do escoamento, modelos baseados em Navier-Stokes podem simular a rotação do fluido e a turbulência gerada pela quebra da onda.Modelos como o IH-2VOF utilizam as equações RANS (Reynolds-Averaged Navier-Stokes) para prever com alta precisão a estabilidade de quebra-mares de armadura de pedra, permitindo a simulação do fluxo dentro do núcleo poroso da estrutura. A comparação entre modelos mostra que, embora as equações de Boussinesq sejam eficientes para propagação em longas distâncias, as equações de Navier-Stokes são indispensáveis nas proximidades imediatas de obstáculos onde os efeitos viscosos e a turbulência são dominantes.Sedimentologia e Dinâmica do Leito MarinhoO estudo do transporte de sedimentos e da formação de erosão (scour) ao redor de infraestruturas como monopilhas de turbinas eólicas offshore e cabos submarinos exige uma abordagem multifásica das equações de Navier-Stokes.Abordagens de Modelagem de SedimentosExistem três metodologias principais para lidar com o transporte de sedimentos integradas ao solver de Navier-Stokes:Abordagem Simples: Utilizada para deformação do leito via acoplamento bidirecional. O escoamento é resolvido por RANS, e o transporte de carga de fundo é estimado por fórmulas empíricas, enquanto a carga em suspensão utiliza uma equação de advecção-difusão.Abordagem de Biot: Foca no comportamento mecânico do leito marinho como um meio poroso elástico. As equações de Navier-Stokes fornecem a pressão e as tensões na superfície do leito, que servem como condições de contorno para as equações de Biot que governam a estabilidade do solo e a liquefação potencial.Abordagem Multifásica Completa: Resolve as equações de conservação de massa e momentum para cada fase (água, ar e sedimento) simultaneamente. Embora seja a mais precisa para capturar a física detalhada na interface leito-fluido, seu custo computacional é extremamente elevado.As fórmulas de transporte, como as de Ribberink ou Van Rijn, relacionam a tensão de cisalhamento do leito ($\tau_b$) — calculada a partir do gradiente de velocidade de Navier-Stokes — com a taxa de fluxo de sedimentos, permitindo a atualização da morfologia do fundo através da equação de Exner.Desafios Numéricos e o Problema da TurbulênciaA turbulência é talvez o aspecto mais desafiador da dinâmica dos fluidos regida pelas equações de Navier-Stokes. Ela se caracteriza por um comportamento caótico, tridimensional e multiescala, onde redemoinhos de grandes dimensões transferem energia para escalas cada vez menores até a dissipação viscosa.Hierarquia de Modelagem de TurbulênciaDevido à vasta gama de escalas espaciais e temporais envolvidas, o custo computacional de resolver todas as escalas (DNS) cresce com $Re^3$, tornando-o inviável para aplicações de engenharia em larga escala.MétodoDescriçãoAplicabilidadeCusto ComputacionalDNS (Direct Numerical Simulation)Resolve todas as escalas da turbulênciaPesquisa fundamental, baixos $Re$Extremamente AltoLES (Large Eddy Simulation)Resolve grandes escalas, modela as pequenasEscoamentos complexos, descolamentoMédio-AltoRANS (Reynolds-Averaged NS)Modela toda a turbulência via estatísticaPadrão industrial, design de engenhariaBaixo-MédioFonte: Elaborado com base em.O problema de fechamento de turbulência surge no RANS devido à não linearidade do termo advectivo ($\mathbf{u} \cdot \nabla \mathbf{u}$), que, após a média temporal, gera termos desconhecidos chamados Tensões de Reynolds. Modelos clássicos como o $k-\epsilon$ ou $k-\omega$ utilizam a hipótese de viscosidade de redemoinho de Boussinesq para relacionar estas tensões aos gradientes do escoamento médio.Abordagens Lagrangianas vs. EulerianasA simulação numérica pode seguir duas perspectivas distintas. A perspectiva Euleriana observa o fluido passando por pontos fixos no espaço, sendo a base das malhas computacionais tradicionais e do método VOF (Volume of Fluid) para rastreamento de interface. A perspectiva Lagrangiana acompanha parcelas individuais de fluido em suas trajetórias, sendo exemplificada pelo método SPH (Smoothed Particle Hydrodynamics).O método SPH tem ganhado destaque em engenharia marítima por ser intrinsecamente capaz de lidar com superfícies livres complexas e quebra de ondas sem a necessidade de malhas deformáveis, embora apresente desafios em termos de ruído de pressão e custo computacional em comparação com métodos de volumes finitos Eulerianos.A Revolução da Inteligência Artificial em Dinâmica dos FluidosA integração de técnicas de aprendizado de máquina com as equações de Navier-Stokes está transformando a área de fluidos computacional, permitindo acelerações significativas e novas capacidades de assimilação de dados.PINNs: Redes Neurais Informadas pela FísicaAs PINNs (Physics-Informed Neural Networks) representam uma mudança de paradigma ao codificar as leis da física (Navier-Stokes) diretamente na função de perda da rede neural. Diferente dos solvers tradicionais, as PINNs não requerem uma malha pré-definida e podem inferir campos de fluxo ocultos a partir de medições parciais.Por exemplo, uma PINN pode deduzir o potencial de velocidade totalmente não linear em todo o volume de um fluido baseando-se apenas em medições da elevação da superfície livre. Modelos como o SIMPLE-PINN incorporam termos de correção de pressão inspirados em algoritmos clássicos de CFD para estabilizar o treinamento e acelerar a convergência em problemas altamente não lineares.PINOs e Operadores NeuraisEnquanto as PINNs são treinadas para resolver uma instância específica de um problema, os PINOs (Physics-Informed Neural Operators) aprendem o operador de solução completo. Isso significa que, uma vez treinado, um PINO pode prever a evolução de um campo de ondas ou a dispersão de um poluente para novas condições iniciais de forma quase instantânea.Estudos demonstraram que PINOs podem ser usados para reconstruir campos de ondas oceânicas em tempo real a partir de dados esparsos de boias ou radar, superando as limitações dos modelos espectrais tradicionais em lidar com a não linearidade extrema. Além disso, técnicas de ajuste fino informadas pela física (PFT-CNO) estão sendo desenvolvidas para mitigar o problema do desequilíbrio de gradientes entre os dados e as perdas físicas, melhorando a precisão em até 26,4% em relação aos métodos puramente orientados por dados.O Problema do Prêmio Millennium: Existência e SuavidadeApesar de sua onipresença prática, as equações de Navier-Stokes em três dimensões ocultam um dos maiores mistérios da matemática. O Problema do Prêmio Millennium, proposto pelo Clay Mathematics Institute, desafia a comunidade científica a provar se soluções suaves e globalmente definidas sempre existem para condições iniciais razoáveis.O Desafio da SingularidadeA questão central é se o modelo de continuum pode "quebrar" sob certas condições, levando a uma singularidade onde a velocidade ou a energia do fluido se tornam infinitas em um tempo finito. Em duas dimensões, a suavidade das soluções já foi comprovada, mas a terceira dimensão introduz o mecanismo de estiramento de vórtices, que tem o potencial de concentrar vorticidade de forma descontrolada.As duas conjecturas em competição são:Conjectura de Suavidade: Para qualquer campo de velocidade inicial suave, existirá sempre uma solução suave e bem comportada para todo o tempo.Conjectura de Ruptura (Breakdown): Existe pelo menos um conjunto de condições iniciais para as quais a solução deixa de ser suave (desenvolve uma singularidade) em tempo finito.A resolução deste problema teria implicações profundas não apenas para a matemática pura, mas para a validade do uso de Navier-Stokes como uma descrição definitiva da turbulência em escalas extremas.Síntese e Perspectivas FuturasAs equações de Navier-Stokes continuam a ser o epicentro da pesquisa em dinâmica dos fluidos, evoluindo de uma formulação clássica para sistemas complexos que integram termodinâmica oceânica, interações multifásicas e inteligência artificial. A capacidade de modelar com precisão a densidade variável através do TEOS-10 e da aproximação de Boussinesq permitiu avanços críticos na compreensão da circulação termohalina e seu impacto no clima global.Na engenharia prática, a tendência futura aponta para o uso crescente de modelos híbridos que combinam a precisão das equações RANS/LES com a velocidade dos operadores neurais (PINOs). Esta sinergia permitirá a criação de sistemas de previsão em tempo real para segurança marítima, otimização de parques eólicos offshore e monitoramento de riscos costeiros. Simultaneamente, o aprofundamento na modelagem de FFSI e transporte de sedimentos continuará a desafiar os limites computacionais, exigindo algoritmos cada vez mais robustos e eficientes.Em última análise, as equações de Navier-Stokes representam a ponte entre as leis universais da física e a complexidade imprevisível do mundo natural. Seja na busca por uma prova matemática de suavidade ou no desenvolvimento de novos solvers neurais, o estudo contínuo destas equações permanece como um dos empreendimentos mais vitais e intelectuais da ciência moderna, garantindo que o movimento dos fluidos possa ser compreendido, previsto e aproveitado em benefício da sociedade.


***

Enquanto os **gráficos** que vimos antes servem para visualizar dados estatísticos, os **grafos** em computação e matemática são estruturas usadas para modelar **conexões entre objetos**.

Imagine um mapa de metrô ou uma rede de amizades: isso é um grafo.

### 1. Elementos Fundamentais
Um grafo é composto basicamente por dois elementos:
* **Vértices (ou Nós):** Representam as entidades (ex: cidades, pessoas, roteadores).
* **Arestas:** Representam as conexões ou relações entre esses nós (ex: estradas, amizades, cabos de rede).

### 2. Tipos de Grafos
* **Direcionados:** As conexões têm um sentido (como uma rua de mão única ou seguir alguém no Instagram).
* **Não Direcionados:** A conexão é mútua (como uma amizade no Facebook).
* **Ponderados:** As arestas possuem um "peso" ou custo (ex: a distância em quilômetros entre duas cidades num GPS).

---

### 3. Para que servem na prática?
Os grafos são a base de algoritmos fundamentais que usamos todos os dias:

* **Redes Sociais:** Sugestão de "pessoas que você talvez conheça".
* **GPS (Google Maps):** Para encontrar o caminho mais curto entre dois pontos (Algoritmo de Dijkstra).
* **Mecanismos de Busca:** O Google usa grafos para entender como as páginas da web se conectam (PageRank).
* **Logística:** Otimização de rotas de entrega.

### Representação em Código
Em programação, você geralmente representa um grafo de duas formas:
1.  **Matriz de Adjacência:** Uma tabela onde 1 indica conexão e 0 indica ausência dela.
2.  **Lista de Adjacência:** Uma lista onde cada nó guarda quem são seus "vizinhos" diretos.

Se você estiver estudando algoritmos, os grafos são o próximo passo natural depois de listas e árvores!
