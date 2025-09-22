import numpy as np
import matplotlib.pyplot as plt

class Shooting:
    def __init__(self, K=100, r=0.05, sigma=0.2):
        """
        PASO 1: Inicialización de parámetros

        - Define los parámetros del put americano perpetuo
        - Calcula el valor teórico de L* para comparar después
        - El L* teórico nos sirve para validar nuestro resultado
        """
        self.K = K           # Strike price
        self.r = r           # Tasa libre de riesgo
        self.sigma = sigma   # Volatilidad
        
        # Fórmula teórica para L*
        self.L_star_teorico = (2 * self.r / (2 * self.r + self.sigma**2)) * self.K
    
    def ode_system(self, S, y):
        """
        PASO 2: Sistema de ODEs de primer orden

        - Convierte tu ODE de 2° orden en sistema de 1° orden
        - y[0] = V(S)    <- la función que buscamos
        - y[1] = V'(S)   <- su derivada

        La transformación:
        Original: ½σ²S²V'' + rSV' - rV = 0
        Despejo:  V'' = (rV - rSV') / (½σ²S²)
        Sistema:  y₁' = y₂
                 y₂' = (ry₁ - rSy₂) / (½σ²S²)
        """
        V, V_prime = y
        
        # Evitar división por cero cuando S es muy pequeño
        if S <= 1e-10:
            return np.array([V_prime, 0])
        
        # Calcular V'' despejado de la ODE original
        V_double_prime = (self.r * V - self.r * S * V_prime) / (0.5 * self.sigma**2 * S**2)
        
        # Retornar [y₁', y₂'] = [V', V'']
        return np.array([V_prime, V_double_prime])
    
    def runge_kutta_4(self, f, S_span, y0, h):
        """
        PASO 3: Integrador Runge-Kutta 4 
        
        - Integra numéricamente el sistema de ODEs
        
        Parámetros:
        - f: función del sistema (self.ode_system)
        - S_span: [S_inicial, S_final] 
        - y0: condiciones iniciales [V(S₀), V'(S₀)]
        - h: tamaño del paso
        """
        S_start, S_end = S_span
        n_steps = int((S_end - S_start) / h)
        
        # Arrays para almacenar resultados
        S_vals = np.linspace(S_start, S_end, n_steps + 1)
        y_vals = np.zeros((len(y0), n_steps + 1))
        y_vals[:, 0] = y0  # Condición inicial
        
        # Bucle de integración RK4
        for i in range(n_steps):
            S_i = S_vals[i]
            y_i = y_vals[:, i]
            
            # Los 4 pasos de Runge-Kutta
            k1 = h * f(S_i, y_i)
            k2 = h * f(S_i + h/2, y_i + k1/2)
            k3 = h * f(S_i + h/2, y_i + k2/2)
            k4 = h * f(S_i + h, y_i + k3)
            
            # Combinación ponderada
            y_vals[:, i+1] = y_i + (k1 + 2*k2 + 2*k3 + k4) / 6
        
        return S_vals, y_vals

    def shoot_from_L(self, L, S_max=1000, h=0.1):
        """
        PASO 4: Función de shooting

        - Para un valor dado de L, resuelve la ODE
        - Impone las condiciones iniciales en S = L
        - Integra hasta S = S_max
        
        Condiciones iniciales:
        - V(L) = K - L    (value matching)
        - V'(L) = -1      (smooth pasting)
        """
        # Condiciones iniciales en S = L
        y0 = np.array([self.K - L, -1])  # [V(L), V'(L)]
        
        try:
            # Integrar desde L hasta S_max
            S_vals, y_vals = self.runge_kutta_4(
                self.ode_system, [L, S_max], y0, h
            )
            
            # Retornar S, V(S), V'(S)
            return S_vals, y_vals[0], y_vals[1]
            
        except:
            return None, None, None
    
    def boundary_condition_error(self, L):
        """
        PASO 5: Función objetivo

        - Mide qué tan cerca estamos del objetivo, que es V(S_max) ≈ 0
        - Retorna qué tan lejos estamos de 0
        
        - Cuando esta función sea 0, habremos encontrado L*
        """
        S_vals, V_vals, V_prime_vals = self.shoot_from_L(L)
        
        if V_vals is None:
            return float('inf')  # Error infinito si falló la integración
        
        # Verificar si la solución explota
        if np.any(np.abs(V_vals) > 1e6) or np.any(np.isnan(V_vals)):
            return float('inf')
        
        # El error es V en el punto más lejano
        # Queremos que V(S_max) ≈ 0
        return V_vals[-1]
    
    def bisection_method(self, f, a, b, tol=1e-6, max_iter=100):
        """
        PASO 6: Método de bisección
        
        - Encuentra la raíz de una función 
        
        Algoritmo:
        1. Verifica que f(a) y f(b) tengan signos opuestos
        2. Toma el punto medio c = (a+b)/2
        3. Si f(c) tiene el mismo signo que f(a), reemplaza a por c
        4. Si no, reemplaza b por c
        5. Repite hasta que c sea suficientemente preciso
        """
        fa = f(a)
        fb = f(b)
        
        if fa * fb > 0:
            raise ValueError("f(a) y f(b) deben tener signos opuestos para bisección")
        
        for i in range(max_iter):
            c = (a + b) / 2
            fc = f(c)
            
            if abs(fc) < tol or (b - a) / 2 < tol:
                return c
            
            if fa * fc < 0:
                b = c
                fb = fc
            else:
                a = c
                fa = fc
        
        return c
    
    def find_optimal_L(self, L_min=None, L_max=None):
        """
        PASO 7: Búsqueda del L óptimo

        - Encuentra el L que hace boundary_condition_error(L) = 0
        - Usa bisección para encontrar la raíz
        """
        if L_min is None:
            L_min = 0.1 * self.K
        if L_max is None:
            L_max = 0.9 * self.K
        
        # Verificar que hay cambio de signo
        error_min = self.boundary_condition_error(L_min)
        error_max = self.boundary_condition_error(L_max)

        if error_min * error_max > 0:
            # Buscar un rango válido
            L_test = np.linspace(L_min, L_max, 20)
            for L in L_test:
                error = self.boundary_condition_error(L)
                if error * error_min < 0:
                    L_max = L
                    break
            else:
                return None
        
        try:
            # Encontrar L óptimo usando bisección
            L_optimal = self.bisection_method(
                self.boundary_condition_error, L_min, L_max
            )

            return L_optimal
            
        except Exception as e:
            return None
    
    def validate_solution(self, L_optimal):
        """
        PASO 8: Validación de la solución

        - Verifica que la solución encontrada cumpla todas las condiciones
        - Chequea value matching, smooth pasting, límite infinito, etc.
        """
        print(f"\n=== VALIDACIÓN DE LA SOLUCIÓN ===")
        
        S_vals, V_vals, V_prime_vals = self.shoot_from_L(L_optimal)
        
        if V_vals is None:
            print("Error: No se pudo integrar la ODE")
            return False
        
        # 1. Value matching: V(L*) debe ser K - L*
        V_L = V_vals[0]
        expected_V_L = self.K - L_optimal
        value_match_error = abs(V_L - expected_V_L)
        print(f"1. Value matching:")
        print(f"   V(L*) = {V_L:.6f}")
        print(f"   K-L* = {expected_V_L:.6f}")
        print(f"   Error: {value_match_error:.2e}")
        
        # 2. Smooth pasting: V'(L*) debe ser -1
        V_prime_L = V_prime_vals[0]
        smooth_error = abs(V_prime_L - (-1))
        print(f"2. Smooth pasting:")
        print(f"   V'(L*) = {V_prime_L:.6f}")
        print(f"   Debe ser: -1.000000")
        print(f"   Error: {smooth_error:.2e}")
        
        # 3. Límite al infinito: V(S_max) debe ser ≈ 0
        boundary_error = abs(V_vals[-1])
        print(f"3. Límite infinito:")
        print(f"   V(S_max) = {V_vals[-1]:.6f}")
        print(f"   |V(S_max)| = {boundary_error:.2e}")
        
        # 4. Cota inferior: V(S) >= (K-S)+ para todo S
        S_test = S_vals[S_vals <= 2*self.K]  # Solo valores razonables
        V_test = V_vals[:len(S_test)]
        payoff = np.maximum(self.K - S_test, 0)
        violations = np.sum(V_test < payoff - 1e-6)
        
        print(f"4. Cota inferior V(S) >= (K-S)+:")
        print(f"   Puntos verificados: {len(S_test)}")
        print(f"   Violaciones: {violations}")
        
        # Resumen final
        is_valid = (value_match_error < 1e-4 and 
                   smooth_error < 1e-4 and 
                   boundary_error < 1e-2 and 
                   violations == 0)
        
        print(f"\n¿Solución válida? {'SÍ' if is_valid else 'NO'}")
        
        return is_valid
    
class FiniteDifferences:
    def __init__(self, K=50, r=0.05, sigma=0.2):
        self.K = K
        self.r = r
        self.sigma = sigma
        self.L_analitico = (2*r / (2*r + sigma**2)) * K

    def solve(self, S_max=120, N=2000, epsilon=1e-6, tol=1e-6, max_iter=2000, rho=0.1):
        dx = S_max / N
        grid = np.linspace(0, S_max, N+1)
        
        # Inicializar con el payoff
        u = np.maximum(self.K - grid, 0.0)
        payoff = u.copy()
        
        for k in range(max_iter):
            u_prev = u.copy()
            
            # Armar sistema lineal
            M = np.zeros((N+1, N+1))
            f = np.zeros(N+1)
            
            # Condiciones de borde
            M[0, 0] = 1
            f[0] = self.K
            M[N, N] = 1  
            f[N] = 0
            
            # Nodos internos
            for j in range(1, N):
                S_j = grid[j]
                
                if self.K - S_j > u[j] + epsilon:
                    # Región de ejercicio
                    M[j, j] = 1
                    f[j] = self.K - S_j
                else:
                    # Aplicar ODE discretizada
                    sigma2_S2 = self.sigma**2 * S_j**2
                    r_S = self.r * S_j
                    
                    M[j, j-1] = 0.5 * sigma2_S2 / dx**2 - 0.5 * r_S / dx
                    M[j, j] = -(sigma2_S2 / dx**2 + self.r)
                    M[j, j+1] = 0.5 * sigma2_S2 / dx**2 + 0.5 * r_S / dx
                    f[j] = 0
            
            # Resolver y aplicar relajación
            u_new = np.linalg.solve(M, f)
            u = (1 - rho) * u_prev + rho * u_new
            
            if np.linalg.norm(u - u_prev, np.inf) < tol:
                print(f"Convergió en {k+1} iteraciones")
                break
        else:
            print(f"No convergió en {max_iter} iteraciones")
        
        # Encontrar frontera de ejercicio
        diff = u - payoff
        L_num = None
        for i in range(len(diff)-1):
            if diff[i] <= 0 and diff[i+1] > 0:
                L_num = grid[i]
                break
        
        self.results = {
            'S': grid,
            'V': u,
            'Payoff': payoff,
            'L_numerico': L_num,
            'L_analitico': self.L_analitico,
            'iterations': min(k+1, max_iter)
        }
        
        return self.results

def V_analitica(S, K, r, sigma):

    sigma_sq = sigma**2
    L_star = (2*r / (2*r + sigma_sq)) * K
    alpha = 2*r / sigma_sq
    A = K * sigma_sq / (2*r + sigma_sq)
    
    # Evitar problemas numéricos
    S_safe = np.maximum(S, 1e-12)
    # opción 1 (con factor explícito)
    continuation_part = A * (((2*r + sigma_sq)/(2*r)) * (S_safe / K))**(-alpha)
    
    V = np.where(S <= L_star, K - S, continuation_part)
    return V, L_star

def comparar_metodos(K=100, r=0.05, sigma=0.2, S_max=300, N=2000):

    print("="*60)
    print("COMPARACIÓN DE MÉTODOS - PUT AMERICANO PERPETUO")
    print("="*60)
    print(f"Parámetros: K={K}, r={r}, σ={sigma}")
    print("-"*60)
    
    # 1. SOLUCIÓN ANALÍTICA
    S_grid = np.linspace(0, S_max, N+1)
    V_ana, L_ana = V_analitica(S_grid, K, r, sigma)
    print(f"L* Analítico:        {L_ana:.6f}")
    
    # 2. MÉTODO SHOOTING
    print("\nEjecutando Shooting Method...")
    shooting = Shooting(K, r, sigma)
    L_shoot = shooting.find_optimal_L()
    
    if L_shoot is not None:
        # Generar solución completa de shooting
        S_sh, V_cont_sh, _ = shooting.shoot_from_L(L_shoot, S_max=S_max, h=0.05)
        V_shoot = np.where(S_grid <= L_shoot, K - S_grid, np.interp(S_grid, S_sh, V_cont_sh))
        print(f"L* Shooting:         {L_shoot:.6f}")
        error_L_shoot = abs(L_shoot - L_ana) / L_ana * 100
        print(f"Error L* Shooting:   {error_L_shoot:.3f}%")
    else:
        V_shoot = None
        L_shoot = None
        print("L* Shooting:         FALLÓ")
    
    # 3. DIFERENCIAS FINITAS
    print("\nEjecutando Diferencias Finitas...")
    fd = FiniteDifferences(K, r, sigma)
    fd_results = fd.solve(S_max=S_max, N=N)
    V_fd = fd_results['V']
    L_fd = fd_results['L_numerico']
    
    if L_fd is not None:
        print(f"L* Dif. Finitas:     {L_fd:.6f}")
        error_L_fd = abs(L_fd - L_ana) / L_ana * 100
        print(f"Error L* Dif. Finitas: {error_L_fd:.3f}%")
    else:
        print("L* Dif. Finitas:     NO ENCONTRADO")
    
    # TABLA RESUMEN
    print("\n" + "="*60)
    print("TABLA RESUMEN DE L*")
    print("="*60)
    print(f"{'Método':<20} {'L*':<12} {'Error (%)':<10}")
    print("-"*42)
    print(f"{'Analítico':<20} {L_ana:<12.6f} {'0.000':<10}")
    
    if L_shoot is not None:
        print(f"{'Shooting':<20} {L_shoot:<12.6f} {error_L_shoot:<10.3f}")
    else:
        print(f"{'Shooting':<20} {'FALLÓ':<12} {'N/A':<10}")
    
    if L_fd is not None:
        print(f"{'Dif. Finitas':<20} {L_fd:<12.6f} {error_L_fd:<10.3f}")
    else:
        print(f"{'Dif. Finitas':<20} {'NO ENCONTRADO':<12} {'N/A':<10}")
    
    print("="*60)
    
    return {
        'S': S_grid,
        'V_ana': V_ana, 'L_ana': L_ana,
        'V_shoot': V_shoot, 'L_shoot': L_shoot,
        'V_fd': V_fd, 'L_fd': L_fd
    }

def graficar_comparacion(resultados):
    S = resultados['S']
    V_ana = resultados['V_ana']
    L_ana = resultados['L_ana']
    V_shoot = resultados['V_shoot']
    L_shoot = resultados['L_shoot']
    V_fd = resultados['V_fd']
    L_fd = resultados['L_fd']
    
    plt.figure(figsize=(12, 8))

    mask = S <= 2*L_ana
    S_plot = S[mask]
    
    # Calcular el payoff para la región graficada
    # Asumir K del primer resultado disponible
    K = L_ana / (2*0.05/(2*0.05 + 0.04))  
    K = 100  # Valor directo si se conoce
    payoff = np.maximum(K - S_plot, 0)
    
    # Graficar payoff primero (para que quede detrás)
    plt.plot(S_plot, payoff, 'g:', linewidth=2, label='Payoff (K-S)+', alpha=0.6)
    
    # Graficar soluciones
    plt.plot(S_plot, V_ana[mask], 'k-', linewidth=3, label='Analítica', alpha=0.8)
    
    if V_shoot is not None:
        plt.plot(S_plot, V_shoot[mask], 'b--', linewidth=2, label='Shooting', alpha=0.7)
    
    plt.plot(S_plot, V_fd[mask], 'r:', linewidth=2, label='Diferencias Finitas', alpha=0.7)
    
    # Líneas verticales para L*
    plt.axvline(L_ana, color='k', linestyle='-', alpha=0.5, 
                label=f'L* Analítica = {L_ana:.3f}')
    
    if L_shoot is not None:
        plt.axvline(L_shoot, color='b', linestyle='--', alpha=0.5,
                    label=f'L* Shooting = {L_shoot:.3f}')
    
    if L_fd is not None:
        plt.axvline(L_fd, color='r', linestyle=':', alpha=0.5,
                    label=f'L* Dif. Finitas = {L_fd:.3f}')
    
    plt.xlabel('Precio del Subyacente (S)')
    plt.ylabel('Valor de la Opción V(S)')
    plt.title('Put Americano Perpetuo - Comparación de Métodos')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.ylim(0, None)
    
    plt.tight_layout()
    plt.show()

resultados = comparar_metodos(K=100, r=0.05, sigma=0.2)
graficar_comparacion(resultados)