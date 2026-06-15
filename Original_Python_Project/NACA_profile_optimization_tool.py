import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os
from scipy.optimize import minimize
import shlex
import shutil

# ==============================================================================
# FUNZIONI DI GENERAZIONE DEL PROFILO ALARE
# ==============================================================================

def naca4(m_param, p_param, t_param, c=1.0, n=100):
    """
    Genera le coordinate di un profilo alare NACA a 4 cifre da parametri.
    """
    x = np.linspace(0, c, n)
    
    # Distribuzione dello spessore
    yt = 5 * t_param * c * (0.2969 * np.sqrt(x/c) - 0.1260 * (x/c) - 0.3516 * (x/c)**2 + 0.2843 * (x/c)**3 - 0.1015 * (x/c)**4)

    if p_param == 0 or m_param == 0:
        # Profilo simmetrico
        xu, yu = x, yt
        xl, yl = x, -yt
    else:
        # Profilo curvo
        yc = np.zeros_like(x)
        dyc_dx = np.zeros_like(x)
        
        front_x = x[x < p_param * c]
        back_x = x[x >= p_param * c]

        if len(front_x) > 0:
            yc_front = (m_param / p_param**2) * (2 * p_param * (front_x / c) - (front_x / c)**2)
            dyc_dx_front = (2 * m_param / p_param**2) * (p_param - front_x / c)
            yc[:len(front_x)] = yc_front
            dyc_dx[:len(front_x)] = dyc_dx_front

        if len(back_x) > 0:
            yc_back = (m_param / (1 - p_param)**2) * ((1 - 2 * p_param) + 2 * p_param * (back_x / c) - (back_x / c)**2)
            dyc_dx_back = (2 * m_param / (1 - p_param)**2) * (p_param - back_x / c)
            yc[len(front_x):] = yc_back
            dyc_dx[len(front_x):] = dyc_dx_back
            
        theta = np.arctan(dyc_dx)
        xu = x - yt * np.sin(theta)
        yu = yc + yt * np.cos(theta)
        xl = x + yt * np.sin(theta)
        yl = yc - yt * np.cos(theta)

    # Unisci le coordinate per il file di XFOIL (in senso orario, partendo dal bordo d'uscita)
    X = np.concatenate((np.flip(xu), xl[1:]))
    Y = np.concatenate((np.flip(yu), yl[1:]))
    
    return X, Y, (xu, yu, xl, yl)

def save_airfoil_to_file(X, Y, filename):
    """Salva le coordinate del profilo in un file .dat per XFOIL."""
    with open(filename, "w") as f:
        for i in range(len(X)):
            f.write(f"{X[i]:.6f} {Y[i]:.6f}\n")

# ==============================================================================
# FUNZIONE DI ANALISI CFD (WRAPPER XFOIL)
# ==============================================================================

def run_xfoil_analysis(airfoil_file, alpha, Re, Mach=0.0):
    """
    Esegue un'analisi XFOIL per un dato profilo e condizioni di volo.
    Restituisce (CL, CD).
    """
    xfoil_input_file = "xfoil_input.in"
    polar_file = "polar.dat"

    with open(xfoil_input_file, "w") as f:
        f.write(f"LOAD {airfoil_file}\n")
        f.write("PANE\n")  
        f.write("OPER\n")
        f.write(f"Visc {Re}\n")
        f.write(f"Mach {Mach}\n")
        f.write("PACC\n")
        f.write(f"{polar_file}\n\n") 
        f.write(f"ALFA {alpha}\n")
        f.write("\n")      
        f.write("QUIT\n")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    xfoil_exe_path = os.path.join(script_dir, "xfoil.exe")
    
    command = f'"{xfoil_exe_path}" < "{xfoil_input_file}"'

    try:
        subprocess.run(command, shell=True, check=True, capture_output=True, text=True, timeout=30)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        return None, None

    cl, cd = None, None
    try:
        with open(polar_file, "r") as f:
            lines = [line for line in f if not line.startswith("#")]
            if lines:
                data = lines[-1].split()
                if len(data) >= 3:
                    cl = float(data[1])
                    cd = float(data[2])
    except (IOError, IndexError):
        pass
    
    for f in [xfoil_input_file, polar_file, airfoil_file]:
        if os.path.exists(f):
            os.remove(f)
            
    return cl, cd

# ==============================================================================
# FUNZIONE OBIETTIVO PER L'OTTIMIZZAZIONE
# ==============================================================================

def objective_function(params, Re, alpha):
    """
    Funzione obiettivo da minimizzare. -CL/CD
    """
    m, p, t = params
    
    # Vincoli impliciti: i parametri devono avere senso
    if not (0.0 <= m < 0.1 and 0.1 <= p < 0.8 and 0.05 <= t < 0.25):
        return 1e6

    airfoil_name = f"temp_naca.dat"
    X, Y, _ = naca4(m, p, t)
    save_airfoil_to_file(X, Y, airfoil_name)
    
    cl, cd = run_xfoil_analysis(airfoil_name, alpha, Re)
    
    if cl is not None and cd is not None and cd > 0:
        efficiency = cl / cd
        print(f"Test NACA(m={m:.3f}, p={p:.3f}, t={t:.3f}) -> CL/CD = {efficiency:.2f}")
        return -efficiency
    else:
        print(f"--- Analisi fallita per NACA(m={m:.3f}, p={p:.3f}, t={t:.3f}). Assegno penalità.")
        return 1e6

# ==============================================================================
# FUNZIONI DI INPUT INTERATTIVO
# ==============================================================================

def get_float_input(prompt, default_val):
    """Richiede un numero float, usa il default se l'input è vuoto."""
    while True:
        val = input(f"{prompt} [predefinito: {default_val}]: ").strip()
        if not val:
            return default_val
        try:
            return float(val)
        except ValueError:
            print("  Errore: Inserisci un numero valido.")

def get_naca_input(prompt, default_val="0012"):
    """Richiede 4 cifre NACA e restituisce [m, p, t] e la stringa inserita."""
    while True:
        val = input(f"{prompt} [predefinito: {default_val}]: ").strip()
        if not val:
            val = default_val
        
        if len(val) == 4 and val.isdigit():
            m = int(val[0]) / 100.0
            p = int(val[1]) / 10.0
            # Se p è 0, lo impostiamo a un valore fittizio (es. 0.3) per evitare errori matematici
            if p == 0: p = 0.3 
            t = int(val[2:4]) / 100.0
            return [m, p, t], val
        else:
            print("  Errore: Devi inserire esattamente 4 cifre (es. 0012, 2412).")

# ==============================================================================
# BLOCCO PRINCIPALE DI ESECUZIONE
# ==============================================================================

if __name__ == "__main__":
    
    # CONTROLLO PREREQUISITI
    script_dir = os.path.dirname(os.path.abspath(__file__))
    xfoil_exe_path = os.path.join(script_dir, "xfoil.exe")
    
    if not os.path.exists(xfoil_exe_path):
        print("="*60)
        print("!!! ERRORE CRITICO: Eseguibile 'xfoil.exe' non trovato.")
        print(f"Assicurati che XFOIL sia presente in questa cartella:\n{script_dir}")
        print("="*60)
        exit()

    # --- MENU INTERATTIVO ---
    print("\n" + "="*50)
    print(" BENVENUTO NELLO STRUMENTO DI OTTIMIZZAZIONE ALARE")
    print("="*50)
    
    TARGET_REYNOLDS = get_float_input("1. Inserisci il numero di Reynolds", 1000000.0)
    TARGET_ALPHA = get_float_input("2. Inserisci l'angolo d'attacco in gradi", 1.0)
    initial_guess, str_naca = get_naca_input("3. Inserisci il profilo NACA iniziale (4 cifre)")
    
    # Limiti allargati per accogliere una gamma più ampia di profili
    bounds = [(0.0, 0.09), (0.1, 0.8), (0.05, 0.25)]

    print("\n" + "-" * 35)
    print("--- INIZIO OTTIMIZZAZIONE ---")
    print(f"Condizioni: Re = {TARGET_REYNOLDS}, Angolo d'attacco = {TARGET_ALPHA}°")
    print(f"Profilo iniziale: NACA {str_naca} (m={initial_guess[0]}, p={initial_guess[1]}, t={initial_guess[2]})")
    print("-" * 35)

    # Esegui l'ottimizzazione
    result = minimize(
        objective_function,
        initial_guess,
        args=(TARGET_REYNOLDS, TARGET_ALPHA),
        method='SLSQP',
        bounds=bounds,
        options={'disp': True, 'maxiter': 150, 'ftol': 1e-8}
    )

    print("\n" + "-" * 35)
    print("--- RIEPILOGO OTTIMIZZAZIONE ---")
    print(f"Messaggio di terminazione: {result.message}")
    
    if result.success and -result.fun > 0:
        optimal_params = result.x
        max_efficiency = -result.fun
        
        # Formattiamo i parametri ottimali per mostrarli come una sigla NACA approssimata
        m_opt_str = str(int(round(optimal_params[0] * 100)))
        p_opt_str = str(int(round(optimal_params[1] * 10)))
        t_opt_str = f"{int(round(optimal_params[2] * 100)):02d}"
        naca_opt_str = f"{m_opt_str}{p_opt_str}{t_opt_str}"

        print(f"\n✅ Ottimizzazione completata con successo!")
        print(f"Efficienza massima (CL/CD): {max_efficiency:.2f}")
        print(f"Parametri ottimali (m, p, t): {optimal_params[0]:.4f}, {optimal_params[1]:.4f}, {optimal_params[2]:.4f}")
        print(f"Profilo ottimizzato risultante (Approssimato): NACA {naca_opt_str}")
        
        # Visualizza i profili
        _, _, coords_initial = naca4(*initial_guess)
        _, _, coords_optimal = naca4(*optimal_params)

        plt.figure(figsize=(12, 8))
        plt.plot(coords_initial[0], coords_initial[1], 'b--', label=f'Iniziale (NACA {str_naca})')
        plt.plot(coords_initial[2], coords_initial[3], 'b--')
        plt.plot(coords_optimal[0], coords_optimal[1], 'r-', label=f'Ottimale (NACA ~{naca_opt_str} | CL/CD={max_efficiency:.2f})')
        plt.plot(coords_optimal[2], coords_optimal[3], 'r-')
        
        plt.title(f"Confronto tra Profilo Iniziale e Ottimale (Re={TARGET_REYNOLDS}, Alpha={TARGET_ALPHA}°)")
        plt.xlabel('x/c')
        plt.ylabel('y/c')
        plt.axis('equal')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    else:
        print("\n❌ L'ottimizzazione non è riuscita a trovare una soluzione migliorativa o è fallita.")
        print(f"Causa: {result.message}")
        print(f"Ultimi parametri tentati: {result.x}")