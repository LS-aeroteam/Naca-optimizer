import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os
from scipy.optimize import minimize

# Contatore globale per la tabella di output
eval_count = 0

# ==============================================================================
# FUNZIONI DI GENERAZIONE DEL PROFILO ALARE
# ==============================================================================

def naca4(m_param, p_param, t_param, c=1.0, n=100):
    x = np.linspace(0, c, n)
    yt = 5 * t_param * c * (0.2969 * np.sqrt(x/c) - 0.1260 * (x/c) - 0.3516 * (x/c)**2 + 0.2843 * (x/c)**3 - 0.1015 * (x/c)**4)

    if p_param == 0 or m_param == 0:
        xu, yu = x, yt
        xl, yl = x, -yt
    else:
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

    X = np.concatenate((np.flip(xu), xl[1:]))
    Y = np.concatenate((np.flip(yu), yl[1:]))
    
    return X, Y, (xu, yu, xl, yl)

def save_airfoil_to_file(X, Y, filename):
    with open(filename, "w") as f:
        for i in range(len(X)):
            f.write(f"{X[i]:.6f} {Y[i]:.6f}\n")

# ==============================================================================
# FUNZIONE DI ANALISI CFD (WRAPPER XFOIL)
# ==============================================================================

def run_xfoil_analysis(airfoil_file, alpha, Re, Mach=0.0):
    xfoil_input_file = "xfoil_input.in"
    polar_file = "polar.dat"

    with open(xfoil_input_file, "w") as f:
        f.write(f"LOAD {airfoil_file}\n")
        f.write("PANE\n")  
        f.write("OPER\n")
        f.write(f"Visc {Re}\n")
        f.write(f"Mach {Mach}\n")
        f.write("ITER 300\n") # Aumentate le iterazioni di convergenza interna
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
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None, None

    cl, cd = None, None
    try:
        with open(polar_file, "r") as f:
            lines = [line for line in f if not line.startswith("#")]
            if lines:
                data = lines[-1].split()
                if len(data) >= 3:
                    try:
                        cl = float(data[1])
                        cd = float(data[2])
                    except ValueError:
                        # Gestisce il caso in cui XFOIL stampi '--------'
                        cl, cd = None, None
    except (IOError, IndexError):
        pass
    
    for f in [xfoil_input_file, polar_file, airfoil_file]:
        if os.path.exists(f):
            os.remove(f)
            
    return cl, cd

# ==============================================================================
# FUNZIONE OBIETTIVO PER DESIGN INVERSO
# ==============================================================================

def objective_function(params, Re, alpha, target_cl, max_cd):
    global eval_count
    eval_count += 1
    m, p, t = params
    
    # Vincoli estremi allargati (per profili ad alta curvatura)
    if not (0.0 <= m <= 0.20 and 0.1 <= p <= 0.8 and 0.05 <= t <= 0.35):
        print(f"| {eval_count:4d} | {m:.4f} | {p:.4f} | {t:.4f} |  Limiti  |  Limiti  | {1e6:.4e} |")
        return 1e6

    airfoil_name = f"temp_naca.dat"
    X, Y, _ = naca4(m, p, t)
    save_airfoil_to_file(X, Y, airfoil_name)
    
    cl, cd = run_xfoil_analysis(airfoil_name, alpha, Re)
    
    if cl is not None and cd is not None and cd > 0:
        errore_cl = abs(cl - target_cl)
        penalita_cd = (cd - max_cd) * 1000 if cd > max_cd else 0
        punteggio_totale = errore_cl + penalita_cd
        
        print(f"| {eval_count:4d} | {m:.4f} | {p:.4f} | {t:.4f} |  {cl:.4f}  |  {cd:.5f} | {punteggio_totale:.4e} |")
        return punteggio_totale
    else:
        print(f"| {eval_count:4d} | {m:.4f} | {p:.4f} | {t:.4f} | Staccato | Staccato | {1e6:.4e} |")
        return 1e6

# ==============================================================================
# FUNZIONI DI INPUT INTERATTIVO
# ==============================================================================

def get_float_input(prompt, default_val):
    while True:
        val = input(f"{prompt} [predefinito: {default_val}]: ").strip()
        if not val:
            return default_val
        try:
            return float(val)
        except ValueError:
            print("Errore: inserimento non valido. Riprovare.")

# ==============================================================================
# BLOCCO PRINCIPALE DI ESECUZIONE
# ==============================================================================

if __name__ == "__main__":
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    xfoil_exe_path = os.path.join(script_dir, "xfoil.exe")
    
    if not os.path.exists(xfoil_exe_path):
        print("ERRORE CRITICO: Eseguibile 'xfoil.exe' non trovato.")
        print(f"Percorso verificato: {script_dir}")
        exit()

    print("======================================================================")
    print(" OTTIMIZZAZIONE PROFILO ALARE (DESIGN INVERSO)")
    print("======================================================================")
    
    TARGET_REYNOLDS = get_float_input("Numero di Reynolds", 400000.0)
    TARGET_ALPHA = get_float_input("Angolo d'attacco (gradi)", 8.0)
    TARGET_CL = get_float_input("Coefficiente di Portanza (Cl) target", 1.8)
    MAX_CD = get_float_input("Coefficiente di Resistenza (Cd) massimo", 0.15)
    
    # Parametri iniziali (Profilo generico con moderata curvatura, es. NACA 6412)
    initial_guess = [0.06, 0.4, 0.12]
    # Limiti estesi per favorire profili per alto carico aerodinamico
    bounds = [(0.0, 0.20), (0.1, 0.8), (0.05, 0.35)]

    print("\n----------------------------------------------------------------------")
    print(f"Obiettivo: Cl = {TARGET_CL} | Vincolo: Cd <= {MAX_CD}")
    print(f"Condizioni: Re = {TARGET_REYNOLDS}, Alpha = {TARGET_ALPHA}°")
    print("----------------------------------------------------------------------")
    
    # Intestazione della tabella
    print(f"+------+--------+--------+--------+----------+----------+--------------+")
    print(f"| Eval |   m    |   p    |   t    |    Cl    |    Cd    |    Errore    |")
    print(f"+------+--------+--------+--------+----------+----------+--------------+")

    result = minimize(
        objective_function,
        initial_guess,
        args=(TARGET_REYNOLDS, TARGET_ALPHA, TARGET_CL, MAX_CD),
        method='SLSQP',
        bounds=bounds,
        options={
            'disp': True, 
            'maxiter': 2000,
            'ftol': 1e-9,
            'eps': 1e-4
        }
    )

    print("+------+--------+--------+--------+----------+----------+--------------+")
    print("\n======================================================================")
    print(" RISULTATI DELL'OTTIMIZZAZIONE")
    print("======================================================================")
    
    if result.success or result.nfev > 0:
        optimal_params = result.x
        
        # Convalida finale per estrarre i dati esatti dell'ultimo profilo elaborato
        airfoil_name = "final_naca.dat"
        X, Y, coords_optimal = naca4(optimal_params[0], optimal_params[1], optimal_params[2])
        save_airfoil_to_file(X, Y, airfoil_name)
        final_cl, final_cd = run_xfoil_analysis(airfoil_name, TARGET_ALPHA, TARGET_REYNOLDS)
        if os.path.exists(airfoil_name): os.remove(airfoil_name)
        
        m_opt_str = str(int(round(optimal_params[0] * 100)))
        p_opt_str = str(int(round(optimal_params[1] * 10)))
        t_opt_str = f"{int(round(optimal_params[2] * 100)):02d}"
        naca_opt_str = f"{m_opt_str}{p_opt_str}{t_opt_str}"

        print(f"Stato: Completato (Iterazioni: {result.nit}, Valutazioni: {result.nfev})")
        print(f"Punteggio di errore finale: {result.fun:.6f}")
        
        print("\nPRESTAZIONI OTTENUTE:")
        if final_cl is not None and final_cd is not None:
            print(f"Cl: {final_cl:.4f} (Target: {TARGET_CL})")
            print(f"Cd: {final_cd:.5f} (Limite: {MAX_CD})")
            if final_cd > MAX_CD:
                print("Nota: il Cd finale supera il limite imposto per il Cl richiesto.")
        else:
            print("Cl: Dati non disponibili (Flusso staccato o non convergente)")
            print("Cd: Dati non disponibili (Flusso staccato o non convergente)")
            
        print("\nGEOMETRIA TROVATA:")
        print(f"Curvatura massima (m): {optimal_params[0]:.4f}")
        print(f"Posizione curvatura (p): {optimal_params[1]:.4f}")
        print(f"Spessore massimo (t): {optimal_params[2]:.4f}")
        print(f"Profilo NACA Approssimato: NACA {naca_opt_str}")
        print("======================================================================\n")
        
        # Generazione grafico
        plt.figure(figsize=(12, 6))
        plt.plot(coords_optimal[0], coords_optimal[1], 'k-', linewidth=2, label=f'Profilo NACA {naca_opt_str}')
        plt.plot(coords_optimal[2], coords_optimal[3], 'k-', linewidth=2)
        
        # Prevenzione crash se i risultati finali sono vuoti
        if final_cl is not None and final_cd is not None:
            titolo = f"Risultato Ottimizzazione: Cl={final_cl:.3f} | Cd={final_cd:.4f} (Re={TARGET_REYNOLDS}, Alpha={TARGET_ALPHA}°)"
        else:
            titolo = f"Risultato Ottimizzazione: Parametri aerodinamici non convergenti (Re={TARGET_REYNOLDS}, Alpha={TARGET_ALPHA}°)"
            
        plt.title(titolo)
        plt.xlabel('x/c')
        plt.ylabel('y/c')
        plt.axis('equal')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()
        
    else:
        print(f"Ottimizzazione fallita: {result.message}")