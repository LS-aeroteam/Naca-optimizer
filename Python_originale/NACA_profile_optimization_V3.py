import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os
from scipy.optimize import minimize

# ==============================================================================
# FUNZIONI DI GENERAZIONE DEL PROFILO ALARE
# ==============================================================================

def naca4(m_param, p_param, t_param, c=1.0, n=100):
    """Genera le coordinate di un profilo alare NACA a 4 cifre da parametri."""
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

   # Leggi i risultati dal file polare
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
                        # XFOIL non è asintoticamente convergente e ha stampato '--------'
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
    """
    Minimizza l'errore sul Cl desiderato. Se il Cd supera il limite,
    assegna una multa (penalità) all'algoritmo.
    """
    m, p, t = params
    
   # PRIMA ERA: if not (0.0 <= m < 0.1 and 0.1 <= p < 0.8 and 0.05 <= t < 0.25):
    if not (0.0 <= m < 0.20 and 0.1 <= p < 0.8 and 0.05 <= t < 0.35):
        return 1e6

    airfoil_name = f"temp_naca.dat"
    X, Y, _ = naca4(m, p, t)
    save_airfoil_to_file(X, Y, airfoil_name)
    
    cl, cd = run_xfoil_analysis(airfoil_name, alpha, Re)
    
    if cl is not None and cd is not None and cd > 0:
        # 1. Calcola l'errore sul Cl (distanza dal target)
        errore_cl = abs(cl - target_cl)
        
        # 2. Applica penalità sul Cd se viene superato il limite
        penalita_cd = 0
        if cd > max_cd:
            # Moltiplichiamo per 1000 per rendere la penalità severa
            penalita_cd = (cd - max_cd) * 1000 
            
        punteggio_totale = errore_cl + penalita_cd
        
        print(f"Test (m={m:.4f}, p={p:.4f}, t={t:.4f}) -> Cl: {cl:.4f}, Cd: {cd:.5f} | Errore: {punteggio_totale:.6f}")
        return punteggio_totale
    else:
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
            print("  Errore: Inserisci un numero valido.")

# ==============================================================================
# BLOCCO PRINCIPALE DI ESECUZIONE
# ==============================================================================

if __name__ == "__main__":
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    xfoil_exe_path = os.path.join(script_dir, "xfoil.exe")
    
    if not os.path.exists(xfoil_exe_path):
        print("="*60)
        print("!!! ERRORE CRITICO: Eseguibile 'xfoil.exe' non trovato.")
        print(f"Assicurati che XFOIL sia presente in questa cartella:\n{script_dir}")
        print("="*60)
        exit()

    print("\n" + "="*60)
    print(" 🎯 STRUMENTO DI DESIGN AERODINAMICO INVERSO (MODALITA' PROFONDA)")
    print("="*60)
    
    TARGET_REYNOLDS = get_float_input("1. Inserisci il numero di Reynolds", 1000000.0)
    TARGET_ALPHA = get_float_input("2. Inserisci l'angolo d'attacco in gradi", 2.0)
    TARGET_CL = get_float_input("3. Inserisci il Coefficiente di Portanza (Cl) desiderato", 0.5)
    MAX_CD = get_float_input("4. Inserisci il Coefficiente di Resistenza (Cd) massimo", 0.015)
    
    # Parametri iniziali nascosti all'utente
    initial_guess = [0.02, 0.4, 0.12]
    bounds = [(0.0, 0.09), (0.1, 0.8), (0.05, 0.25)]

    print("\n" + "-" * 50)
    print("--- INIZIO RICERCA PROFILO ---")
    print(f"Obiettivo: Cl = {TARGET_CL} | Vincolo: Cd <= {MAX_CD}")
    print(f"Condizioni: Re = {TARGET_REYNOLDS}, Alpha = {TARGET_ALPHA}°")
    print("L'algoritmo esplorerà in profondità. Potrebbe richiedere diversi minuti...")
    print("-" * 50)

    # --- MODIFICA CHIAVE QUI ---
    # maxiter aumentato a 2000, ftol ridotto a 1e-9 per una precisione estrema
    result = minimize(
        objective_function,
        initial_guess,
        args=(TARGET_REYNOLDS, TARGET_ALPHA, TARGET_CL, MAX_CD),
        method='SLSQP',
        bounds=bounds,
        options={
            'disp': True, 
            'maxiter': 2000,   # Consente fino a 2000 cicli
            'ftol': 1e-9,      # Cerca miglioramenti anche infinitesimali
            'eps': 1e-4        # Passo di campionamento per il calcolo delle derivate
        }
    )

    print("\n" + "-" * 50)
    print("--- RISULTATI ---")
    
    if result.success or result.nfev > 0:
        optimal_params = result.x
        
        # Analisi finale di convalida
        airfoil_name = "final_naca.dat"
        X, Y, coords_optimal = naca4(optimal_params[0], optimal_params[1], optimal_params[2])
        save_airfoil_to_file(X, Y, airfoil_name)
        final_cl, final_cd = run_xfoil_analysis(airfoil_name, TARGET_ALPHA, TARGET_REYNOLDS)
        if os.path.exists(airfoil_name): os.remove(airfoil_name)
        
        m_opt_str = str(int(round(optimal_params[0] * 100)))
        p_opt_str = str(int(round(optimal_params[1] * 10)))
        t_opt_str = f"{int(round(optimal_params[2] * 100)):02d}"
        naca_opt_str = f"{m_opt_str}{p_opt_str}{t_opt_str}"

        print(f"✅Ricerca terminata (Iterazioni effettuate: {result.nit} | Punteggio errore: {result.fun:.6f})")
        
        if final_cl is not None:
            print("\nPRESTAZIONI OTTENUTE:")
            print(f"  Cl Raggiunto: {final_cl:.4f} (Target era {TARGET_CL})")
            print(f"  Cd Raggiunto: {final_cd:.5f} (Limite era {MAX_CD})")
            
            if final_cd > MAX_CD:
                print("  ⚠️ATTENZIONE: L'algoritmo ha esaurito i tentativi ma non è riuscito a scendere sotto il Cd massimo richiesto per questo Cl. (Forse stai chiedendo troppo alla fisica!)")
            
        print("\nGEOMETRIA TROVATA:")
        print(f"  Curvatura max (m): {optimal_params[0]:.4f}")
        print(f"  Posizione curvatura (p): {optimal_params[1]:.4f}")
        print(f"  Spessore max (t): {optimal_params[2]:.4f}")
        print(f"  Sigla NACA Approssimata: {naca_opt_str}")
        
        # Grafico
        plt.figure(figsize=(12, 6))
        plt.plot(coords_optimal[0], coords_optimal[1], 'r-', label=f'Profilo Generato (~NACA {naca_opt_str})')
        plt.plot(coords_optimal[2], coords_optimal[3], 'r-')
        
        plt.title(f"Design Inverso (Analisi Profonda): Cl Raggiunto={final_cl:.3f} | Cd={final_cd:.4f}")
        plt.xlabel('x/c')
        plt.ylabel('y/c')
        plt.axis('equal')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    else:
        print("\n❌Ottimizzazione fallita.")
        print(f"Causa: {result.message}")