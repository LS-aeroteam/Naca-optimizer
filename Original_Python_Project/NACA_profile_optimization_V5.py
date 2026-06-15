import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os
from scipy.optimize import minimize

# ==============================================================================
# FUNZIONI DI GENERAZIONE DEL PROFILO ALARE
# ==============================================================================

def naca4(m_param, p_param, t_param, c=1.0, n=100):
    # Spaziatura a coseno (Cosine Spacing): addensa i punti al bordo d'attacco e d'uscita.
    # Indispensabile per impedire a XFOIL di fallire e disconnettere il flusso a bassi Reynolds!
    beta = np.linspace(0, np.pi, n)
    x = c * (0.5 * (1 - np.cos(beta)))
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

    # Pulisci i file precedenti per evitare che XFOIL si blocchi su "Overwrite (Y/N)?"
    for f in [xfoil_input_file, polar_file]:
        if os.path.exists(f):
            os.remove(f)

    with open(xfoil_input_file, "w") as f:
        f.write(f"LOAD {airfoil_file}\n")
        f.write("PANE\n")  
        f.write("OPER\n")
        
        # TRUCCO 1: Riscaldamento Inviscido (stabilizza lo strato limite)
        f.write("ALFA 0\n") 
             
        f.write(f"Visc {Re}\n")
        f.write(f"Mach {Mach}\n")
        f.write("ITER 500\n") 
        
        # Attiviamo PACC *prima* dello sweep per assicurarci di registrarli tutti
        f.write("PACC\n")
        f.write(f"{polar_file}\n\n") 
        
        if alpha == 0.0:
            f.write("ALFA 0.0\n")
        else:
            step = 1.0 if alpha > 0 else -1.0
            # Rampa di Sweep usando ASEQ per robustezza
            f.write(f"ASEQ 0.0 {alpha - step/2} {step}\n")
            # Forza l'ultimo angolo esatto
            f.write(f"ALFA {alpha}\n")
            
        f.write("\n")      
 
        f.write("QUIT\n")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    xfoil_exe_path = os.path.join(script_dir, "xfoil.exe")

    try:
        with open(xfoil_input_file, "r") as stdin_file:
            subprocess.run([xfoil_exe_path], stdin=stdin_file, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        # Se va in timeout o fallisce, passiamo comunque a leggere i dati parziali recuperati!
        pass

    cl, cd, achieved_alpha = None, None, None
    try:
        with open(polar_file, "r") as f:
            lines = [line for line in f if not line.startswith("#") and len(line.strip()) > 0]
            best_diff = 1e6
            for line in lines:
                data = line.split()
                if len(data) >= 3:
                    try:
                        a_val = float(data[0])
                        cl_val = float(data[1])
                        cd_val = float(data[2])
                        
                        diff = abs(a_val - alpha)
                        if diff < best_diff:
                            best_diff = diff
                            achieved_alpha = a_val
                            cl = cl_val
                            cd = cd_val
                    except ValueError:
                        pass
    except (IOError, IndexError):
        pass
    
    for f in [xfoil_input_file, polar_file, airfoil_file]:
        if os.path.exists(f):
            os.remove(f)
            
    return cl, cd, achieved_alpha

# ==============================================================================
# CALCOLO REYNOLDS E FUNZIONE OBIETTIVO
# ==============================================================================

def calcola_reynolds(velocita, corda, viscosita_cinematica):
    return (velocita * corda) / viscosita_cinematica

def objective_function(params, Re, alpha, target_cl, max_cd):
    objective_function.eval_count += 1
    eval_count = objective_function.eval_count
    m, p, t = params
    
    if not (0.0 <= m <= 0.20 and 0.1 <= p <= 0.8 and 0.05 <= t <= 0.35):
        print(f"| {eval_count:4d} | {m:.4f} | {p:.4f} | {t:.4f} |  Limiti  |  Limiti  | {1e6:.4e} |")
        return 1e6

    airfoil_name = f"temp_naca.dat"
    X, Y, _ = naca4(m, p, t)
    save_airfoil_to_file(X, Y, airfoil_name)
    
    cl, cd, achieved_alpha = run_xfoil_analysis(airfoil_name, alpha, Re)
    
    if cl is not None and cd is not None and cd > 0:
        # Utilizziamo l'errore quadratico invece di abs() per rendere la funzione "liscia" (derivabile) per SLSQP
        errore_cl = ((cl - target_cl) * 10) ** 2
        penalita_cd = (max(0, cd - max_cd) * 1000) ** 2
        
        # Se XFOIL perde aderenza (stalla) PRIMA di arrivare all'angolo richiesto:
        if abs(achieved_alpha - alpha) > 0.1:
            penalita_alfa = (abs(achieved_alpha - alpha) * 1000) ** 2
            punteggio_totale = errore_cl + penalita_cd + penalita_alfa
            print(f"| {eval_count:4d} | {m:.4f} | {p:.4f} | {t:.4f} | {cl:.4f}*  | {cd:.5f}* | {punteggio_totale:.4e} |")
            return punteggio_totale
            
        punteggio_totale = errore_cl + penalita_cd
        
        print(f"| {eval_count:4d} | {m:.4f} | {p:.4f} | {t:.4f} |  {cl:.4f}  |  {cd:.5f} | {punteggio_totale:.4e} |")
        return punteggio_totale
    else:
        # Gradiente d'emergenza: Se fallisce perfino a 0°, crea una pendenza verso ali più spesse e stabili
        emergenza = 1e6 + ((0.12 - t)**2 * 1e5) + ((0.05 - m)**2 * 1e5)
        print(f"| {eval_count:4d} | {m:.4f} | {p:.4f} | {t:.4f} | Staccato | Staccato | {emergenza:.4e} |")
        return emergenza

# ==============================================================================
# FUNZIONI DI INPUT INTERATTIVO
# ==============================================================================

def get_float_input(prompt):
    while True:
        val = input(f"{prompt}: ").strip()
        try:
            return float(val)
        except ValueError:
            print("Errore: inserimento non valido. È richiesto un valore numerico.")

def get_fluid_selection():
    fluids = {
        '1': {'nome': 'Aria (Standard SL)', 'viscosita': 1.46e-5},
        '2': {'nome': 'Acqua (20 gradi)', 'viscosita': 1.00e-6}
    }
    
    while True:
        print("\nSeleziona il fluido operativo:")
        print("1. Aria (Viscosita' cinematica: 1.46e-5 m^2/s)")
        print("2. Acqua (Viscosita' cinematica: 1.00e-6 m^2/s)")
        scelta = input("Scelta (1 o 2): ").strip()
        
        if scelta in fluids:
            return fluids[scelta]
        print("Errore: selezione non valida.")

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
    
    # Acquisizione parametri operativi per calcolo Reynolds
    fluido = get_fluid_selection()
    velocita = get_float_input("Inserisci la velocita' di progetto (m/s)")
    corda = get_float_input("Inserisci la lunghezza della corda alare (m)")
    
    TARGET_REYNOLDS = calcola_reynolds(velocita, corda, fluido['viscosita'])
    
    # Acquisizione parametri di ottimizzazione
    TARGET_ALPHA = get_float_input("\nInserisci l'angolo d'attacco (gradi)")
    TARGET_CL = get_float_input("Inserisci il Coefficiente di Portanza (Cl) target")
    MAX_CD = get_float_input("Inserisci il Coefficiente di Resistenza (Cd) massimo tollerato")
    
    # Partenza da un profilo "quasi piatto" (m=0.01) invece di m=0.0 per evitare il blocco sui gradienti di "p"
    initial_guess = [0.01, 0.4, 0.12]
    bounds = [(0.0, 0.20), (0.1, 0.8), (0.05, 0.35)]

    print("\n----------------------------------------------------------------------")
    print(f"Fluido: {fluido['nome']} | Velocita': {velocita} m/s | Corda: {corda} m")
    print(f"Numero di Reynolds Calcolato: {TARGET_REYNOLDS:.1f}")
    print(f"Obiettivo: Cl = {TARGET_CL} | Vincolo: Cd <= {MAX_CD} a {TARGET_ALPHA} gradi")
    print("Profilo iniziale: NACA 1412 (Quasi-Simmetrico)")
    print("----------------------------------------------------------------------")
    
    print(f"+------+--------+--------+--------+----------+----------+--------------+")
    print(f"| Eval |   m    |   p    |   t    |    Cl    |    Cd    |    Errore    |")
    print(f"+------+--------+--------+--------+----------+----------+--------------+")

    objective_function.eval_count = 0

    result = minimize(
        objective_function,
        initial_guess,
        args=(TARGET_REYNOLDS, TARGET_ALPHA, TARGET_CL, MAX_CD),
        method='SLSQP',
        bounds=bounds,
        options={
            'disp': True, 
            'maxiter': 2000,
            'ftol': 1e-4, # Tolleranza abbassata: tiene conto del rumore numerico intrinseco di XFOIL
            'eps': 1e-4
        }
    )

    print("+------+--------+--------+--------+----------+----------+--------------+")
    print("\n======================================================================")
    print(" RISULTATI DELL'OTTIMIZZAZIONE")
    print("======================================================================")
    
    if result.success or result.nfev > 0:
        optimal_params = result.x
        
        airfoil_name = "final_naca.dat"
        X, Y, coords_optimal = naca4(optimal_params[0], optimal_params[1], optimal_params[2])
        save_airfoil_to_file(X, Y, airfoil_name)
        final_cl, final_cd, final_alpha = run_xfoil_analysis(airfoil_name, TARGET_ALPHA, TARGET_REYNOLDS)
        if os.path.exists(airfoil_name): os.remove(airfoil_name)
        
        m_opt_str = str(int(round(optimal_params[0] * 100)))
        p_opt_str = str(int(round(optimal_params[1] * 10)))
        t_opt_str = f"{int(round(optimal_params[2] * 100)):02d}"
        naca_opt_str = f"{m_opt_str}{p_opt_str}{t_opt_str}"

        print(f"Stato: Completato (Iterazioni: {result.nit}, Valutazioni: {result.nfev})")
        print(f"Punteggio di errore finale: {result.fun:.6f}")
        
        print("\nPRESTAZIONI OTTENUTE:")
        if final_cl is not None and final_cd is not None:
            if abs(final_alpha - TARGET_ALPHA) > 0.1:
                print(f"Cl: {final_cl:.4f} (Ottenuto allo stallo a {final_alpha}° invece di {TARGET_ALPHA}°)")
                print(f"Cd: {final_cd:.5f}")
                print("ATTENZIONE: Il profilo stalla prima di raggiungere l'angolo richiesto.")
            else:
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
        
        plt.figure(figsize=(12, 6))
        plt.plot(coords_optimal[0], coords_optimal[1], 'k-', linewidth=2, label=f'Profilo NACA {naca_opt_str}')
        plt.plot(coords_optimal[2], coords_optimal[3], 'k-', linewidth=2)
        
        if final_cl is not None and final_cd is not None:
            if abs(final_alpha - TARGET_ALPHA) > 0.1:
                titolo = f"Risultato (Stallo a {final_alpha}°): Cl={final_cl:.3f} | Cd={final_cd:.4f} (Re={TARGET_REYNOLDS:.1f})"
            else:
                titolo = f"Risultato Ottimizzazione: Cl={final_cl:.3f} | Cd={final_cd:.4f} (Re={TARGET_REYNOLDS:.1f}, Alpha={TARGET_ALPHA}°)"
        else:
            titolo = f"Risultato Ottimizzazione: Parametri non convergenti (Re={TARGET_REYNOLDS:.1f}, Alpha={TARGET_ALPHA}°)"
            
        plt.title(titolo)
        plt.xlabel('x/c')
        plt.ylabel('y/c')
        plt.axis('equal')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()
        
    else:
        print(f"Ottimizzazione fallita: {result.message}")