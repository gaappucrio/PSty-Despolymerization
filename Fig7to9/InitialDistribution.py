import numpy as np 
import matplotlib.pyplot as plt
def Moleculas(wx, Na, MW, p_polymer, V):
    D =  np.zeros(len(wx), dtype=np.int64)
    for i in range(0, len(wx)):
        if wx[i] > 0: D[i] = ((Na)*wx[i]/(i*MW))   # molecules/mol * g * mol/g = molecules
    

    # 
    # carbons = np.arange(0, len(wx), 1)
    # plt.figure()
    # plt.plot(carbons, wx)

    Msample =  1
    Vsample =  Msample/p_polymer
    D       =  D*V/Vsample


    D     =  np.zeros(len(wx), dtype=np.int64)
    # D[50] = 93670
    D[280] = 93670

    lambda2, lambda1, lambda0  = 0,0,0
    for i in range(len(wx)):
        lambda2 += i*i*D[i]
        lambda1 += i*D[i]
        lambda0 += D[i]
        
    Mn0 = lambda1*MW/lambda0
    Mw0 = lambda2*MW/lambda1
    print("Mn: ", Mn0)
    print("Mw: ", Mw0)
    print("Where maximum is (Mn)", np.argmax(D))
    print("Molecules in maximum length", D[np.argmax(D)])


    print("D[1]", D[1])
    Dplotar = np.copy(D)
    D = np.append(D, Dplotar*0, axis=0)
    D = np.rint(D)
    Styrene_t0 = 0.0 # D[1]
    D[:11] = 0        # D[:2] = 0
    P0    = np.sum(D[2:])
    print("P0", P0)
    print("Soma D", D.sum())

    carbons = np.arange(0, len(D), 1)
    plt.figure()
    plt.plot(carbons, D)
    plt.show()

    xPlotar =    np.linspace(0,len(D),num=len(D)) 
    wx_PS   =    np.zeros(len(xPlotar)) 
    for i in range(0, 6000):
        if Dplotar[i] > 0:  wx_PS[i] = D[i]*(Vsample/V)*(i*MW)/Na  # NOT DEGRADADED
    print("Area PS original:",  np.trapz(wx_PS   , x=xPlotar))
    
    # --------------------------------------------------------------------------------------
    
    
    # INITIAL CONDITIONS (MOLECULES)
    n = 31  # number of species in the system
    X = np.zeros(n, dtype=np.int64)  # X = np.zeros(n, dtype=np.uint64)

    X[0] = P0                  # Dead Polymer Saturated 
    X[1] = 0                   # Dead Polymer with 1 Unsaturated End # obs.: não sei se a melhor estratégia é criar essa nova espécie ou um vetor que registre o número de insaturações
    X[2] = 0                   # Dead Polymer with 2 Unsaturated End 
    X[3] = 0                   # Polymer Radical (Secondary)
    X[4] = 0                   # Polymer Radical (Primary)
    X[5] = 0                   # Polymer mid-chain radical
    X[6] = 0                   # Mid-chain 13 radical
    X[7] = 0                   # Mid-chain 15 radical
    X[8] = 0                   # Mid-chain 17 radical
    X[9] = 0                   # Allyl benzene radical
    X[10] = 0                  # Benzyl radical =  Tolune radical (TOLradical)
    X[11] = 0                  # DPP radical 
    X[12] = Styrene_t0         # Monomer
    X[13] = 0                  # Alfa-metil estireno
    X[14] = 0                  # Dimer
    X[15] = 0                  # DPP = 1,3-Diphenylpropane
    X[16] = 0                  # Trimer 
    X[17] = 0                  # Ethylbenzene radical
    X[18] = 0                  # Ethylbenzene
    X[19] = 0                  # Toluene
    X[20] = 0                  # Styryl radical (very unlikely) # not being used
    X[21] = 0                  # 2,4-diphenyl-1,4-pentadiene (Dimer with two unsaturations)

    X[22] = 0                  # Styrene radical (1-Phenylethyl radical)
    X[23] = 0                  # Dimer radical (thermal initiation)
    
    
    X[24] = 0                  # Toluene evaporated
    X[25] = 0                  # Ethylbenzene evaporated
    X[26] = 0                  # Styrene evaporated
    X[27] = 0                  # aMS evaporated
    X[28] = 0                  # Dimer evaporated
    X[29] = 0                  # DPP evaporated
    X[30] = 0                  # Trimer evaporated
        

    # 2007: Chemical Recycling of Polystyrene by Pyrolysis: Potential Use of the Liquid Product for the Reproduction of Polymer
    # MISSING: ethylbenzene (maybe 2,2-Diphenylpropane)
    # Missing as well: indane/indene; 1,2-diphenylETHANE, 1-methyl-1,2-diphenylethane, 1,1-diphenylpropene
    # Gas phase: methane, ethylene and propylene are also formed

    # ---------- DEFININDO AS TAXAS DE REACOES DE MONTE CARLO
    nr = 54 # number of reactions
    R = np.zeros(nr, dtype=np.float64)

    return D,Dplotar,P0,Styrene_t0,X,R,Mn0,Mw0,Vsample,xPlotar,wx_PS