
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from numba import jit

@jit(nopython=True)  # @jit(nopython=True, cache=True) 
def KineticConstants(RT):
    # kdm    = 1.314e7 * np.exp(-27.440/(Rcte * T))          # styrene thermal initiation (L2/(mol2 min))
    # kdm    = 7.884e8/60  * np.exp(-27.440/(Rcte * T))
    kdm    = 2.19*10**5*np.exp(-27.440/(Rcte * T))
    
    print("kdm:", kdm)
    k11    = 6.36e8 * np.exp(-7067.0/(Rcte * T))            # propagation constant (L/(mol min))

    
    # ---------- KINETIC CONSTANTS - MACROSCOPIC
    # kf0 = 1e15*np.exp(-67.3/(RT))   # Levine et al. (2008) Mid chain scisson (or 'chain fission') (s-1) (Dn -> Pr + Pn-r) (unimolecular reactions)
    # kf0 = 1e16*np.exp(-67.3/(RT))         # Kruse et al., 2002
    kf0   = 2*10**(16.2)*np.exp(-72.9/(RT)) # Poutsma 2006


    # kfs0 = 5.5e13*np.exp(-57.3/(RT))  # Levine et al. (2008) (= Kruse et al., 2002) End chain scisson (or chain fission allyl) (carbon-hydrogen bond fission) (s-1) (Dn -> M + Pn-1) (unimolecular reactions)
    kfs0 = 10**(15.6)*np.exp(-61.7/(RT))    # Poutsma 2006 (aMS saiu de 0.04% com Kruse2002 para 0.13%!)


    # kH0  = 2.10e6*np.exp(-10.5/(RT))       # Levine (2008). Hydrogen abstraction (L/mol*seg)   (bimolecular reactions between different molecules) # = Kruse et al., 2002  # Aumenta a concentração de Pmorto
    kH0    = 10**(8.10)*np.exp(-15.85/(RT))    # POUTSMA 2006 (SB, TB) = POUTSMA 2009 (FICA BEM MELHOR!!!)
    kH0_Rp = 10**8*np.exp(-14.35/(RT))       # POUTSMA 2009 (PB, TB) 
    # kH0_Rp = 10**(7.6)*np.exp(-7.8/(RT))   # POUTSMA 2006 (TABLE 6, NÃO ENTENDO PQ ESSE VALOR É DIFERENTE DE TABLE 2)
        # kH(pb,tb) ----> TOLrad to TOLUENE
        # kH(tb,tb) ----> NOT CONSIDERED

    # kdp0 = 2.8e13*np.exp(-26.4/(RT))  # Levine et al. (2008) ---- Depropagation (end chain β-scission) (s-1) (unimolecular reactions)
    kdp0 = 3.1e12*np.exp(-23.9/(RT))  # Depropagation (end chain β-scission) (s-1) (unimolecular reactions)
        # Levine et al. (2008) considers 2 reactions: 
        #   i) end-chain beta-scission; 
        #   ii) depropagation (the formation of monomer from low molecular weight radicals with chain length of five carbon atoms or less)
        # But I'm considering only one using the end-chain beta-scission expression 
    ## kdp0 = 4.1e12*np.exp(-24.7/(RT))       # Kruse et al., 2002
    # kdp0 = 10**13.37*np.exp(-24.20/(RT))    # POUTSMA, 2009           94.83
    # !!!!!!!!! era esse # kdp0 = (10**13)*np.exp(-25.20/(RT))     # POUTSMA, 2009 --> kb,pm(sb,sb)

    kdp_Rp0 = 10**(12.9)*np.exp(-24.9/(RT)) # Poutsma 2006 - Depropagation via Rp 


    # kbs0 = 3.10e12*np.exp(-27.3/(RT))       # Mid-chain beta scission
    # kbs0 = 4.10e12*np.exp(-28.1/(RT))       # Mid-chain beta scission
    # kbs0 =  1e14*np.exp(-24.3/(RT))         # Poutsma 2006
    # kbs0 =  10**(13.91)*np.exp(-26.2/(RT))   # Poutsma 2006
    kbs0 =  10**(13.28)*np.exp(-24.24/(RT))  # Poutsma 2009 (kb,pm(tb,sb)) (tem também kb,pm(tb,pb) -> mas a diferença parece pequena)


    kbs0_LMWS = 1.25e13*np.exp(-26.1/(RT))       # Mid-chain beta scission to LMWS <<
    kbs0_LMWS = (10**13.65)*np.exp(-23.16/(RT))  # Poutsma 2009 (kb,pm(tb,sb)) 


    kc0  = 1.1e11*np.exp(-2.3/(RT)) # Levine et al. (2008) (= Kruse et al., 2002) Termination by combination (radical recombination) (L/mol*seg) (bimolecular reactions between equal molecules)
    ktd0 = 5.5e9*np.exp(-2.3/(RT))  # Levine et al. (2008) (= Kruse et al., 2002) Termination by disproportionation (L/mol*seg) (bimolecular reactions between equal molecules)


    k130 = 5.01e12*np.exp(-37.4/(RT))        # Levine et al., 2008 
    ## k130 = 4.5e11*np.exp(-23.5/(RT))        # Kruse et al., 2002
    # k130 = 1.1220000e+05    # ESTIMACAO
    # k130 = 10**(11.7)*np.exp(-23.5/(RT))     # Poutsma 2006
    k130 = 10**(12.7)*np.exp(-37.4/(RT))       # Poutsma 2009
    # k130 = 0    # Dá 3.34% de dimer (k170 e k730); ou 0.66% com a constante proposta de Poutsma (k170 = 0.1*k150)
                # Com todas as ctes de Poutsma dá 1.78% de dimer e (3.49% de toluene)

    ## 1,5 --- REACTION THAT LEADS TO TRIMER
    ## The Broadbelt constant (Levine or Kruse) is smaller than the Poutsma one
    ## But the experimental data used by Broadbelt was obtained in a closed system and got a higher amount of dimer
    ## This comparision can be seen between Figure 1 and Fig. 2. of Poutsma 2009
    ## k150 = 1.35e9*np.exp(-16.2/(RT))         # Levine et al., 2008
    ## k150 = 5.0e6*np.exp(-10.5/(RT))       # Kruse et al., 2002
    k150 = 10**(9.75)*np.exp(-16.28/(RT))      # POUTSMA, 2009
    ## k150 = 10**9.13*np.exp(-16.2/(RT))       # tbm POUTSMA, 2009
    ## k150 = 10**9.79*np.exp(-16.63/(RT))       # tbm POUTSMA, 2009

    # k170 = 1.02e9*np.exp(-15.7/(RT))
    k170 = 0.1*k150    # According to Poutsma k17 << k15. 
    # k170 = 0.0

    k730 = 6.31e9*np.exp(-16.6/(RT))    # Levine, 2009 (considered the same by Poutsma, 2009)
    # k730 = 0.0

    kra0 = 4.3e7*np.exp(-6.4/(RT)) # General radical addition (opposite of depropagation) (Pr + Dr -> Mid-chain radical, Pr + M -> Pr+1)
    kra0 = 10**(8)*np.exp(-6.94/(RT))  # Poutsma, 2009

    # General radical addition is different than termination by combination (because termination is termination, this one is like a propagation)
    kbra0  = 2.75e8*np.exp(-4.3/(RT))        # Levine, 2008. 
    kbra0  = 10**(8.13)*np.exp(-8.62/(RT))   # Poutsma, 2009: Benzyl radical addition: TolRAD + PSunsat -> Rtb (1,3)  [kadd(pb,tb)]
    kdppa0 = 10**(7.63)*np.exp(-7.92/(RT))  # Poutsma, 2009: dppRAD + PSunsat -> Rtb (1,5)  [kadd(sb,tb)]> 10**(7.63)*np.exp(-7.92/(RT))
    return kf0, kfs0, kH0, kH0_Rp, kdp0, kdp_Rp0, kbs0, kbs0_LMWS, kc0, ktd0, k130,k150, k170, k730, kra0, kbra0, kdppa0, kdm, k11

def Moleculas(wx, Na, MW, p_polymer, V):

    D =  np.zeros(len(wx), dtype=np.int64)
    for i in range(0, len(wx)):
        if wx[i] > 0: D[i] = ((Na)*wx[i]/(i*MW))   # molecules/mol * g * mol/g = molecules

    Msample = 1
    Vsample =  Msample/p_polymer
    D = D*V/Vsample

    lambda2, lambda1, lambda0  = 0,0,0
    for i in range(len(wx)):
        lambda2 += i*i*D[i]
        lambda1 += i*D[i]
        lambda0 += D[i]
        
    Mn0 = lambda1*MW/lambda0
    Mw0 = lambda2*MW/lambda1
    # print("Mn: ", Mn0)
    # print("Mw: ", Mw0)
    # print("Where maximum is (Mn)", np.argmax(D))
    # print("Molecules in maximum length", D[np.argmax(D)])


    Dplotar = np.copy(D)
    D = np.append(D, Dplotar*0, axis=0)
    D = np.rint(D)
    Styrene_t0 = D[1]
    D[:2] = 0
    P0 = np.sum(D[2:])
    # print("P0", P0)
    # print("Soma D", D.sum())

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

@jit(nopython=True)  # @jit(nopython=True, cache=True) 
def MC(V, Na, RT, T, t,tend,time_interval,X,R,D):
    kf0, kfs0, kH0, kH0_Rp, kdp0, kdp_Rp0, kbs0, kbs0_LMWS, kc0, ktd0, k130,k150, k170, k730, kra0, kbra0, kdppa0,kdm, k11 = KineticConstants(RT)

    P        = []   # Living Polymer (Secondary)
    Pprimary = []   # Living Polymer (Primary)
    MID      = []
    RAD13    = []
    RAD15    = []
    RAD17    = []


    # Lists to follow the number of unsaturations in the tail ends
    P1_Unsat    = [] # Polymer Radical (Primary) 
    P2_Unsat    = [] # Polymer Radical (Secondary)
    MID_Unsat   = []
    RAD13_Unsat = []
    RAD15_Unsat = []
    RAD17_Unsat = []

    ### Arrays that will be registering the size of dead polymers with at least one unsaturation in a tail end
    # D1Unsat = np.zeros(len(D), dtype=np.int64)  
    # D2Unsat = np.zeros(len(D), dtype=np.int64)  
    D1Unsat = [int(0)]*len(D)
    D2Unsat = [int(0)]*len(D)


    # Lists that conts the number of C-C bonds in the dead polymer chains 
    Dsoma = int(0)  # (np.uint32) meli
    for i in range(1, len(D),1): Dsoma += (i*(D[i])) # Dsoma != D.sum()
    SomaD1Unsat = int(0)
    SomaD2Unsat = int(0)

    

    reacao = "nenhuma"
    explicacao = "nenhuma"

    ######
    # Styrene     = np.zeros(len(time_interval), dtype=np.int64)     
    # DeadPolymer = np.zeros(len(time_interval), dtype=np.int64) 
    # Rs          = np.zeros(len(time_interval), dtype=np.int64)
    # Rp          = np.zeros(len(time_interval), dtype=np.int64)
    # Rm          = np.zeros(len(time_interval), dtype=np.int64)
    # Dimer       = np.zeros(len(time_interval), dtype=np.int64)
    # Trimer      = np.zeros(len(time_interval), dtype=np.int64)
    # # PSalkane    = np.zeros(len(time_interval), dtype=np.float64)
    # # PSalkene    = np.zeros(len(time_interval), dtype=np.float64)
    # # PSdialkene  = np.zeros(len(time_interval), dtype=np.float64)
    # Mn          = np.zeros(len(time_interval), dtype=np.int64)
    # Mw          = np.zeros(len(time_interval), dtype=np.int64)
    ###
    # Styrene     = [(0)]*len(time_interval) 
    # DeadPolymer = [(0)]*len(time_interval) 
    # Rs          = [(0)]*len(time_interval) 
    # Rp          = [(0)]*len(time_interval) 
    # Rm          = [(0)]*len(time_interval) 
    # Dimer       = [(0)]*len(time_interval) 
    # Trimer      = [(0)]*len(time_interval) 
    # Mn          = [(0)]*len(time_interval) 
    # Mw          = [(0)]*len(time_interval) 
    ###
    Styrene     = np.zeros(len(time_interval))
    DeadPolymer = np.zeros(len(time_interval))
    Rs          = np.zeros(len(time_interval)) 
    Rp          = np.zeros(len(time_interval))
    Rm          = np.zeros(len(time_interval))
    Dimer       = np.zeros(len(time_interval)) 
    Trimer      = np.zeros(len(time_interval)) 
    Mn          = np.zeros(len(time_interval)) 
    Mw          = np.zeros(len(time_interval)) 

    
    SumBonds   = np.zeros(len(time_interval)) # [(0)]*len(time_interval)   # np.zeros(len(time_interval), dtype=np.int64)
    sumCCbonds = (Dsoma + SomaD1Unsat + SomaD2Unsat) 
    SumBonds[0] = sumCCbonds
    DeadPolymer[0] = D.sum()





    j, l, i = int(0), int(0), int(0)
    erro = (0)
    
    ll = int(1)      # A variable to register the information according to a defined time interval 
    SomaD = (0) # A variable that is used in the routine to found the size of the polymer with that specific C-C bond
    maior_que_DP = 0    # A variable that register the number of dead polymer chains that eventually has a more meres than DP
    tfim = 0  
    
    t_temp_thermalinit = 0
    

    R[23] = kf*Dsoma

    dog1, dog2, dog3, dog4, dog5, dog6, dog7, dog8, dog9, dog10 = 0, 0,0,0,0,0,0,0,0,0
    dog11, dog12, dog13, dog14, dog15, dog16, dog17, dog18, dog19, dog20 = 0, 0,0,0,0,0,0,0,0,0
    dog21, dog22, dog23, dog24, dog25, dog26, dog27, dog28, dog29, dog30 = 0, 0,0,0,0,0,0,0,0,0
    dog31, dog32, dog33, dog34, dog35, dog36, dog37, dog38, dog39, dog40 = 0, 0,0,0,0,0,0,0,0,0
    dog41, dog42, dog43, dog44, dog45, dog46, dog47, dog48 = 0, 0,0,0,0,0,0,0

    X3length = int(0)

    while t < tend:
        R0 = np.sum(R) # sum of all Reaction rates
        dt = 1.0/(R0)*np.log(1.0/np.random.rand()) # Gillespie traditional way
        rnd = np.random.uniform(0.0, 1.0)*R0    # Random Number

        if   rnd <= R[0]:            # Mid-chain beta scission of lengthy polymer
            reacao = "R[9]"
            dog1 += 1
            ## MID[i][0]: tamanho da cadeia
            ## MID[i][1]: onde tá o radical intermediário
            i = int(np.random.rand()*X[5])

            if int(MID[i][1]) == int(1): 
                X[10] += 1   # TOLradical
                if int(MID[i][0] - MID[i][1]) == 1: 
                    X[12] += 1   # I had made a restriction that a polymer with MID chain radical must have more than 3 meres
                else:
                    if MID_Unsat[i] == 0:
                        D1Unsat[int(MID[i][0] - MID[i][1])] += 1
                        SomaD1Unsat += int(MID[i][0] - MID[i][1])
                        X[1] += 1
                    elif MID_Unsat[i] == 1:
                        D2Unsat[int(MID[i][0] - MID[i][1])] += 1 
                        SomaD2Unsat += int(MID[i][0] - MID[i][1])
                        X[2] += 1
                    else:
                        X[10] -= 1
                        X[20] += 1  # Styryl radical (very unlikely)
                        D2Unsat[int(MID[i][0] - MID[i][1])] += 1 
                        SomaD2Unsat += int(MID[i][0] - MID[i][1])
                        X[2] += 1                      
            else: 
                P.append(MID[i][1])
                X[3] += 1   # POLYMER RADICAL (SECONDARY)
                P2_Unsat.append(0)
                X3length += MID[i][1]

                if int(MID[i][0] - MID[i][1]) == 1: 
                    X[12] = X[12] + 1   # I had made a restriction that a polymer with MID chain radical must have more than 3 meres
                else:
                    if MID_Unsat[i] == int(0):
                        D1Unsat[int(MID[i][0] - MID[i][1])] += 1 
                        SomaD1Unsat += int(MID[i][0] - MID[i][1])
                        X[1] += 1
                    # elif MID_Unsat[i] == int(1):
                    #     D2Unsat[int(MID[i][0] - MID[i][1])] += 1 
                    #     SomaD2Unsat += int(MID[i][0] - MID[i][1])
                    #     X[2] += 1
                    else:
                        D2Unsat[int(MID[i][0] - MID[i][1])] += 1 
                        SomaD2Unsat += int(MID[i][0] - MID[i][1])
                        X[2] += 1

            MID.pop(i) 
            X[5] -= 1   # Polymer mid-chain radical (tertiary mid-chain radical)
            # X[0] += 1   
            MID_Unsat.pop(i)          
        elif rnd <= R[0]+R[1]:       # General radical addition X[1] (opposite of depropagation)
            # print("R11: General radical addition")
            dog2 += 1
            reacao = "R[16]"
            i = int(np.random.uniform(0.0, 1.0)*X[3])
            # if (X[1]) == int(1): j = int(1)
            # else: j = int(np.random.uniform(0.0, 1.0)*(X[1]))
            j = int(np.random.uniform(0.0, 1.0)*(X[1]))
            if j > 0:
                explicacao = "R11. j<=X[1]: j <= X[1]"
                SomaD = int(0)
                for L in range(1,len(D1Unsat),1):
                    SomaD += (D1Unsat[int(L)])
                    if SomaD >= j: break

                # D1Unsat_ = D1Unsat.sum()
                MID_Unsat.append(0)
                D1Unsat[int(L)] -= 1
                SomaD1Unsat -= int(L)
                X[1] -= 1
            else: # if j == (0):
                explicacao = "R11. j == 0: X[1] > 0"
                for L in range(1,len(D1Unsat),1):
                    if (D1Unsat[int(L)]) > (0): break
                MID_Unsat.append(0)
                D1Unsat[int(L)] -= 1
                SomaD1Unsat -= int(L)
                X[1] -= 1

            MID.append((L + P[i], int(L)))
            X[5] += 1   # MID-CHAIN RADICAL
            P.pop(i)
            P2_Unsat.pop(i)
            X[3] -= 1   # POLYMER RADICAL (SECONDARY)   
            X3length -= i
        elif rnd <= np.sum(R[:3]):   # General radical addition X[2] (opposite of depropagation)
            # print("R11: General radical addition")
            reacao = "R[6]"
            dog3 += 1
            i = int(np.random.uniform(0.0, 1.0)*X[3])
            ## j = int(np.random.uniform(0.0, 1.0)*(X[1]+X[2]))
            # if (X[2]) == int(1): j = int(1)
            # else: j = int(np.random.uniform(0.0, 1.0)*(X[2]))
            j = int(np.random.uniform(0.0, 1.0)*(X[2]))

            if j == (0):
                explicacao = "R11. j == 0: X[2] > 0"
                for L in range(1,len(D2Unsat),1):
                    if D2Unsat[int(L)] > 0: break
                # if int(L) < int(3): continue
            # else:
            # elif j == (X[2]):
            #     for L in range(len(D2Unsat)-1,0, -1):
            #         if D2Unsat[int(L)] > 0: break
            #     explicacao = "R11: > X[1]: entrou em j == (X[1] + X[2])"
            # elif int(j) == (1):
            #     for L in range(1,len(D2Unsat),1):
            #         if D2Unsat[int(L)] > 0: break
            #     explicacao = "R11: > X[1]: entrou em j == (X[1] + 1)"
            else:
                SomaD = 0
                for L in range(1,len(D2Unsat),1):
                    SomaD += (D2Unsat[int(L)])
                    if SomaD >= j: break
                explicacao = "R11: > X[1]: entrou em j == X[2]*?"
            
            MID_Unsat.append(1)
            D2Unsat[int(L)] -= 1
            SomaD2Unsat -= int(L)
            X[2] -= 1

            MID.append((L + P[i], int(L)))
            X[5] += 1   # MID-CHAIN RADICAL
            P.pop(i)
            P2_Unsat.pop(i)
            X[3] -= 1   # POLYMER RADICAL (SECONDARY)
            X3length -= i
        elif rnd <= np.sum(R[:4]):   # Hydrogen abstraction X[0]
            # print("R2: Hydrogen abstraction")
            dog4 += 1
            reacao = "R[1]"
            i = int(np.random.uniform(0.0, 1.0)*X[3])
            # if X[0]+X[1]+X[2] == 1: j = 1
            # else: j = int(np.random.uniform(0.0, 1.0)*(X[0]+X[1]+X[2]))

            # if X[0] == 1: j = 1
            # else: j = int(np.random.uniform(0.0, 1.0)*(X[0]))
            j = int(np.random.uniform(0.0, 1.0)*(X[0]))

            if (j) == (0):
                for l in range(1,len(D),1):
                    if D[int(l)] > 0: break
            # elif int(j) == (X[0]):
            #     for l in range(len(D)-1,0,-1):
            #         if (D[int(l)]) > int(0): break
            else: 
                SomaD = int(0)
                for l in range(1,len(D),1):
                    SomaD += (D[int(l)])
                    if SomaD >= j: break

            # if D[int(l)] - 1 < 0: continue   # Checking ERROR 
            PositionMidRadical = int(np.random.rand()*(l))  # Due to open limit of np.random.rand, l will never be chosen 
            # if int(PositionMidRadical) == 0: continue       # But hydrogen abstraction should not be at the end of a chain.
                # Here should be WHILE int(PositionMidRadical) == 0: continue
                # Because 'l' will never be lower than 4 (monomer, dimer and trimer are volatile)
            while int(PositionMidRadical) == 0: PositionMidRadical = int(np.random.rand()*(l))

            if   P2_Unsat[i] == 0:
                D[int(P[i])] += 1   # Dead polymer chain is formed with size P[i]
                Dsoma += P[i]
                X[0] += 1
            elif P2_Unsat[i] == 1:
                D1Unsat[int(P[i])] += 1
                SomaD1Unsat += P[i]
                X[1] += 1
            else:
                D2Unsat[int(P[i])] += 1
                SomaD2Unsat += P[i]
                X[2] += 1                    
            P.pop(i)
            P2_Unsat.pop(i)
            X[3] -= 1   # Polymer Radical (Secondary)
            X3length -= i

            D[int(l)] -= 1
            Dsoma -= int(l)
            X[0] -= 1
            MID_Unsat.append(0)

            MID.append((l,int(PositionMidRadical)))
            X[5] += 1   # Polymer mid-chain radical        
        elif rnd <= np.sum(R[:5]):   # Depropagation
            dog5 += 1
            antes = X[12]
            
            i = int(np.random.uniform(0.0, 1.0)*X[3])     # X[3]: Polymer radical (Secondary) 
                 
            if P[int(i)] == int(2): # If it is a dimer 
                explicacao = "R3: P[i] == 2"
                X[3]  -= 1   # Polymer Radical (Secondary)
                X[12] += 1   # Monomer
                if int(P2_Unsat[i]) == int(0): X[10] += 1   # Benzyl radical
                else: X[20] += 1   # Styryl radical (very unlikely)
                P.pop(i)
                P2_Unsat.pop(i)
                X3length -= i 
            else:
                explicacao = "R3: P[i] > 2"
                P[i] = P[i] -  1
                X[12] = (X[12] + 1)  # Monomer    
                X3length -= 1
        elif rnd <= np.sum(R[:6]):   # Hydrogen abstraction X[1]
            dog6 += 1
            i = int(np.random.uniform(0.0, 1.0)*X[3])
            j = int(np.random.uniform(0.0, 1.0)*(X[1]))

            if (j) == (0):
                for l in range(1,len(D1Unsat),1):
                    if (D1Unsat[int(l)]) > (0): break                    
            else:
                explicacao = "int(j) < int(j) <= (X[0] + X[1])"
                SomaD = 0
                for l in range(1,len(D1Unsat),1):
                    SomaD += (D1Unsat[int(l)])
                    if SomaD >= j: break

            PositionMidRadical = int(np.random.rand()*(l))
            while int(PositionMidRadical) == 0: PositionMidRadical = int(np.random.rand()*(l))

            if P2_Unsat[i] == 0:
                D[int(P[i])] += 1   # Dead polymer chain is formed with size P[i]
                Dsoma += P[i]
                X[0] += 1
            elif P2_Unsat[i] == 1:
                D1Unsat[int(P[i])] += 1
                SomaD1Unsat += P[i]
                X[1] += 1
            else:
                D2Unsat[int(P[i])] += 1
                SomaD2Unsat += P[i]
                X[2] += 1                    
            P.pop(i)
            P2_Unsat.pop(i)
            X[3] -= 1   # Polymer Radical (Secondary)
            X3length -= i 

            D1Unsat[int(l)] -= 1
            SomaD1Unsat  -= int(l)
            X[1] -= 1
            MID_Unsat.append(1)

            MID.append((l,int(PositionMidRadical)))
            X[5] += 1   # Polymer mid-chain radical 
        elif rnd <= np.sum(R[:7]):   # Hydrogen abstraction X[2]
            reacao = "R[3]"
            dog7 += 1
            i = int(np.random.uniform(0.0, 1.0)*X[3])
            j = int(np.random.uniform(0.0, 1.0)*(X[2]))

            if (j) == int(0):
                for l in range(1,len(D2Unsat),1):
                    if D2Unsat[int(l)] > 0: break 
            else:
                SomaD = 0         
                for l in range(0,len(D2Unsat),1):
                    SomaD += (D2Unsat[int(l)])
                    if SomaD >= j: break

            PositionMidRadical = int(np.random.rand()*(l))
            while int(PositionMidRadical) == 0: PositionMidRadical = int(np.random.rand()*(l))

            if P2_Unsat[i] == 0:
                D[int(P[i])] += 1   # Dead polymer chain is formed with size P[i]
                Dsoma += P[i]
                X[0] += 1
            elif P2_Unsat[i] == 1:
                D1Unsat[int(P[i])] += 1
                SomaD1Unsat += P[i]
                X[1] += 1
            else:
                D2Unsat[int(P[i])] += 1
                SomaD2Unsat += P[i]
                X[2] += 1                    
            P.pop(i)
            P2_Unsat.pop(i)
            X[3] -= 1   # Polymer Radical (Secondary)
            X3length -= i 

            D2Unsat[int(l)] -= 1
            SomaD2Unsat  -= int(l)
            X[2] -= 1
            MID_Unsat.append(2)

            MID.append((l,int(PositionMidRadical)))
            X[5] += 1   # Polymer mid-chain radical 
        elif rnd <= np.sum(R[:8]):   # Mid-chain beta scission of 13 mid chain 
            # print("R14: Mid-chain beta scission of 13 mid chain ")
            reacao = "R[21]"
            dog8 += 1
            ## 1.4 - 2.23 dimer molecules are produced for one molecule of toluene (1989 - Bouster, Vermande and Veron - Evolution of the product yield with temperature and molecular weight in the pyrolysis of polystyrene)

            i = int(np.random.rand()*X[6])
            j = np.random.uniform(0.0, 1.0)
            if j < 0.5: #  Formation of Dimer
                if int(RAD13[i] - 2) == int(1):
                    if int(RAD13_Unsat[i]) == int(0): X[10] += 1                    # Benzyl radical =  Tolune radical (TOLradical)
                    else: X[20] += 1 # Styryl radical (very unlikely)    
                else:      
                    X[3]  += 1                   # Polymer Radical (Secondary)
                    P.append(RAD13[i]-2)
                    P2_Unsat.append(RAD13_Unsat[i])
                    X3length += RAD13[i]-2
                X[14]  += 1                   # Dimer
                X[6] -= 1                   # Mid-chain 13 radical    
            else:
                X[10]  += 1  # X[13] += 1 # Tolune radical (TOLradical)
                X[6]   -= 1 # Mid-chain 13 radical
                X[0]   += 1 # Dead polymer 
                
                D[int(RAD13[i] - 1)] += 1
                Dsoma += int(RAD13[i] - 1)

            RAD13.pop(i)
            RAD13_Unsat.pop(i)
        elif rnd <= np.sum(R[:9]):   # TOL formation (hydrogen abstraction) X[0]   
            # print("R17: TOL formation (hydrogen abstraction)")
            reacao = "R[26]"
            dog9 += 1  
            i = int(np.random.uniform(0.0, 1.0)*(X[0]))

            if (i) == (0):   
                if (X[0]) == int(1): 
                    j = (Dsoma)
                    explicacao = "i == 0, X[0]=1"
                else:
                    for j in range(1,len(D),1):
                        if (D[int(j)]) > (0): break
                    explicacao = "i == 0, X[0]>0"
            else:
                SomaD = 0
                for j in range(0,len(D),1):
                    SomaD += D[int(j)]
                    if SomaD >= i: break
                explicacao = "i <= X[0]"


            PositionMidRadical = int(np.random.rand()*j)
            while int(PositionMidRadical) == 0: PositionMidRadical = int(np.random.rand()*j) 
            MID.append((j,int(PositionMidRadical)))

            X[10] -= 1  # X[13] -= 1 # Tolune radical (TOLradical)# TOLradical
            X[5]  += 1  # Polymer mid-chain radical   
            X[19] += 1  # Toluene

            X[0] -= 1
            D[int(j)] -= 1
            Dsoma -= int(j)
            MID_Unsat.append(0)
        elif rnd <= np.sum(R[:10]):  # Mid-chain beta scission of 15 mid chain 
            # print("R13: Mid-chain beta scission of 15 mid chain ")
            dog10 += 1
            reacao = "R[20]"
            i = int(np.random.uniform(0.0, 1.0)*X[7])
            j = np.random.uniform(0.0, 1.0)
            if j < 0.5: #  Formation of Trimer 
                if int(RAD15[i] - 3) == int(1):
                    if int(RAD15_Unsat[i]) == int(0): X[10] += 1  # Benzyl radical =  Tolune radical (TOLradical)
                    else: X[20] += 1 # Styryl radical (very unlikely)
                else:
                    X[3] += 1 # Polymer Radical (Secondary)
                    P.append(RAD15[i] - 3)
                    P2_Unsat.append(RAD15_Unsat[i])
                    X3length += RAD15[i] - 3
                X[16] += 1 # Trimer
                X[7]-= 1 # Mid-chain 15 radical
            else:
                X[7] -= 1 # Mid-chain 15 radical
                X[11] += 1 # DPP radical 

                # Para ter entrado aqui, P[m] >= 4
                if int(RAD15[i] - 2) == int(2):
                    if int(RAD15_Unsat[i]) == int(0): X[14] += 1 # DIMER 
                    else: 
                        # X[14] += 1  # Dimer 
                        X[21] += 1    # Dimer with 2 unsaturations 
                        # print("!!!!!! R[13]: DIMER DIETENE (DUPLA INSATURAÇÃO)!!!")     #  2,4-diphenyl-1,4-pentadiene
                else:
                    if int(RAD15_Unsat[i]) == int(0):
                        X[1]  += 1 # Dead polymer Unsaturated 
                        D1Unsat[int(RAD15[i] - 2)] += 1
                        SomaD1Unsat += int(RAD15[i] - 2)
                    else:
                        X[2] += 1 # Dead polymer 2xUnsaturated 
                        D2Unsat[int(RAD15[i] - 2)] += 1
                        SomaD2Unsat += int(RAD15[i] - 2)
            RAD15.pop(i)
            RAD15_Unsat.pop(i)
        elif rnd <= np.sum(R[:11]):  # k15
            # print("R8: k15")
            reacao = "R[13]"
            dog11 += 1
            m = int(np.random.uniform(0.0, 1.0)*(X[3]))
            if P[m] < 5: continue   # if P[m] < 4: continue

            RAD15.append(P[m])
            RAD15_Unsat.append(P2_Unsat[m])
            P.pop(m)
            P2_Unsat.pop(m)
            X[3] -= 1                    # Polymer radical (primary + secondary radical)
            X3length  -= m        
            X[7] += 1                   # Mid-chain 15 radical        
        elif rnd <= np.sum(R[:12]):  # Benzyl radical addition X[1] (Levine, 2008)
            # print("R12: Benzyl radical addition")
            reacao = "R[19]"

            j = int(np.random.uniform(0.0, 1.0)*(X[1]))
            if (j) == (0):
                if (X[1]) == int(1):
                    l = SomaD1Unsat
                    # if int(l+1) < int(3): continue
                    ## if int(SomaD1Unsat - l) < 0: print("R[12]! j=0! X[1]=1! SomaD1Unsat:", SomaD1Unsat, ". l:", l)
                    RAD13_Unsat.append(0)
                    RAD13.append(l+1)           # RAD13.append(l)
                    X[6]  += 1                  # Mid-chain 13 radical
                    X[10] -= 1                  # Benzyl radical
                    X[1]  -= 1
                    D1Unsat[int(l)] -= 1
                    SomaD1Unsat -= int(l)                        
                    explicacao = "int(j) == 0; X[1] = 1"
                else:
                    for l in range(1,len(D1Unsat),1):
                        if D1Unsat[int(l)] > 0: break
                    # if int(l+1) < int(3): continue
                    RAD13_Unsat.append(0)
                    RAD13.append(l+1) # RAD13.append(l)
                    X[6] += 1                   # Mid-chain 13 radical
                    X[10] -= 1                    # Benzyl radical
                    X[1] -= 1
                    D1Unsat[int(l)] -= 1
                    # if int(SomaD1Unsat - l) < 0: print("R[12]! j=0! X[1]>0! SomaD1Unsat:", SomaD1Unsat, ". l:", l)
                    if (X[1]) < int(0): print("R[12]: X[1]<0=", X[1])
                    SomaD1Unsat -= int(l)
                    explicacao = "int(j) == 0; X[1] > 0"
            else:
                if (X[1]) == int(1): l = SomaD1Unsat
                else: 
                    SomaD = int(0)
                    for l in range(0,len(D1Unsat),1):
                        SomaD += (D1Unsat[int(l)])
                        if SomaD >= j: break 
                    explicacao = "int(j) == (X[1])*?"
                # if int(l+1) < int(3): continue
                # if int(SomaD1Unsat - l) < 0: print("R[12]! j<=X[1]! SomaD1Unsat:", SomaD1Unsat, ". l:", l, "j:", j, "; X[1]", X[1])
                RAD13_Unsat.append(0)
                RAD13.append(l+1)           # RAD13.append(l)
                X[6]  += 1                  # Mid-chain 13 radical
                X[10] -= 1                  # Benzyl radical
                X[1]  -= 1
                D1Unsat[int(l)] -= 1
                SomaD1Unsat -= int(l)
                # if X[1] < 0: print("R[12]! j<=X[1]! X[1]<0=", X[1])          
        elif rnd <= np.sum(R[:13]):  # Benzyl radical addition X[2] (Levine, 2008)
            # print("R12: Benzyl radical addition")
            reacao = "R[19]"
            dog13 += 1  

            j = int(np.random.uniform(0.0, 1.0)*(X[2]))
            if (j) == (0):          
                if (X[2]) == int(1):
                    l = SomaD2Unsat
                    # if int(l+1) < int(3): continue
                    RAD13_Unsat.append(1)
                    RAD13.append(l+1)           # RAD13.append(l)
                    X[6]  += 1                  # Mid-chain 13 radical
                    X[10] -= 1                  # Benzyl radical
                    X[2]  -= 1 
                    D2Unsat[int(l)] -= 1
                    SomaD2Unsat -= int(l)
                    explicacao = "int(j) == 0; X[1] = 0; X[2] = 1"
                else:
                    for l in range(1,len(D2Unsat),1):
                        if D2Unsat[int(l)] > 0: break
                    # if int(l+1) < int(3): continue
                    RAD13_Unsat.append(1)
                    RAD13.append(l+1) # RAD13.append(l)
                    X[6] += 1                   # Mid-chain 13 radical
                    X[10] -= 1                    # Benzyl radical
                    X[2] -= 1 
                    D2Unsat[int(l)] -= 1
                    SomaD2Unsat -= int(l)
                    explicacao = "int(j) == 0; X[2] > 0"
            else:
                SomaD = int(0)
                for l in range(1,len(D2Unsat),1):
                    SomaD += (D2Unsat[int(l)])
                    if SomaD >= j: break
                explicacao = "int(j) => (X[2])*?"

                # if int(l+1) < int(3): continue
                RAD13_Unsat.append(1)
                RAD13.append(l+1)           # RAD13.append(l)
                X[6]  += 1                  # Mid-chain 13 radical
                X[10] -= 1                  # Benzyl radical
                X[2]  -= 1 
                D2Unsat[int(l)] -= 1
                SomaD2Unsat -= int(l)
        elif rnd <= np.sum(R[:14]):  # Termination by combination (radical recombination)
            # print("R5: Termination by combination (radical recombination)")   # 2 + 2
            reacao = "R[14]"
            dog14 += 1
            m, n = int(np.random.uniform(0.0, 1.0)*X[3]), int(np.random.uniform(0.0, 1.0)*X[3])
            if m == n:
                if m == 0: n += 1
                else: n -= 1

            if int(P[m]+P[n]) > len(D): # DP = 6000
                maior_que_DP += 1
                if P2_Unsat[m] > 0 or P2_Unsat[n] > 0: 
                    # if (P2_Unsat[m] + P2_Unsat[n]) > 2:
                    #     print("R[5]! Error! P2_Unsat[m] + P2_Unsat[n] > 2")
                    if (P2_Unsat[m] + P2_Unsat[n]) == 2:
                        D2Unsat[-1] += 1
                        SomaD2Unsat += len(D)
                        X[2] += 1
                    else:
                        D1Unsat[-1] += 1
                        SomaD1Unsat += len(D)
                        X[1] += 1      
                else:
                    D[-1] += 1
                    Dsoma += len(D)
                    X[0] += 1
            else:
                if P2_Unsat[m] > 0 or P2_Unsat[n] > 0:
                    # if (P2_Unsat[m] + P2_Unsat[n]) > 2: print("R[5]! Error! P2_Unsat[m] + P2_Unsat[n] > 2")
                    if (P2_Unsat[m] + P2_Unsat[n]) == 2:
                        D2Unsat[int(P[m]+P[n])] += 1
                        SomaD2Unsat += int(P[m]+P[n])
                        X[2] += 1
                    else:
                        D1Unsat[int(P[m]+P[n])] += 1
                        SomaD1Unsat += int(P[m]+P[n])
                        X[1] += 1
                else:                                                        
                    D[int(P[m]+P[n])] += 1
                    Dsoma += int(P[m]+P[n])
                    X[0] += 1
            

            if m > n: ind = [m, n]
            else: ind = [n, m]
            for index in ind: 
                P.pop(index)
                P2_Unsat.pop(index)
            X[3] -= 2  
            X3length = X3length - m - n
        elif rnd <= np.sum(R[:15]):  # TOL formation (hydrogen abstraction) X[1]   
            # print("R17: TOL formation (hydrogen abstraction)")
            reacao = "R[27]"
            dog15 += 1  

            i = int(np.random.uniform(0.0, 1.0)*(X[1]))

            if (i) == (0):   
                if (X[1]) == (1):
                    j = (SomaD1Unsat)
                    explicacao = "i == 0, X[1]=1"
                else:
                    for j in range(1,len(D1Unsat),1):
                        if (D1Unsat[int(j)]) > (0): break
                    explicacao = "i == 0, X[1]>0"
            else:
                SomaD = 0                  
                for j in range(1,len(D1Unsat),1):
                    SomaD += (D1Unsat[int(j)])
                    if SomaD >= (i): break
                explicacao = "i < X[0] + X[1]"

            PositionMidRadical = int(np.random.rand()*j)
            # if int(PositionMidRadical) == 0: continue
            while int(PositionMidRadical) == 0: PositionMidRadical = int(np.random.rand()*j)
            MID.append((j,int(PositionMidRadical)))

            X[10] -= 1  # X[13] -= 1 # Tolune radical (TOLradical)# TOLradical
            X[5]  += 1  # Polymer mid-chain radical   
            X[19] += 1  # Toluene

            X[1] -= 1
            D1Unsat[int(j)] -= 1
            SomaD1Unsat -= int(j)
            MID_Unsat.append(1)
        elif rnd <= np.sum(R[:16]):  # alpha-Methylstyrene FORMATION X[0], hydrogen abstraction 
            # i = int(np.random.uniform(0.0, 1.0)*(X[0]+X[1]+X[2]))  # X[0]: Polymer; Mid- chain scisson will be occuring in Polymer i
            reacao = "R[29]"
            dog16 += 1  
            
            # if (X[0]) == int(1): i = int(1)
            # else: i = int(np.random.uniform(0.0, 1.0)*(X[0])) 
            i = int(np.random.uniform(0.0, 1.0)*(X[0])) 
            
            reacao = "R[18]"
            if (i) == (0):
                if (X[0]) == int(1):
                    j = Dsoma
                else:
                    for j in range(1,len(D),1):
                        if D[int(j)] > 0: break
            else:
                SomaD = int(0)
                for j in range(0,len(D),1):
                    SomaD += (D[int(j)])
                    if SomaD >= i: break

            PositionMidRadical = int(np.random.rand()*(j))
            # if int(PositionMidRadical) == 0: continue
            while int(PositionMidRadical) == 0: PositionMidRadical = int(np.random.rand()*(j))
            MID.append((j,int(PositionMidRadical)))

            X[9]  -= 1  # Allyl benzene radical  # X[13] -= 1 # Tolune radical (TOLradical)# TOLradical
            X[5]  += 1  # MID-CHAIN RADICAL   
            X[13] += 1  # ALPHA-METHYLSTYRENE

            X[0] -= 1
            D[int(j)] -= 1
            Dsoma -= int(j)
            MID_Unsat.append(0)
        elif rnd <= np.sum(R[:17]):  # DPP formation  (hydrogen abstraction)
            # print("R15: DPP formation  (hydrogen abstraction)")
            reacao = "R[22]"
            dog17 += 1

            i = int(np.random.uniform(0.0, 1.0)*(X[0]))

            if (i) == (0):
                for j in range(1,len(D),1):
                    if D[int(j)] > 0: break
            else:
                SomaD = int(0)
                for j in range(1,len(D),1):
                    SomaD += (D[int(j)])
                    if SomaD >= i: 
                        if D[int(j)] <= 0: print("R15: D[int(j)] - 1 < 0. i:", i, "j:", j, "X[0]", X[0], ". D[:5]", D[int(j)])
                        break    

            PositionMidRadical = int(np.random.rand()*(j))
            # if int(PositionMidRadical) == 0: continue    # Entra muito aqui 
            while int(PositionMidRadical) == 0: PositionMidRadical = int(np.random.rand()*(j))
            MID.append((j,int(PositionMidRadical)))
                
            X[11] -= 1  # DPP radical
            X[15] += 1  # DPP 
            X[5]  += 1  # Polymer mid-chain radical

            X[0] -= 1  # -1 Polymer
            D[int(j)] -= 1
            Dsoma -= int(j)
            # if Dsoma < 0: print("R15: Dsoma < 0! X[0]=", X[0], "Dsoma", Dsoma)  
            MID_Unsat.append(0)
        elif rnd <= np.sum(R[:18]):  # End chain scisson X[1]
            # print("R1: End chain scisson")
            reacao = "R[4]"
            dog18 += 1

            i = int(np.random.uniform(0.0, 1.0)*X[1])

            if i == int(0): 
                for j in range(1,len(D1Unsat),1):
                    if D1Unsat[int(j)] > 0: break 
            # elif int(i) == (X[1]):
            #     for j in range(len(D1Unsat)-1,0, -1):
            #         if D1Unsat[int(j)] > 0: break
            else:
                SomaD = int(0)
                for j in range(0,len(D1Unsat),1): # --- ACHANDO O TAMANHO DESSE POLÍMERO: j
                    SomaD += (D1Unsat[int(j)])
                    if SomaD >= i: break 

            ## if int(j) == int(1): print("R[1]: Polymer with size 1 (j = 1)")
            # if int(j - 1) <= 0: continue  # Polymer size <= 1; This cannot happen

            if int(j - 1) == int(1): # Polymer size (dimer) = 2
                X[9] += 1 # Allyl Benzene Radical
                if int(i) > (X[1]):  X[9] += 1 # Allyl Benzene Radical
                else: X[10] += 1  # Benzyl radical =  Tolune radical (TOLradical)
            else:
                P.append(j - 1)
                X3length += (j - 1)
                X[3] += 1 # Polymer Radical (Secondary)
                X[9] += 1 # Allyl Benzene Radical
                if int(i) > (X[0] + X[1]): P2_Unsat.append(1) # This Polymer Radical (Secondary) has a Unsaturated Tail End 
                else:  P2_Unsat.append(0)
            
            if (i) == (0):
                D1Unsat[int(j)] -= 1
                SomaD1Unsat -= int(j)  
                X[1] -= 1          
            else:
                D1Unsat[int(j)] -= 1
                SomaD1Unsat -= int(j)  
                X[1] -= 1
                # if (X[1]) < int(0): print("R[1]: X[1]<0. X[1]:", X[1], "; X2:", X[2], "; i:", i)
        elif rnd <= np.sum(R[:19]):  # DPP radical addition X[1]: dppRAD + PSunsat -> Rtb (1,5)  [kadd(sb,tb)]
            reacao = "R[39]"
            dog19 += 1  

            j = int(np.random.uniform(0.0, 1.0)*(X[1]))
            if (j) == (0):
                if (X[1]) == int(1):
                    l = SomaD1Unsat
                    # if int(l+1) < int(3): continue
                    # if int(SomaD1Unsat - l) < 0: print("DPPrad addition! j=0! X[1]=1! SomaD1Unsat:", SomaD1Unsat, ". l:", l)
                    RAD15_Unsat.append(0)
                    RAD15.append(l+2)
                    X[7]  += 1                  # Mid-chain 15 radical
                    X[11] -= 1                  # DPP radical 
                    X[1]  -= 1
                    D1Unsat[int(l)] -= 1
                    SomaD1Unsat -= int(l)                        
                    explicacao = "DPPrad addition: int(j) == 0; X[1] = 1"
                else:
                    for l in range(1,len(D1Unsat),1):
                        if (D1Unsat[int(l)]) > int(0): break
                    # if int(l+1) < int(3): continue
                    RAD15_Unsat.append(0)
                    RAD15.append(l+2)
                    X[7]  += 1                  # Mid-chain 15 radical
                    X[11] -= 1                  # DPP radical 
                    X[1]  -= 1
                    D1Unsat[int(l)] -= 1
                    # if int(SomaD1Unsat - l) < int(0): print("R[12]! j=0! X[1]>0! SomaD1Unsat:", SomaD1Unsat, ". l:", l)
                    if (X[1]) < int(0): print("R[12]: X[1]<0=", X[1])
                    SomaD1Unsat -= int(l)
                    explicacao = "int(j) == 0; X[1] > 0"                 
            else:
                SomaD = int(0)
                for l in range(0,len(D1Unsat),1):
                    SomaD += (D1Unsat[int(l)])
                    if SomaD >= j: break 
                explicacao = "int(j) == (X[1])*?"

                # if int(l+1) < int(3): continue
                # if int(SomaD1Unsat - l) < 0: print("R[12]! j<=X[1]! SomaD1Unsat:", SomaD1Unsat, ". l:", l, "j:", j, "; X[1]", X[1])
                RAD15_Unsat.append(0)
                RAD15.append(l+2)
                X[7]  += 1                   # Mid-chain 15 radical
                X[11] -= 1                   # DPP radical
                X[1]  -= 1
                D1Unsat[int(l)] -= 1
                SomaD1Unsat -= int(l)
                # if X[1] < 0: print("R[12]! j<=X[1]! X[1]<0=", X[1])
        elif rnd <= np.sum(R[:20]):  # TOL formation (hydrogen abstraction) X[2]   
            # print("R19: TOL formation (hydrogen abstraction)")
            i = int(np.random.uniform(0.0, 1.0)*(X[2]))

            if (i) == (0):   
                if (X[2]) == int(1):
                    j = (SomaD2Unsat)
                    explicacao = "i == 0, X[2]=1"
                else:
                    for j in range(1,len(D2Unsat),1):
                        if (D2Unsat[int(j)]) > (0): break
                    explicacao = "i == 0, X[2]>0"
            else:
                SomaD = 0              
                for j in range(0,len(D2Unsat),1):
                    SomaD += (D2Unsat[int(j)])
                    if SomaD >= int(i): break
                explicacao = "i < X[0] + X[1] + X[2]"

            PositionMidRadical = int(np.random.rand()*j)
            while int(PositionMidRadical) == 0: PositionMidRadical = int(np.random.rand()*j)   ## Raising errors
            MID.append((j,int(PositionMidRadical)))

            X[10] -= 1  # X[13] -= 1 # Tolune radical (TOLradical)# TOLradical
            X[5]  += 1  # Polymer mid-chain radical   
            X[19] += 1  # Toluene

            X[2] -= 1
            D2Unsat[int(j)] -= 1
            SomaD2Unsat -= int(j)                
            MID_Unsat.append(2)      
        elif rnd <= np.sum(R[:21]):  # End chain scisson X[2]
            # print("R1: End chain scisson")
            i = int(np.random.uniform(0.0, 1.0)*X[2]) 

            if (i) == (0): 
                for j in range(1,len(D2Unsat),1):
                    if D2Unsat[int(j)] > 0: break 
            else:
                    SomaD = 0                
                    for j in range(0,len(D2Unsat),1):
                        SomaD += (D2Unsat[int(j)])
                        if SomaD >= i: break

            ## if int(j) == int(1): print("R[1]: Polymer with size 1 (j = 1)")
            # if int(j - 1) <= 0: continue  # Polymer size <= 1; This cannot happen

            if int(j - 1) == int(1): # Polymer size (dimer) = 2
                X[9] += 1 # Allyl Benzene Radical
                if int(i) > (X[1]):  X[9] += 1 # Allyl Benzene Radical
                else: X[10] += 1  # Benzyl radical =  Tolune radical (TOLradical)
            else:
                P.append(j - 1)
                X3length += (j-1)
                X[3] += 1 # Polymer Radical (Secondary)
                X[9] += 1 # Allyl Benzene Radical
                if int(i) > (X[0] + X[1]): P2_Unsat.append(1) # This Polymer Radical (Secondary) has a Unsaturated Tail End 
                else:  P2_Unsat.append(0)
            
            if int(i) == int(0):
                D2Unsat[int(j)] -= 1
                SomaD2Unsat -= int(j)   
                X[2] -= 1                                   
            else:
                D2Unsat[int(j)] -= 1
                SomaD2Unsat -= int(j)   
                X[2] -= 1
        elif rnd <= np.sum(R[:22]):  # Styryl formation  (hydrogen abstraction)
            reacao = "R[45]"
            dog22 += 1  

            i = int(np.random.uniform(0.0, 1.0)*(X[0]))

            if (i) == (0):
                for j in range(1,len(D),1):
                    if D[int(j)] > 0: break
            else:
                SomaD = int(0)
                for j in range(1,len(D),1):
                    SomaD += (D[int(j)])
                    if SomaD >= i: break    

            # if j < 3: 
            #     print("R[32]: Tamanho do polímero não pode ser menor que 3")
            #     continue
            # else:
            #     PositionMidRadical = int(np.random.rand()*(j))
            #     # if int(PositionMidRadical) == 0: continue   # RAISING ERRORS
            #     while int(PositionMidRadical) == 0: PositionMidRadical = int(np.random.rand()*(j))
            #     MID.append((j,int(PositionMidRadical)))

            #     X[20] -= 1  # Styryl radical (very unlikely)
            #     X[12] += 1  # Monomer (styrene)
            #     X[5]  += 1  # Polymer mid-chain radical

            #     X[0] -= 1  # -1 Polymer
            #     D[int(j)] -= 1
            #     Dsoma -= int(j)
            #     MID_Unsat.append(0)

            PositionMidRadical = int(np.random.rand()*(j))
            # if int(PositionMidRadical) == 0: continue   # RAISING ERRORS
            while int(PositionMidRadical) == 0: PositionMidRadical = int(np.random.rand()*(j))
            MID.append((j,int(PositionMidRadical)))

            X[20] -= 1  # Styryl radical (very unlikely)
            X[12] += 1  # Monomer (styrene)
            X[5]  += 1  # Polymer mid-chain radical

            X[0] -= 1  # -1 Polymer
            D[int(j)] -= 1
            Dsoma -= int(j)
            MID_Unsat.append(0)
        elif rnd <= np.sum(R[:23]):  # DPP radical addition X[2]: dppRAD + PSunsat -> Rtb (1,5)  [kadd(sb,tb)]
            reacao = "R[40]"
            dog23 += 1  

            j = int(np.random.uniform(0.0, 1.0)*(X[2]))
            if (j) == (0):
                if (X[2]) == int(1):
                    l = SomaD2Unsat
                    # if int(l+1) < int(3): continue
                    RAD15_Unsat.append(0)
                    RAD15.append(l+2)
                    X[7]  += 1                   # Mid-chain 15 radical
                    X[11] -= 1                   # DPP radical 
                    X[2]  -= 1 
                    D2Unsat[int(l)] -= 1
                    SomaD2Unsat -= int(l)
                    explicacao = "int(j) == 0; X[1] = 0; X[2] = 1"
                else:
                    for l in range(1,len(D2Unsat),1):
                        if (D2Unsat[int(l)]) > int(0): break
                    # if int(l+1) < int(3): continue
                    RAD15_Unsat.append(0)
                    RAD15.append(l+2)
                    X[7]  += 1                   # Mid-chain 15 radical
                    X[11] -= 1                   # DPP radical
                    X[2] -= 1 
                    D2Unsat[int(l)] -= 1
                    SomaD2Unsat -= int(l)
                    explicacao = "int(j) == 0; X[2] > 0"
            else:            
                SomaD = 0
                for l in range(1,len(D2Unsat),1):
                    SomaD += (D2Unsat[int(l)])
                    if SomaD >= j: break
                explicacao = "int(j) => (X[2])*?"
                # if int(l+1) < int(3): continue
                RAD15_Unsat.append(0)
                RAD15.append(l+2)
                X[7]  += 1                   # Mid-chain 15 radical
                X[11] -= 1                   # DPP radical
                X[2]  -= 1 
                D2Unsat[int(l)] -= 1
                SomaD2Unsat -= int(l)
        elif rnd <= np.sum(R[:24]):  # Mid-chain scission X[0]
            reacao = "R[6]"
            dog24 += 1
            explicacao = "nenhuma"
            i = int(np.random.uniform(0.0, 1.0)*(X[0])) # -- MID-CHAIN SCISSION WILL BE OCCURING IN POLYMER i
            if i > 0:
                explicacao = "R0: i < X[0]"
                SomaD = (0)
                for j in range(1,len(D),1):
                    SomaD += (D[int(j)])
                    if SomaD >= i: break                
            else: # if (i) ==(0):
                if (X[0]) == (1): 
                    explicacao = "R2. X[0] == 1 and X[1] == 0 and X[2]==0"
                    j = Dsoma
                    i = X[0]
                else:
                    explicacao = "R0: i=0, X[0] > 0"
                    for j in range(1,len(D),1):
                        if D[int(j)] > 0: break         


            # -- ESCOLHENDO ONDE ESSE POLÍMERO i VAI QUEBRAR
            l = int(np.random.rand()*(j-1))  # l = int(np.random.rand()*D[j-1])  # np.random.rand is a uniform distribution over [0, 1)
            # if i == (0) or l <= 1: continue # Error in random number generation (i) or the polymer will be breaking at the end (l <= 1)
            while l <= 1: l = int(np.random.rand()*(j-1)) 
            ## If there is a polymer specie with unsaturated end, the radical l will keep this unsaturated end
            ## l is the Primary Radical
            ## (j - l) os the Secondary radical 

            if (j - l) == (1):
                X[10] += 1  # Benzyl radical =  Tolune radical (TOLradical)
                if (l) == (1): 
                    X[17] += 1        # Ethylbenzene radical 
                else:
                    Pprimary.append(l)
                    X[4] += 1   # Polymer Radical (Primary)
                    P1_Unsat.append(0)
            else:
                # POLYMER RADICAL (PRIMARY)
                Pprimary.append(l)
                X[4] += 1
                P1_Unsat.append(0)
                # POLYMER RADICAL (SECONDARY)
                P.append(j-l)
                X[3] += 1
                X3length += (j-l)
                P2_Unsat.append(0)

            X[0] -= 1 
            D[int(j)] -= 1
            Dsoma -= (j)
        elif rnd <= np.sum(R[:25]):  # k17
            # print("R9: k17")
            reacao = "R[14]"        
            dog25 += 1    
            m = int(np.random.uniform(0.0, 1.0)*(X[3]))

            if P[m] < 5: continue

            RAD17.append(P[m])
            RAD17_Unsat.append(P2_Unsat[m])
            P.pop(m)
            P2_Unsat.pop(m)
            X3length -= m
            X[3] -= 1                    # Polymer radical (primary + secondary radical)
            X[8] += 1                   # Mid-chain 17 radical
        elif rnd <= np.sum(R[:26]):  # alpha-Methylstyrene FORMATION X[1], hydrogen abstraction 
            # i = int(np.random.uniform(0.0, 1.0)*(X[0]+X[1]+X[2]))  # X[0]: Polymer; Mid- chain scisson will be occuring in Polymer i
            reacao = "R[30]"
            dog26 += 1  
            
            # if (X[1]) == int(1): i = int(1)
            # else: i = int(np.random.uniform(0.0, 1.0)*(X[1])) 
            i = int(np.random.uniform(0.0, 1.0)*(X[1])) 
            
            reacao = "R[18]"
            if (i) == (0):
                if (X[1]) == int(1): j = SomaD1Unsat
                else:
                    for j in range(1,len(D1Unsat),1):
                        if D1Unsat[int(j)] > 0: break
            else:      
                # if (X[1]) == int(1):
                #     j = SomaD1Unsat
                # elif int(i) == (X[1]):
                #     for j in range(len(D1Unsat)-1,0,-1):
                #         if D1Unsat[int(j)] > 0: break
                # else:
                #     SomaD = 0              
                #     for j in range(0,len(D1Unsat),1):
                #         SomaD += (D1Unsat[int(j)])
                #         if SomaD >= int(i): break
                SomaD = 0              
                for j in range(0,len(D1Unsat),1):
                    SomaD += (D1Unsat[int(j)])
                    if SomaD >= (i): break

            
            PositionMidRadical = int(np.random.rand()*(j))
            # if int(PositionMidRadical) == 0: continue
            while int(PositionMidRadical) == 0: PositionMidRadical = int(np.random.rand()*(j))

            MID.append((j,int(PositionMidRadical)))

            X[9]  -= 1  # Allyl benzene radical  # X[13] -= 1 # Tolune radical (TOLradical)# TOLradical
            X[5]  += 1  # MID-CHAIN RADICAL   
            X[13] += 1  # ALPHA-METHYLSTYRENE

            X[1] -= 1
            D1Unsat[int(j)] -= 1
            SomaD1Unsat -= int(j)
            MID_Unsat.append(1)
        elif rnd <= np.sum(R[:27]):  # DPP formation X[1] (hydrogen abstraction)
            reacao = "R[10]"
            dog27 += 1
            i = int(np.random.uniform(0.0, 1.0)*(X[1]))
            if (i) == int(0):
                for j in range(1,len(D1Unsat),1):
                    if D1Unsat[int(j)] > 0: break
            # elif int(i) == (X[1]):
            #     for j in range(len(D1Unsat)-1,0,-1):
            #         if D1Unsat[int(j)] > 0: break
            else:
                SomaD = 0
                for j in range(1,len(D1Unsat),1):
                    SomaD += D1Unsat[int(j)]
                    if SomaD >= i: break

            PositionMidRadical = int(np.random.rand()*(j))
            # if int(PositionMidRadical) == 0: continue    # Entra muito aqui 
            while int(PositionMidRadical) == 0: PositionMidRadical = int(np.random.rand()*(j))
            MID.append((j,int(PositionMidRadical)))

            X[11] -= 1  # DPP radical
            X[15] += 1  # DPP 
            X[5]  += 1  # Polymer mid-chain radical
                
            X[1] -= 1
            D1Unsat[int(j)] -= 1
            SomaD1Unsat -= int(j)
            MID_Unsat.append(1)
        elif rnd <= np.sum(R[:28]):  # Hydrogen Abstraction: Primary Radical X[0] 
            reacao = "R[32]"
            dog28 += 1  
            i = int(np.random.uniform(0.0, 1.0)*X[4])
            j = int(np.random.uniform(0.0, 1.0)*(X[0]))

            if (j) == (0):
                explicacao = "j == 0, X[0] > 0"
                for l in range(1,len(D),1):
                    if (D[int(l)]) > int(0): break 
            else: 
                explicacao = "j < X[0]"
                SomaD = 0
                for l in range(1,len(D),1):
                    SomaD += D[int(l)]
                    if SomaD >= j: break
            
            PositionMidRadical = int(np.random.rand()*(l))
            while int(PositionMidRadical) == int(0): PositionMidRadical = int(np.random.rand()*(l))

            if int(P1_Unsat[i]) == int(0):
                D[int(Pprimary[i])] += int(1)
                Dsoma += int(Pprimary[i])
                X[0] += 1
            elif int(P1_Unsat[i]) == int(1):
                D1Unsat[int(Pprimary[i])] += 1
                SomaD1Unsat += int(Pprimary[i])
                X[1] += 1
            else:
                D2Unsat[int(Pprimary[i])] += 1
                SomaD2Unsat += int(Pprimary[i])
                X[2] += 1     
            Pprimary.pop(i)
            P1_Unsat.pop(i)
            X[4] -= 1

            D[int(l)] -= 1
            Dsoma -= int(l)
            X[0] -= 1
            MID_Unsat.append(0)
            MID.append((l,int(PositionMidRadical)))
            X[5] += 1   # Polymer mid-chain radical 
        elif rnd <= np.sum(R[:29]):  # k73
            # print("R10: k73")
            reacao = "R[15]"  
            dog29 += 1          
            m = int(np.random.uniform(0.0, 1.0)*(X[8]))

            RAD13.append(RAD17[m])  # RAD73.append(RAD17[m])
            RAD13_Unsat.append(RAD17_Unsat[m])
            RAD17.pop(m)
            RAD17_Unsat.pop(m)
            X[8] -= 1                   # Mid-chain 17 radical
            X[6] += 1                   # Mid-chain 73 radical = Mid-chain 13 radical <<<<<<<<<
        elif rnd <= np.sum(R[:30]):  # Styryl formation  X[1] (hydrogen abstraction)
            reacao = "R[46]"
            dog30 += 1  

            i = int(np.random.uniform(0.0, 1.0)*(X[1]))

            if (i) == (0):
                for j in range(1,len(D1Unsat),1):
                    if D1Unsat[int(j)] > 0: break
            else:
                SomaD = 0
                for j in range(1,len(D1Unsat),1):
                    SomaD += D1Unsat[int(j)]
                    if SomaD >= i: break

            # if j < 3: 
            #     print("R[15]: Tamanho do polímero não pode ser menor que 3")
            #     continue
            # else:
            #     PositionMidRadical = int(np.random.rand()*(j))
            #     if int(PositionMidRadical) == 0: continue # RAISING ERRORS 
            #     MID.append((j,int(PositionMidRadical)))

            #     X[20] -= 1  # Styryl radical (very unlikely)
            #     X[12] += 1  # Monomer (styrene)
            #     X[5]  += 1  # Polymer mid-chain radical

            #     X[1] -= 1
            #     D1Unsat[int(j)] -= 1
            #     SomaD1Unsat -= int(j)
            #     MID_Unsat.append(1)
            PositionMidRadical = int(np.random.rand()*(j))
            while int(PositionMidRadical) == 0: PositionMidRadical = int(np.random.rand()*(j))
            MID.append((j,int(PositionMidRadical)))

            X[20] -= 1  # Styryl radical (very unlikely)
            X[12] += 1  # Monomer (styrene)
            X[5]  += 1  # Polymer mid-chain radical

            X[1] -= 1
            D1Unsat[int(j)] -= 1
            SomaD1Unsat -= int(j)
            MID_Unsat.append(1)
        elif rnd <= np.sum(R[:31]):  # Mid-chain scission X[1]
            reacao = "R[7]"
            dog31 += 1
            if X[1] ==(1):
                explicacao = "R2. X[0] == 0 and X[1] == 1 and X[2]==0"
                j = SomaD1Unsat
                i = X[1]
            else:
                i = int(np.random.uniform(0.0, 1.0)*(X[1])) # -- MID-CHAIN SCISSION WILL BE OCCURING IN POLYMER i
                if (i) ==(0):
                    explicacao = "R0: i=0, X[1] > 0"
                    for j in range(0,len(D1Unsat),1):
                        if D1Unsat[int(j)] > 0: break
                # else:
                #     if    (X[1]) == (1):
                #         explicacao = "R0: i = (X[0] + X[1]). X[1]=1"
                #         j = SomaD1Unsat
                #     elif  (i) == (X[1]):
                #         explicacao = "R0: i = (X[0] + X[1])"
                #         for j in range(len(D1Unsat)-1,0,-1):
                #             if int(D1Unsat[int(j)]) > int(0): break
                else:
                    explicacao = "R0. i <= (X[0] + X[1])*?"
                    SomaD = int(0)
                    for j in range(1,len(D1Unsat),1):
                        SomaD += int(D1Unsat[int(j)])
                        if SomaD >= i: break
                    # if (D1Unsat[int(j)] - 1) < (0):
                    #     print("CONTINUE! R[0]-X[1]: D1Unsat[int(j)] - 1 < 0. D1Unsat[int(j)]:", D1Unsat[int(j)])
                    #     print(explicacao)
                    #     break
            
            # -- ESCOLHENDO ONDE ESSE POLÍMERO i VAI QUEBRAR
            l = int(np.random.rand()*(j-1))  # l = int(np.random.rand()*D[j-1])  # np.random.rand is a uniform distribution over [0, 1)
            # if i == (0) or l <= 1: continue # Error in random number generation (i) or the polymer will be breaking at the end (l <= 1)
            while l <= 1: l = int(np.random.rand()*(j-1)) 

            if (j - l) == (1):
                X[10] += 1  # Benzyl radical =  Tolune radical (TOLradical)

                if (l) == (1): 
                    X[9] += 1  # Allyl Benzene Radical 
                else:
                    Pprimary.append(l)
                    X[4] += 1   # Polymer Radical (Primary)
                    P1_Unsat.append(1)
            else:
                # POLYMER RADICAL (PRIMARY)
                Pprimary.append(l)
                X[4] += 1
                P1_Unsat.append(1)
                # POLYMER RADICAL (SECONDARY)
                P.append(j-l)
                X3length += j - l
                X[3] += 1
                P2_Unsat.append(0)

            X[1] -= 1
            D1Unsat[int(j)] -= 1
            SomaD1Unsat -= (j)

        elif rnd <= np.sum(R[:32]):  # alpha-Methylstyrene FORMATION X[2], hydrogen abstraction 
            reacao = "R[31]"
            dog32 += 1  
            
            # if (X[2]) == int(1): i = int(1)
            # else: i = int(np.random.uniform(0.0, 1.0)*(X[2])) 
            i = int(np.random.uniform(0.0, 1.0)*(X[2])) 
                        
            reacao = "R[18]"
            # if (X[2]) == int(1):
            #     j = SomaD2Unsat
            if (i) == (0):
                for j in range(1,len(D2Unsat),1):
                    if D2Unsat[int(j)] > 0: break                        
            else:
                SomaD = 0              
                for j in range(0,len(D2Unsat),1):
                    SomaD += (D2Unsat[int(j)])
                    if SomaD >= i: break
            
            PositionMidRadical = int(np.random.rand()*(j))
            while int(PositionMidRadical) == 0: PositionMidRadical = int(np.random.rand()*(j))
            MID.append((j,int(PositionMidRadical)))

            X[9]  -= 1  # Allyl benzene radical  # X[13] -= 1 # Tolune radical (TOLradical)# TOLradical
            X[5]  += 1  # MID-CHAIN RADICAL   
            X[13] += 1  # ALPHA-METHYLSTYRENE

            X[2] -= 1
            D2Unsat[int(j)] -= 1
            SomaD2Unsat -= int(j) 
            MID_Unsat.append(2)
        elif rnd <= np.sum(R[:33]):  # DPP formation X[2] (hydrogen abstraction)
            # print("R15: DPP formation  (hydrogen abstraction)")
            reacao = "R[24]"
            dog33 += 1  

            i = int(np.random.uniform(0.0, 1.0)*(X[2]))
            if (i) == (0):
                for j in range(1,len(D2Unsat),1):
                    if D2Unsat[int(j)] > 0: break  
            # elif i == X[2]:
            #     for j in range(len(D2Unsat)-1,0,-1):
            #         if D2Unsat[int(j)] > 0: break
            else:
                SomaD = 0                
                for j in range(0,len(D2Unsat),1):
                    SomaD += (D2Unsat[int(j)])
                    if SomaD >= i: break


            PositionMidRadical = int(np.random.rand()*(j))
            while int(PositionMidRadical) == 0: PositionMidRadical = int(np.random.rand()*(j))    # Entra muito aqui      # RAISING ERRORS        
            MID.append((j,int(PositionMidRadical)))

            X[11] -= 1  # DPP radical
            X[15] += 1  # DPP 
            X[5]  += 1  # Polymer mid-chain radical
                       
            X[2] -= 1
            D2Unsat[int(j)] -= 1
            SomaD2Unsat -= int(j)         
            MID_Unsat.append(2)
        elif rnd <= np.sum(R[:34]):  # Hydrogen Abstraction: Primary Radical X[1]
            reacao = "R[33]"
            dog34 += 1  
            i = int(np.random.uniform(0.0, 1.0)*X[4])
            j = int(np.random.uniform(0.0, 1.0)*(X[1])) 

            if (j) == (0): 
                explicacao = "j == 0, X[1] > 0"
                for l in range(1,len(D1Unsat),1):
                    if (D1Unsat[int(l)]) > int(0): break                
            else:
                explicacao = "j < X[0] + X[1]"
                SomaD = 0
                for l in range(0,len(D1Unsat),1):
                    SomaD += (D1Unsat[int(l)])
                    if SomaD >= j: break

            PositionMidRadical = int(np.random.rand()*(l))
            while int(PositionMidRadical) == int(0): PositionMidRadical = int(np.random.rand()*(l))

            if int(P1_Unsat[i]) == int(0):
                D[int(Pprimary[i])] += int(1)
                Dsoma += int(Pprimary[i])
                X[0] += 1
            elif int(P1_Unsat[i]) == int(1):
                D1Unsat[int(Pprimary[i])] += 1
                SomaD1Unsat += int(Pprimary[i])
                X[1] += 1
            else:
                D2Unsat[int(Pprimary[i])] += 1
                SomaD2Unsat += int(Pprimary[i])
                X[2] += 1    
            Pprimary.pop(i)
            P1_Unsat.pop(i)
            X[4] -= 1

            # if int(1) == (X[1]): SomaD1Unsat  == int(0)
            # else: SomaD1Unsat  -= int(l)
            SomaD1Unsat  -= int(l)
            D1Unsat[int(l)] -= 1
            X[1] -= 1
            MID_Unsat.append(1)
            MID.append((l,int(PositionMidRadical)))
            X[5] += 1   # Polymer mid-chain radical 
        elif rnd <= np.sum(R[:35]):  # Termination by disproportionation (2nd x 2nd)
            # print("R6: Termination by disproportionation")
            reacao = "R[11]"
            dog35 += 1  
            m, n = int(np.random.uniform(0.0, 1.0)*(X[3])), int(np.random.uniform(0.0, 1.0)*(X[3]))
            # if m < 0 or n < 0: print("M OR N IS <0: ERRROOOR! X[3]:", X[3], "m: ", m, "n: ", n)
            if m == n:
                if m == 0: n += 1
                else: n -= 1

            # if P[m] == int(1): # Isso não é possível
            #     print("R[6]! P[m] == 1")
            #     break 
            # elif P[n] == int(1): 
            #     print("R[6]! P[n] == 1")
            #     break                 
            # else:
            # Tô assumindo que o m vai virar insaturado no final 
            if P2_Unsat[m] == 1:
                D2Unsat[int(P[m])] += 1
                SomaD2Unsat += int(P[m])
                X[2] += 1
            else: 
                D1Unsat[int(P[m])] += 1
                SomaD1Unsat += int(P[m])
                X[1] += 1

            if P2_Unsat[n] == 1:
                D1Unsat[int(P[n])] += 1
                SomaD1Unsat += int(P[n])
                X[1] += 1
            else: 
                D[int(P[n])] += 1
                Dsoma += int(P[n])
                X[0] += 1                


            if m > n: ind = [m, n]
            else: ind = [n, m]
            for index in ind: 
                P.pop(index)
                P2_Unsat.pop(index)
            X[3] -= 2        
            X3length = X3length - m - n
        elif rnd <= np.sum(R[:36]):  # Styryl formation  X[2] (hydrogen abstraction)
            reacao = "R[32]"
            dog36 += 1  

            i = int(np.random.uniform(0.0, 1.0)*(X[2]))

            if (i) == (0):
                for j in range(1,len(D2Unsat),1):
                    if D2Unsat[int(j)] > 0: break
            # elif i == X[2]:
            #     for j in range(len(D2Unsat)-1,0,-1):
            #         if D2Unsat[int(j)] > 0: break
            else:
                SomaD = 0              
                for j in range(0,len(D2Unsat),1):
                    SomaD += (D2Unsat[int(j)])
                    if SomaD >= i: break       
            
            PositionMidRadical = int(np.random.rand()*(j))
            while int(PositionMidRadical) == 0: PositionMidRadical = int(np.random.rand()*(j))            
            MID.append((j,int(PositionMidRadical)))
            X[20] -= 1  # Styryl radical (very unlikely)
            X[12] += 1  # Monomer (styrene)
            X[5]  += 1  # Polymer mid-chain radical

            X[2] -= 1
            D2Unsat[int(j)] -= 1
            SomaD2Unsat -= int(j)         
            MID_Unsat.append(2)                                        
        elif rnd <= np.sum(R[:37]):  # Depropagation Rp 
            reacao = "R[41]"
            dog37 += 1  
            i = int(np.random.uniform(0.0, 1.0)*X[4])     # X[4]: Polymer radical (Primary)                 
            if int(Pprimary[i]) == int(2): # If it is a dimer 
                explicacao = "R30: P[i] == 2"
                X[4]  -= int(1)     # Polymer Radical (Primary)
                X[12] += int(1)     # Monomer
                if int(P1_Unsat[i]) == int(0):
                    X[17] += 1 # Ethylbenzene radical
                else: 
                    X[9] += 1 # Allyl Benzene Radical
                Pprimary.pop(i)
                P1_Unsat.pop(i)
            else:
                explicacao   = "30: P[i] > 2"
                Pprimary[i]  = Pprimary[i] - int(1)
                X[12] += int(1)  # Monomer

        elif rnd <= np.sum(R[:38]):  # Mid-chain scission X[2]
            reacao = "R[8]"
            dog38 += 1
            explicacao = "nenhuma"

            if X[2]==int(1):
                explicacao = "R2. X[0] == 0 and X[1] == 0 and X[2]== 1"
                j = SomaD2Unsat
                i = X[2]
            else:
                i = int(np.random.uniform(0.0, 1.0)*(X[2])) # -- MID-CHAIN SCISSION WILL BE OCCURING IN POLYMER i
                if  (i) ==(0):
                    explicacao = "R0: i=0, X[2] > 0"
                    for j in range(0,len(D2Unsat),1):
                        if D2Unsat[int(j)] > 0: break

                else:
                    explicacao = "R0. i < (X[0] + X[1] + X[2])"
                    SomaD = 0
                    for j in range(0,len(D2Unsat),1):
                        SomaD += (D2Unsat[int(j)])
                        if SomaD >= i: break
            
            # -- ESCOLHENDO ONDE ESSE POLÍMERO i VAI QUEBRAR
            l = int(np.random.rand()*(j-1))  # l = int(np.random.rand()*D[j-1])  # np.random.rand is a uniform distribution over [0, 1)
            # if i == (0) or l <= 1: continue # Error in random number generation (i) or the polymer will be breaking at the end (l <= 1)
            while l <= 1: l = int(np.random.rand()*(j-1)) 

            if (j - l) == (1):
                X[9] += 1 # Allyl Benzene Radical
                if (l) == (1): 
                    X[9] += 1  # Allyl Benzene Radical 
                else:
                    Pprimary.append(l)
                    X[4] += 1   # Polymer Radical (Primary)
                    P1_Unsat.append(1)
            else:
                # POLYMER RADICAL (PRIMARY)
                Pprimary.append(l)
                X[4] += 1
                P1_Unsat.append(1)
                # POLYMER RADICAL (SECONDARY)
                P.append(j-l)
                X3length += (j-l)
                X[3] += 1
                P2_Unsat.append(1)
            
            X[2] -= 1
            D2Unsat[int(j)] -= 1
            SomaD2Unsat  -= (j)
        elif rnd <= np.sum(R[:39]):  # Hydrogen Abstraction: Primary Radical X[2]
            reacao = "R[34]"
            dog39 += 1  
            i = int(np.random.uniform(0.0, 1.0)*X[4])
            j = int(np.random.uniform(0.0, 1.0)*(X[2])) 

            if (j) == (0):
                explicacao = "j == 0, X[2] > 0"
                for l in range(1,len(D2Unsat),1):
                    if (D2Unsat[int(l)]) > int(0): break
            else: 
                explicacao = "j < X[0] + X[1] + X[2]"
                SomaD = 0             
                for l in range(0,len(D2Unsat),1):
                    SomaD += (D2Unsat[int(l)])
                    if SomaD >= j: break

            PositionMidRadical = int(np.random.rand()*(l))
            while int(PositionMidRadical) == int(0): PositionMidRadical = int(np.random.rand()*(l))
            if int(P1_Unsat[i]) == int(0):
                D[int(Pprimary[i])] += int(1)
                Dsoma += int(Pprimary[i])
                X[0] += 1
            elif int(P1_Unsat[i]) == int(1):
                D1Unsat[int(Pprimary[i])] += 1
                SomaD1Unsat += int(Pprimary[i])
                X[1] += 1
            else:
                D2Unsat[int(Pprimary[i])] += 1
                SomaD2Unsat += int(Pprimary[i])
                X[2] += 1     
            Pprimary.pop(i)
            P1_Unsat.pop(i)
            X[4] -= 1

            # if int(1) == (X[2]): SomaD2Unsat == int(0)
            # else: SomaD2Unsat  -= int(l)
            SomaD2Unsat  -= int(l)
            D2Unsat[int(l)] -= 1
            X[2] -= 1
            MID_Unsat.append(2)
            MID.append((l,int(PositionMidRadical)))
            X[5] += 1   # Polymer mid-chain radical 
        elif rnd <= np.sum(R[:40]):  # Termination by combination (radical recombination): Primary + Secondary   
            reacao = "R[36]"     
            dog40 += 1  
            m, n = int(np.random.rand()*X[3]), int(np.random.rand()*X[4])


            if int(P[m]+Pprimary[n]) > int(len(D)):
                maior_que_DP += 1
                if (P2_Unsat[m] + P1_Unsat[n]) == int(0):
                    D[-1] += int(1)
                    Dsoma += int(len(D))
                    X[0] += 1
                elif int(P2_Unsat[m] + P1_Unsat[n]) == int(1):
                    D1Unsat[-1] += int(1)
                    SomaD1Unsat += int(len(D))
                    X[1] += 1
                else:
                    D2Unsat[-1] += int(1)
                    SomaD2Unsat += int(len(D))
                    X[2] += 1                        
            else:
                if int(P2_Unsat[m] + P1_Unsat[n]) == int(0):
                    D[int(P[m]+Pprimary[n])] += 1
                    Dsoma += int(P[m]+Pprimary[n])
                    X[0] += 1
                elif int(P2_Unsat[m] + P1_Unsat[n]) == int(1):
                    D1Unsat[int(P[m]+Pprimary[n])] += 1
                    SomaD1Unsat += int(P[m]+Pprimary[n])
                    X[1] += 1
                else:
                    D2Unsat[int(P[m]+Pprimary[n])] += 1
                    SomaD2Unsat += int(P[m]+Pprimary[n])
                    X[2] += 1                              

            P.pop(m)
            P2_Unsat.pop(m)
            Pprimary.pop(n)
            P1_Unsat.pop(n)
            X3length -= m
            X[3] -= 1
            X[4] -= 1
        elif rnd <= np.sum(R[:41]):  # Ethylbenzene formation X[0] (hydrogen abstraction)
            reacao = "R[42]"
            dog41 += 1  

            i = int(np.random.uniform(0.0, 1.0)*(X[0]))

            if (i) == (0):
                for j in range(1,len(D),1):
                    if D[int(j)] > 0: break
            else:
                SomaD = int(0)
                for j in range(1,len(D),1):
                    SomaD += (D[int(j)])
                    if SomaD >= i: 
                        # if D[int(j)] <= 0: print("R15: D[int(j)] - 1 < 0. i:", i, "j:", j, "X[0]", X[0], ". D[:5]", D[int(j)])
                        break    

            
            PositionMidRadical = int(np.random.rand()*(j))
            while int(PositionMidRadical) == 0: PositionMidRadical = int(np.random.rand()*(j))      
            MID.append((j,int(PositionMidRadical)))

            X[17] -= 1  # Ethylbenzene radical
            X[18] += 1  # Ethylbenzene
            X[5]  += 1  # Polymer mid-chain radical

            X[0] -= 1  # -1 Polymer
            D[int(j)] -= 1
            Dsoma -= int(j)
            MID_Unsat.append(0)
        # 41        
        elif rnd <= np.sum(R[:42]):     # k13
            # print("R7: k13")
            dog42 += 1  
            reacao = "R[12]"
            m = int(np.random.uniform(0.0, 1.0)*(X[3]))
            if P[m] <= 3: continue

            RAD13.append(P[m])
            RAD13_Unsat.append(P2_Unsat[m])
            P.pop(m)
            P2_Unsat.pop(m)
            X3length -= m
            X[3] -= 1                   # Polymer radical (primary + secondary radical)
            X[6] += 1                   # Mid-chain 13 radical         
        elif rnd <= np.sum(R[:43]):     # Ethylbenzene formation X[1]  (hydrogen abstraction)
            reacao = "R[43]"
            dog43 += 1  

            i = int(np.random.uniform(0.0, 1.0)*(X[1]))

            if (i) == (0):
                for j in range(1,len(D1Unsat),1):
                    if D1Unsat[int(j)] > 0: break
            else:
                SomaD = 0
                for j in range(1,len(D1Unsat),1):
                    SomaD += D1Unsat[int(j)]
                    if SomaD >= i: break

            PositionMidRadical = int(np.random.rand()*(j))
            while int(PositionMidRadical) == 0: PositionMidRadical = int(np.random.rand()*(j)) 
            MID.append((j,int(PositionMidRadical)))

            X[17] -= 1  # Ethylbenzene radical
            X[18] += 1  # Ethylbenzene
            X[5]  += 1  # Polymer mid-chain radical

            X[1] -= 1
            D1Unsat[int(j)] -= 1
            SomaD1Unsat -= int(j)
            MID_Unsat.append(1)
        elif rnd <= np.sum(R[:44]):     # Ethylbenzene formation X[2] (hydrogen abstraction)
            reacao = "R[44]"
            dog44 += 1  

            i = int(np.random.uniform(0.0, 1.0)*(X[2]))
            if (i) == (0):
                for j in range(1,len(D2Unsat),1):
                    if D2Unsat[int(j)] > 0: break
            else:
                SomaD = 0              
                for j in range(0,len(D2Unsat),1):
                    SomaD += (D2Unsat[int(j)])
                    if SomaD >= i: break
  
            PositionMidRadical = int(np.random.rand()*(j))
            while int(PositionMidRadical) == 0: PositionMidRadical = int(np.random.rand()*(j))       
            MID.append((j,int(PositionMidRadical)))

            X[17] -= 1  # Ethylbenzene radical
            X[18] += 1  # Ethylbenzene
            X[5]  += 1  # Polymer mid-chain radical

            X[2] -= 1
            D2Unsat[int(j)] -= 1
            SomaD2Unsat -= int(j)         
            MID_Unsat.append(2)          
        elif rnd <= np.sum(R[:45]):     # Termination by disproportionation: 1a + 2a        
            reacao = "R[38]"
            dog45 += 1  
            m, n = int(np.random.uniform(0.0, 1.0)*X[3]), int(np.random.uniform(0.0, 1.0)*X[4])

            # if P[m] == 1: # Isso não é possível
            #     print("R[23]! P[m] == 1")
            #     break 
            # elif Pprimary[n] == 1: 
            #     print("R[23]! Pprimary[n] == 1")
            #     break
            # else:   # Como m é referente ao radical segundário, é ele vai virar insaturado 
            if (P2_Unsat[m] + P1_Unsat[n]) == 0:

                D1Unsat[int(P[m])] += 1
                SomaD1Unsat += int(P[m])
                X[1] += 1       

                D[int(Pprimary[n])] += 1
                Dsoma += int(Pprimary[n])
                X[0] += 1

            elif P2_Unsat[m] == 0:
                D1Unsat[int(P[m])] += 1
                SomaD1Unsat += int(P[m])
                X[1] += 1

                D1Unsat[int(Pprimary[n])] += 1
                SomaD1Unsat += int(Pprimary[n])
                X[1] += 1
            else:
                D2Unsat[int(P[m])] += 1
                SomaD2Unsat += int(P[m])
                X[2] += 1       

                D[int(Pprimary[n])] += 1
                Dsoma += int(Pprimary[n])
                X[0] += 1


            P.pop(m)
            Pprimary.pop(n)
            X[4] -= 1
            X[3] -= 1   
            X3length -= m   
            P2_Unsat.pop(m)
            P1_Unsat.pop(n)       
        elif rnd <= np.sum(R[:46]):     # Termination by combination (radical recombination): Primary + Primary 
            reacao = "R[35]"      
            dog46 += 1   
            m, n = int(np.random.rand()*X[4]), int(np.random.rand()*X[4])
            if m == n:
                if m == 0: n += 1
                else: n -= 1

            if int(Pprimary[m]+Pprimary[n]) > len(D):
                maior_que_DP += 1
                if int(P1_Unsat[m] + P1_Unsat[n]) == int(2):
                    X[2] += 1
                    SomaD2Unsat += int(len(D))
                    D2Unsat[-1] += int(1)
                elif int(P1_Unsat[m] + P1_Unsat[n]) == int(1):
                    X[1] += 1
                    SomaD1Unsat += int(len(D))
                    D1Unsat[-1] += int(1)
                else:
                    D[-1] += int(1)
                    Dsoma += int(len(D))
                    X[0] += 1
            else:
                if int(P1_Unsat[m] + P1_Unsat[n]) == int(0):
                    D[int(Pprimary[m]+Pprimary[n])] += 1
                    Dsoma += int(Pprimary[m]+Pprimary[n])
                    X[0] += 1
                elif int(P1_Unsat[m] + P1_Unsat[n]) == int(1):
                    D1Unsat[int(Pprimary[m]+Pprimary[n])] += 1
                    SomaD1Unsat += int(Pprimary[m]+Pprimary[n])
                    X[1] += 1           
                else:
                    D2Unsat[int(Pprimary[m]+Pprimary[n])] += 1
                    SomaD2Unsat += int(Pprimary[m]+Pprimary[n])
                    X[2] += 1                                          

            if m > n: ind = [m, n]
            else: ind = [n, m]
            for index in ind: 
                Pprimary.pop(index)
                P1_Unsat.pop(index)
            X[4] -= 2        
        elif rnd <= np.sum(R[:47]):     # Termination by disproportionation: 1a + 1a      
            reacao = "R[37]"  
            dog47 += 1  
            m, n = int(np.random.rand()*X[4]), int(np.random.rand()*X[4])
            if m < 0 or n < 0: print("M OR N IS <0: ERRROOOR! X[22]:", X[4], "m: ", m, "n: ", n)
            if m == n:
                if m == 0: n += 1
                else: n -= 1

            if int(P1_Unsat[m]) == int(0):    # Tô afirmando que 'm' vai virar insaturado
                D1Unsat[int(Pprimary[m])] += 1
                SomaD1Unsat += int(Pprimary[m])
                X[1] += 1
            else:
                D2Unsat[int(Pprimary[m])] += 1
                SomaD2Unsat += int(Pprimary[m])
                X[2] += 1

            if int(P1_Unsat[n]) == int(0):
                D[int(Pprimary[n])] += 1
                Dsoma += int(Pprimary[n])
                X[0] += 1
            else:
                D1Unsat[int(Pprimary[m])] += 1
                SomaD1Unsat += int(Pprimary[m])
                X[1] += 1                    

            if m > n: ind = [m, n]
            else: ind = [n, m]
            for index in ind: 
                Pprimary.pop(index)
                P1_Unsat.pop(index)
            X[4] -= 2
        elif rnd <= np.sum(R[:48]): # TOLradical formation (DPPradical beta scission)        
            # print("R16: TOLradical formation (DPPradical beta scission)")
            dog48 += 1  
            X[11]  -= 1 # DPP radical
            X[12]  += 1 # Monomer
            X[10]  += 1 # X[13] += 1 # Tolune radical (TOLradical)# TOLradical

            reacao = "R[25]"
            
        # ----------------------------------------------------------------------------
        elif rnd <= np.sum(R[:49]):  # Styrene thermal initiation
            # print("Styrene thermal initiation")
            X[12] -= 3   # Styrene consumption          
            X[17] += 1   # Ethylbenzene radical
            X[23] += 1   # Dimer radical
        elif rnd <= np.sum(R[:50]): # Termination of ethylbenzene radicals 
            X[17] -= 2   # Ethylbenzene radical
            X[14] += 1   # Dimer 
        elif rnd <= np.sum(R[:51]): # Termination of dimer radicals 
            X[23] -= 2   # Dimer radical
            X[0]  += 1   # Tetramer (não é bem X[0] né, já que é um naphthene) 
            D[4]  += 1
            Dsoma += 1   # (D[4])
        elif rnd <= np.sum(R[:52]): # Termination of dimer + styrene radicals 
            X[17] -= 1   # Ethylbenzene radical
            X[23] -= 1   # Dimer radical
            X[16] += 1   # Trimer
        elif rnd <= np.sum(R[:53]): # Hydrogen abstraction of styrene radical (attacking X[0])
            i = int(np.random.uniform(0.0, 1.0)*(X[0]))
            if (i) == (0):
                for j in range(1,len(D),1):
                    if D[int(j)] > 0: break
            else:
                SomaD = int(0)
                for j in range(1,len(D),1):
                    SomaD += (D[int(j)])
                    if SomaD >= i: break    

            
            PositionMidRadical = int(np.random.rand()*(j))
            while int(PositionMidRadical) == 0: PositionMidRadical = int(np.random.rand()*(j))      
            MID.append((j,int(PositionMidRadical)))

            X[17] -= 1  # Ethylbenzene radical
            X[18] += 1  # Ethylbenzene 
            X[5]  += 1  # Polymer mid-chain radical
            X[0] -= 1
            D[int(j)] -= 1
            Dsoma -= int(j)
            MID_Unsat.append(0)    
        else: # Hydrogen abstraction of dimer radical 
            i = int(np.random.uniform(0.0, 1.0)*(X[0]))
            if (i) == (0):
                for j in range(1,len(D),1):
                    if D[int(j)] > 0: break
            else:
                SomaD = int(0)
                for j in range(1,len(D),1):
                    SomaD += (D[int(j)])
                    if SomaD >= i: break    

            
            PositionMidRadical = int(np.random.rand()*(j))
            while int(PositionMidRadical) == 0: PositionMidRadical = int(np.random.rand()*(j))      
            MID.append((j,int(PositionMidRadical)))

            X[23] -= 1  # Dimer radical
            X[14] += 1  # Dimer
            X[5]  += 1  # Polymer mid-chain radical
            X[0] -= 1
            D[int(j)] -= 1
            Dsoma -= int(j)
            MID_Unsat.append(0)                 

        X[12] += D[1] + D1Unsat[1] + D2Unsat[1] 
        Dsoma -= (D[1]) 
        SomaD1Unsat -= (D1Unsat[1])
        SomaD2Unsat -= (D2Unsat[1])     
        X[0] -= D[1]
        X[1] -= D1Unsat[1]
        X[2] -= D2Unsat[1]
        D[1], D1Unsat[1], D2Unsat[1] = 0, 0, 0


        X[14]  = X[14] + D[2] + D1Unsat[2] + D2Unsat[2]                    # Dimer
        Dsoma -= (2*D[2]) 
        SomaD1Unsat -= (2*D1Unsat[2])
        SomaD2Unsat -= (2*D2Unsat[2])
        X[0] -= D[2]
        X[1] -= D1Unsat[2]
        X[2] -= D2Unsat[2]
        D[2], D1Unsat[2], D2Unsat[2] = 0, 0, 0

        X[16]  = X[16] + D[3] + D1Unsat[3] + D2Unsat[3]                     # Trimer 
        Dsoma -= (3*D[3])
        SomaD1Unsat -= (3*D1Unsat[3])
        SomaD2Unsat -= (3*D2Unsat[3])
        X[0] -= D[3] 
        X[1] -= D1Unsat[3]
        X[2] = X[2]- D2Unsat[3]
        D[3], D1Unsat[3], D2Unsat[3]  = 0, 0, 0

        EvaporationFactor = 0.8
        print(X[12], X[26], X[12]*EvaporationFactor)
        if X[19] > 0:
            X[24] += X[19]*EvaporationFactor    # Toluene evaporated
            X[19] = X[19]*(1-EvaporationFactor)
        if X[18] > 0:
            X[24] += X[18]*EvaporationFactor    # Ethylbenzene evaporated   
            X[18] = X[18]*(1-EvaporationFactor)
        if X[12] > 0:
            X[26] += X[12]*EvaporationFactor    # Styrene evaporated
            X[12] = X[12]*(1-EvaporationFactor)
        if X[13] > 0:
            X[27] += X[13]*EvaporationFactor    # aMS evaporated
            X[13] = X[13]*(1-EvaporationFactor)
        if X[14] > 0:
            X[28] += X[14]*EvaporationFactor    # Dimer evaporated
            X[14] = X[14]*(1-EvaporationFactor)
        if X[15] > 0:
            X[29] += X[15]*EvaporationFactor    # DPP evaporated
            X[15] = X[15]*(1-EvaporationFactor)
        if X[16] > 0:
            X[30] += X[16]*EvaporationFactor    # Trimer evaporated
            X[16] = X[16]*(1-EvaporationFactor)
        
        
                        
        

        ## Tetramer has a boiling point of over 761 degrees 

        sumCCbonds = (Dsoma + SomaD1Unsat + SomaD2Unsat) 

        R[0] =  kbs*X[5]*2                   # Mid-chain beta scission (unimolecular reaction) (the midchain radical can undergo midchain â-scission on either side of the midchain radical, thus the factor of 2)
        R[1] =  kra*(X[1])*X[3]              # General radical addition X[1] (opposite of depropagation) 
        R[2] =  kra*(2*X[2])*X[3]            # General radical addition X[2] (opposite of depropagation)
        R[3] =  kH*X[3]*(Dsoma-2*X[0])       # HYDROGEN ABSTRACTION (L/mol*seg) X[0]  (bimolecular reactions between different molecules)
        R[4] =  kdp*X[3]                     # Depropagation (end chain β-scission) (s-1) (unimolecular reactions)
        R[5] =  kH*X[3]*(SomaD1Unsat-2*X[1])          # HYDROGEN ABSTRACTION (L/mol*seg) X[1]  (bimolecular reactions between different molecules)
        R[6] =  kH*X[3]*(SomaD2Unsat-2*X[2])           # HYDROGEN ABSTRACTION (L/mol*seg) X[2]  (bimolecular reactions between different molecules)
        R[7] =  kbs0_LMWS*X[6]               # Formation of dimer  or TOLradical (mid-chain beta scission of 13 mid chain)
        R[8] =  kH_Rp*X[10]*(Dsoma-2*X[0])   # TOL formation (HYDROGEN ABSTRACTION) X[0]   # kH(pb,tb)
        R[9] =  kbs0_LMWS*X[7]             # Formation of trimer or DPPradical (mid-chain beta scission of 15 mid chain)
        R[10] =  k15*X[3]
        R[11] =  kbra*(X[1])*X[10]     # Benzyl radical addition X[1]
        R[12] =  kbra*(2*X[2])*X[10]   # Benzyl radical addition X[2]
        R[13] =  ktc*X[3]*(X[3]-1)/2    # Termination by combination (radical recombination) (L/mol*seg) (bimolecular reactions between equal molecules)
        R[14] =  kH_Rp*X[10]*(SomaD1Unsat-2*X[1])            # TOL formation (HYDROGEN ABSTRACTION) X[1]   # kH(pb,tb)
        R[15] =  kH*X[9]*(Dsoma-2*X[0])  # alpha-Methylstyrene formation X[0]  
        R[16] =  kH*X[11]*(Dsoma-2*X[0]) # DPP formation  (HYDROGEN ABSTRACTION) (bimolecular reactions between different molecules)
        R[17] =  kfs*X[1]                     # End chain scisson (or chain fission allyl) (carbon-hydrogen bond fission) (s-1) (Dn -> M + Pn-1) (unimolecular reactions)
        R[18] =  kdppa*(X[1])*X[11]       # DPP radical addition: dppRAD + PSunsat -> Rtb (1,5)  [kadd(sb,tb)]
        R[19] =  kH_Rp*X[10]*(SomaD2Unsat-2*X[2])            # TOL formation (HYDROGEN ABSTRACTION) X[2]   # kH(pb,tb)
        R[20] =  kfs*(2*X[2])                 # End chain scisson (or chain fission allyl) (carbon-hydrogen bond fission) (s-1) (Dn -> M + Pn-1) (unimolecular reactions)
        R[21] =  kH*X[20]*(Dsoma-2*X[0])           # Styryl reaction (formation of styrene)  (HYDROGEN ABSTRACTION)
        R[22] =  kdppa*(2*X[2])*X[11]     # DPP radical addition: dppRAD + PSunsat -> Rtb (1,5)  [kadd(sb,tb)]
        R[23] =  kf*(Dsoma-2*X[0])                     # Mid chain scisson (or 'chain fission') X[0] (s-1) (Dn -> Pr + Pn-r) (unimolecular reactions)
        # R[24] =  k17*(X[3]-P.count(4)-P.count(5)-P.count(6)-P.count(7))                    # Mid-chain 17 radical
        R[24] =  k17*(X[3])                    # Mid-chain 17 radical        
        R[25] =  kH*X[9]*(SomaD1Unsat-2*X[1])             # alpha-Methylstyrene formation X[1]
        R[26] =  kH*X[11]*(SomaD1Unsat-2*X[1])          # DPP formation  (HYDROGEN ABSTRACTION) (bimolecular reactions between different molecules)
        R[27] =  kH_Rp*X[4]*(Dsoma-2*X[0])                # HYDROGEN ABSTRACTION (L/mol*seg) X[0]  - Primary radical - (bimolecular reactions between different molecules)     
        R[28] =  k73*X[8]                    # Mid-chain 73 radical = Mid-chain 13 radical           
        R[29] =  kH*X[20]*(SomaD1Unsat-2*X[1])            # Styryl reaction (formation of styrene)  (HYDROGEN ABSTRACTION)
        R[30] =  kf*(SomaD1Unsat-2*X[1])                # Mid chain scisson (or 'chain fission') X[1] (s-1) (Dn -> Pr + Pn-r) (unimolecular reactions)
        R[31] =  kH*X[9]*(SomaD2Unsat-2*X[2])             # alpha-Methylstyrene formation X[2]
        R[32] = kH*X[11]*(SomaD2Unsat-2*X[2])            # DPP formation  (HYDROGEN ABSTRACTION) (bimolecular reactions between different molecules)
        R[33] =  kH_Rp*X[4]*(SomaD1Unsat-2*X[1])           # HYDROGEN ABSTRACTION (L/mol*seg) X[1] - Primary radical - (bimolecular reactions between different molecules)     
        R[34] =  ktd*X[3]*(X[3]-1)/2    # Termination by disproportionation (L/mol*seg) (bimolecular reactions between equal molecules)
        R[35] =  kH*X[20]*(SomaD2Unsat-2*X[2])            # Styryl reaction (formation of styrene)  (HYDROGEN ABSTRACTION)
        R[36] =  kdp_Rp*X[4]                   # Depropagation Pprimary        
        R[37] =  kf*(SomaD2Unsat-2*X[2])                # Mid chain scisson (or 'chain fission') X[2] (s-1) (Dn -> Pr + Pn-r) (unimolecular reactions)
        R[38] =  kH_Rp*X[4]*(SomaD2Unsat-2*X[2])           # HYDROGEN ABSTRACTION (L/mol*seg) X[2] - Primary radical - (bimolecular reactions between different molecules)     
        R[39] =  ktc*X[3]*X[4]                 # Termination by combination (1a + 2a) (radical recombination) (L/mol*seg) (bimolecular reactions between equal molecules)
        R[40] =  kH*X[17]*(Dsoma-2*X[0])                 # Ethylbenzene formation  (HYDROGEN ABSTRACTION)
        R[41] =  k13*X[3]
        R[42] =  kH*X[17]*(SomaD1Unsat-2*X[1])           # Ethylbenzene formation  (HYDROGEN ABSTRACTION)
        R[43] =  kH*X[17]*(SomaD2Unsat-2*X[2])            # Ethylbenzene formation  (HYDROGEN ABSTRACTION)
        R[44] =  ktd*X[3]*X[4]                 # Termination by disproportionation (1a + 1a) (L/mol*seg) (bimolecular reactions between equal molecules)
        R[45] =  ktc*X[4]*(X[4]-1)/2           # Termination by combination (1a + 1a) (radical recombination) (L/mol*seg) (bimolecular reactions between equal molecules)
        R[46] =  ktd*X[4]*(X[4]-1)/2           # Termination by disproportionation (1a + 1a) (L/mol*seg) (bimolecular reactions between equal molecules)
        R[47] =  0.0 # kbs0_LMWS*X[11]               # TOLradical formation (beta scission of DPPradical) (unimolecular reaction)



        # Still missing 
        # - Termination by disproportion
        # - Hydrogen abstraction X[1] e X[2]
        

        # -------------------------------------- Thermal Initiation 
        t_temp_thermalinit += t 
        
        factor = 1
        R[48] =  kdmMC * (X[12]*factor) * ((X[12]*factor) - 1) * ((X[12]*factor) - 2)/6.0  # Styrene thermal initiation 
        R[49] =  ktc*X[17]*(X[17]-1)/2                          # Termination by combination (ethylbenzene radicals)
        R[50] =  ktc*X[23]*(X[23]-1)/2                          # Termination by combination (dimer radicals)
        R[51] =  ktc*X[23]*X[17]                                # Termination by combination (styrene + dimer radicals)
        R[52] =  0.0 # There are already reactions for ethylbenzene radical # kH*X[22]*(Dsoma-2*X[0])       # HYDROGEN ABSTRACTION by styrene radical attack (forming styrene) (L/mol*seg) X[0]
        R[53] =  kH*X[23]*(Dsoma-2*X[0])       # HYDROGEN ABSTRACTION by dimer radical attack (forming dimer) (L/mol*seg) X[0]
        
        
        # -------------------------------------------------------------------------
        
        
        
        
        
            
        j, l, i = int(0), int(0), int(0)       
        
        if t>= time_interval[ll]:
            MWSty = 104.15 # g/mol # molecular weight of styrene
            lambda2 = 0.0
            lambda1 = 0.0
            lambda0 = 0.0
            for i in range(len(D)):
                lambda2 = lambda2 + i**2*(D[i] + D1Unsat[i] + D2Unsat[i])
                lambda1 = lambda1 + i*(D[i] + D1Unsat[i] + D2Unsat[i])
                lambda0 = lambda0 + (D[i] + D1Unsat[i] + D2Unsat[i])
            
            if X[3] >= int(1):
                for j in range(0, len(P), 1):
                    i = int(P[int(j)])
                    lambda2 += i*i
                    lambda1 += i
                    lambda0 += 1
            
            if X[4] >= 1:
                for j in range(0, len(Pprimary), 1):
                    i = int(Pprimary[int(j)])
                    lambda2 += i*i
                    lambda1 += i
                    lambda0 += 1
            
            if X[5] >= 1:
                for j in range(0, X[5], 1):
                    i = int(MID[int(j)][0])
                    lambda2 += i*i
                    lambda1 += i
                    lambda0 += 1 
                    
                    ## MID[i][0]: tamanho da cadeia
                    ## MID[i][1]: onde tá o radical intermediário
                    
            if X[6] >= 1: 
                for j in range(len(RAD13)):
                    i = RAD13[int(j)]
                    lambda2 += i**2
                    lambda1 += i
                    lambda0 += 1 
                    
            if X[7] >= 1:  
                for j in range(len(RAD15)):
                    i = RAD15[int(j)]
                    lambda2 += i**2
                    lambda1 += i
                    lambda0 += 1                      
                    
            if X[8] >= 1:  
                for j in range(len(RAD17)):
                    i = RAD17[int(j)]
                    lambda2 += i**2
                    lambda1 += i
                    lambda0 += 1  

            if lambda0 > 0 and lambda1 > 0 and lambda2 > 0:
                Mn[ll] = lambda1*MWSty/lambda0
                Mw[ll] = lambda2*MWSty/lambda1
            else:
                Mn[ll], Mw[ll] = 0, 0 

            Styrene[ll] = X[26] # X[12]
            Dimer[ll]   = X[28] # X[14]
            Trimer[ll]  = X[30] # X[16]
            
            DeadPolymer[ll] = (X[0]+X[1]+X[2])
            Rs[ll] = X[3]
            Rp[ll] = X[4]
            Rm[ll] = X[5]
            SumBonds[ll] = sumCCbonds
            
            if (SumBonds[0]-SumBonds[ll])*100/SumBonds[0] > 99.9999: # 97.5
                print("SumBonds[0]-SumBonds[ll]*100/SumBonds[0] > 99.0", (SumBonds[0]-SumBonds[ll])*100/SumBonds[0])
                print("SumBonds[0]", SumBonds[0], "SumBonds[ll]", SumBonds[ll])
                print("time:", t)
                print("X:", X)
                print("  ")
                break 
            ll += (1)

        
        tfim = t
        t += dt

    if X[0] != 0: tfim = t
    lambda2 = 0.0
    lambda1 = 0.0
    lambda0 = 0.0
    for i in range(len(D)):
        lambda2 = lambda2 + i**2*(D[i] + D1Unsat[i] + D2Unsat[i])
        lambda1 = lambda1 + i*(D[i] + D1Unsat[i] + D2Unsat[i])
        lambda0 = lambda0 + (D[i] + D1Unsat[i] + D2Unsat[i])

    if lambda0 > 0 and lambda1 > 0 and lambda2 > 0:
        Mn[-1] = lambda1*MWSty/lambda0
        Mw[-1] = lambda2*MWSty/lambda1
    else:
        Mn[-1], Mw[-1] = 0, 0 

    Styrene[-1] = X[12]
    DeadPolymer[-1] = (X[0]+X[1]+X[2])
    Rs[-1] = X[3]
    Rp[-1] = X[4]
    Rm[-1] = X[5]
    Dimer[-1] = X[14]
    Trimer[-1] = X[16]
    
    Styrene[-1] = X[26] # X[12]
    Dimer[-1]   = X[28] # X[14]
    Trimer[-1]  = X[30] # X[16]
            
    SumBonds[-1] = sumCCbonds
    
    print(D[:10])
            
    return X,P,D,D1Unsat,D2Unsat,Styrene,Dimer,Trimer,DeadPolymer,Rs,Rp,Rm,Mn,Mw,tfim,erro,SumBonds 

Na = 6.022e23              # 1/mol --> Avogadro number
T = 500.0 + 273.15         # K          510
Rcte = 1.987/1000          # kcal/mol K
RT = Rcte*T
MW = 104.15

t = 0.0
if T ==  (400.0 + 273.15):
    tend = 120*60  # seconds
    step = 0.1
elif T ==  (500.0 + 273.15):
    tend = 50 # 44  # seconds
    step = 0.0001
elif T ==  (600.0 + 273.15):
    tend = 0.5  # seconds   # 15*60=900
    step = 0.0001
else: # T ==  (700.0 + 273.15):
    tend = 0.033  # 0.03
    step = 0.000001   
    
time_interval = np.arange(t, tend+step, step)


p_polymer = 1000/(0.9219+5.576e-4*(T-273.15) + 1.0648E-7*(T-273.15)**2) # DENSITY POLYMER (FUNCION OF TEMPERATURE)

p = 0.998  # Extent of reaction 
DP = 6000
x = np.linspace(0,DP,num=DP)  # Degree of polymerization
wx = x*((1-p)**2)*p**(x-1)      # Weight fraction 
print("Soma wx", np.sum(wx))
Mn_Flory = MW/(1-p)          # g/mol
Mw_Flory = (1+p)*MW/(1-p)    # g/mol 
print("Mn Flory:", Mn_Flory)
print("Mw Flory:", Mw_Flory)

V = 1.0e-16

print (" ")
print("Volume: ", V, " L")
print(" ")

D,Dplotar,P0,Styrene_t0,X,R,Mn0,Mw0,Vsample,xPlotar,wx_PS = Moleculas(wx, Na, MW, p_polymer, V) # X, R, D     

#print("Initializing kMC")
print('\033[1m' + 'Initializing kMC' + '\033[0m')



startime = time.time()
X,P,D2,D1_Unsat,D2_Unsat,Styrene,Dimer,Trimer,DeadPolymer,Rs,Rp,Rm,Mn_MC,Mw_MC,tfim,erro,SumBonds = MC(V, Na, RT, T, 0.0,tend,time_interval,X,R,D)


endtime = time.time() - startime
print(" ")
print("tCPU:", endtime/60, " min")

# np.savetxt("500_Styrene" + str(V) + ".txt", Styrene)
# np.savetxt("500_DeadPolymer"+ str(V) + ".txt", DeadPolymer)
# np.savetxt("500_SumBonds"+ str(V) + ".txt", SumBonds)
# np.savetxt("500_timeinterval.txt"+ str(V) + ".txt", time_interval)

print("CONVERSION-WITHOUT-CONSIDERED-POLYMER (%)")
print(" ")
print("Polymer:", np.round((SumBonds[0]-SumBonds[-1])*100/SumBonds[0],2))           # "Polymer:", np.round((X[0]+X[1]+X[2])*100/np.sum(X[3:]),2))
print(" ")
print("Mol %")
print(" ")
print("Monomer:", np.round(X[12]*100/np.sum(X[3:]),2))
print("alpha-Methylstyrene:", np.round(X[13]*100/np.sum(X[3:]),2))
print("DPP:", np.round(X[15]*100/np.sum(X[3:]),2))
print("Dimer:", np.round(X[14]*100/np.sum(X[3:]),2))
print("Trimer:", np.round(X[16]*100/np.sum(X[3:]),2))
print("Toluene:", np.round(X[19]*100/np.sum(X[3:]), 2))
print("Ethylbenzene:", X[18]*100/np.sum(X[3:]))
print("aMS_radical:", X[9]*100/np.sum(X[3:]))               # Allyl benzene (aMS) radical:
print("TOL_radical:", X[10]*100/np.sum(X[3:]))     # Benzyl (TOL) radical
print("DPP_radical:", X[11]*100/np.sum(X[3:]))
print("Polymer_secondary_radical:", X[3]*100/np.sum(X[3:]))
print("Polymer_primary_radical:", X[4]*100/np.sum(X[3:]))
print("Polymer_mid-chain_radical:", X[5]*100/np.sum(X[3:]))
print("Mid-chain_13_radical:", X[6]*100/np.sum(X[3:]))
print("Mid-chain_15_radical:", X[7]*100/np.sum(X[3:]))
print("Mid-chain_17_radical:", X[8]*100/np.sum(X[3:]))
print(" ")

## Importing from Molecules: xPlotar,wx_PS

# xPlotar =    np.linspace(0,len(D),num=len(D))  
x_Dsoma_Ds = np.linspace(0, len(D), len(D))

Dsoma_Ds = np.zeros(len(D))
if (X[0]+X[1]+X[2]) == 0: Dsoma_Ds = Dsoma_Ds
else:
    for i in range(0, len(D2)):       Dsoma_Ds[i] += D2[i]
    for i in range(0, len(D1_Unsat)): Dsoma_Ds[i] += D1_Unsat[i]
    for i in range(0, len(D2_Unsat)): Dsoma_Ds[i] += D2_Unsat[i]
    
    
# wx_PS      =  np.zeros(len(xPlotar))   # NOT DEGRADADED
wx_Dsoma   =  np.zeros(len(Dsoma_Ds))  # DEGRADED (SUM PS SAT + UNSAT + DIUNSAT)
for i in range(0, 6000):
    # if Dplotar[i] > 0:  wx_PS[i]      = D[i]       *(Vsample/V)*(i*MW)/Na  # NOT DEGRADADED
    if Dsoma_Ds[i] > 0: wx_Dsoma[i]   = Dsoma_Ds[i]*(Vsample/V)*(i*MW)/Na  # DEGRADED (SUM PS SAT + UNSAT + DIUNSAT)
    