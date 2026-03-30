import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from KineticConstantsPS import KineticConstants

@jit(nopython=True, cache=True)  # @jit(nopython=True, cache=True)
def sig(x):
    return 1/(1 + np.exp(-x))

@jit(nopython=True, cache=False)  # @jit(nopython=True, cache=True)
def MC(V, Na, RT, T, t, tend,time_interval,X,R,D):

    # print("tend:", tend)



    # evaporaton           = np.linspace(-10, 10, 100000000)
    # evaporaton             = np.linspace(-10, 10, len(time_interval)*1000)
    evaporaton             = np.linspace(-100, 10, 10000)
    evaporation_contador   = 0




    t = 0
    ktecte = KineticConstants(RT)
    kf0, kfs0, kH0, kH0_Rp, kdp0, kdp_Rp0, kbs0 = ktecte[0],  ktecte[1],  ktecte[2], ktecte[3], ktecte[4], ktecte[5], ktecte[6]
    kbs0_LMWS, kc0,  ktd0                       = ktecte[7],  ktecte[8],  ktecte[9]
    k130,      k150, k170                       = ktecte[10], ktecte[11], ktecte[12]
    k730,      kra0, kbra0, kdppa0              = ktecte[13], ktecte[14], ktecte[15], ktecte[16]
    kdm, k11                                    = ktecte[17], ktecte[18]

    print("kf0", kf0)
    print("V:", V)
    print("Na:", Na)

    # ---------- KINETIC CONSTANTS - MICROSCOPIC
    kf  = kf0           # Mid chain scisson (or 'chain fission') (s-1) (Dn -> Pr + Pn-r) (unimolecular reactions)
    kfs = kfs0          # End chain scisson (or chain fission allyl) (carbon-hydrogen bond fission) (s-1) (Dn -> M + Pn-1) (unimolecular reactions)
    kH =  kH0/(V*Na)    # Hydrogen abstraction (L/mol*seg)   (bimolecular reactions between different molecules)
    kdp = kdp0          # Depropagation (end chain β-scission) (s-1) (unimolecular reactions)
    kbs = kbs0          # Mid-chain beta scission
    kbs_LMWS = kbs0_LMWS
    ktc = 2*kc0/(V*Na)  # Termination by combination (radical recombination) (L/mol*seg) (bimolecular reactions between equal molecules)
    ktd = 2*ktd0/(V*Na) # Termination by disproportionation (L/mol*seg) (bimolecular reactions between equal molecules)
    k13 = k130
    k15 = k150
    k17 = k170
    k73 = k730
    kra = kra0/(V*Na)   # General radical addition (bimolecular reactions between different molecules)
    kbra = kbra0/(V*Na) # Benzyl radical addition (bimolecular reactions between different molecules)
    kdppa = kdppa0/(V*Na)
    kH_Rp = kH0_Rp/(V*Na)
    kdp_Rp = kdp_Rp0
    kdmMC = 6.0*kdm/(V*Na)**2.0    # Thermal initiation
    k11MC = k11/(V*Na)             # Styrene propagation



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
    D1Unsat = [int(0)]*len(D)
    D2Unsat = [int(0)]*len(D)


    # Lists that conts the number of C-C bonds in the dead polymer chains
    Dsoma = int(0)  # (np.uint32) meli
    for i in range(1, len(D),1): Dsoma += (i*(D[i])) # Dsoma != D.sum()
    SomaD1Unsat = int(0)
    SomaD2Unsat = int(0)

    reacao = "nenhuma"
    explicacao = "nenhuma"


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
    SumBonds[:] = sumCCbonds
    DeadPolymer[0] = D.sum()





    j, l, i = int(0), int(0), int(0)
    erro = (0)

    ll = int(1)      # A variable to register the information according to a defined time interval
    SomaD = (0) # A variable that is used in the routine to found the size of the polymer with that specific C-C bond
    maior_que_DP = 0    # A variable that register the number of dead polymer chains that eventually has a more meres than DP
    tfim = 0

    R[23] = kf*Dsoma
    print("R initial: R[23] = ", R[23])

    dog1, dog2, dog3, dog4, dog5, dog6, dog7, dog8, dog9, dog10 = 0, 0,0,0,0,0,0,0,0,0
    dog11, dog12, dog13, dog14, dog15, dog16, dog17, dog18, dog19, dog20 = 0, 0,0,0,0,0,0,0,0,0
    dog21, dog22, dog23, dog24, dog25, dog26, dog27, dog28, dog29, dog30 = 0, 0,0,0,0,0,0,0,0,0
    dog31, dog32, dog33, dog34, dog35, dog36, dog37, dog38, dog39, dog40 = 0, 0,0,0,0,0,0,0,0,0
    dog41, dog42, dog43, dog44, dog45, dog46, dog47, dog48 = 0, 0,0,0,0,0,0,0

    X3length = int(0)
    X3length = int(0)


    print("kdmMC", kdmMC)
    print("SumBonds[0]", SumBonds[0])

    while t < tend:
        R0 = np.sum(R) # sum of all Reaction rates
        if R0 == 0:
            print("Error. R0 == 0")
            for i in range(len(X)):
                print(i, X[i])
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
                        # if D[int(j)] <= 0: print("R15: D[int(j)] - 1 < 0. i:", i, "j:", j, "X[0]", X[0], ". D[:5]", D[int(j)])
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

        # EvaporationFactor = 0.8
        # # print(X[12], X[26], X[12]*EvaporationFactor)
        # if X[19] > 0:
        #     X[24] += int(X[19]*EvaporationFactor)    # Toluene evaporated
        #     X[19] = X[19]*(1-EvaporationFactor)
        # if X[18] > 0:
        #     X[24] += int(X[18]*EvaporationFactor)    # Ethylbenzene evaporated
        #     X[18] = X[18]*(1-EvaporationFactor)
        # if X[12] > 0:
        #     X[26] += int(X[12]*EvaporationFactor)    # Styrene evaporated
        #     X[12] = X[12]*(1-EvaporationFactor)
        # if X[13] > 0:
        #     X[27] += int(X[13]*EvaporationFactor)    # aMS evaporated
        #     X[13] = X[13]*(1-EvaporationFactor)
        # if X[14] > 0:
        #     X[28] += int(X[14]*EvaporationFactor)    # Dimer evaporated
        #     X[14] = X[14]*(1-EvaporationFactor)
        # if X[15] > 0:
        #     X[29] += int(X[15]*EvaporationFactor)    # DPP evaporated
        #     X[15] = X[15]*(1-EvaporationFactor)
        # if X[16] > 0:
        #     X[30] += int(X[16]*EvaporationFactor)    # Trimer evaporated
        #     X[16] = X[16]*(1-EvaporationFactor)


        EvaporationFactor = 1.00


        if X.any() < 0:
            print("X < 0:")
            for i in range(len(X)):
                print("i:", X[i])

        value2 = int(np.sum(X[:24])*EvaporationFactor)
        somatemporaria = np.sum(X[:24])
        if X[19] > 0:
            # value = int(X[19]*EvaporationFactor)
            value = int(X[19]*value2/somatemporaria)
            X[24] += value    # Toluene evaporated
            X[19] -= value    # Tolune liquid
        if X[18] > 0:
            # value = int(X[18]*EvaporationFactor)
            value = int(X[18]*value2/somatemporaria)
            X[25] += value   # Ethylbenzene evaporated
            X[18] -= value
        if X[12] > 0:
            # value = int(X[12]*EvaporationFactor)
            value = int(X[12]*value2/somatemporaria)
            # print(X[12]/np.sum(X[:24]), value2, value, int(X[12]*value2/np.sum(X[:24])))
            X[26] += value    # Styrene evaporated
            X[12] -= value
            if X[12] < 0:
                print("value:", value, X[12], X[26])
                break
        if X[13] > 0:
            # value = int(X[13]*EvaporationFactor)
            value = int(X[13]*value2/somatemporaria)
            X[27] += value    # aMS evaporated
            X[13] -= value
        if X[14] > 0:
            # value = int(X[14]*EvaporationFactor)
            value = int(X[14]*value2/somatemporaria)
            X[28] += value    # Dimer evaporated
            X[14] -= value
        if X[15] > 0:
            # value = int(X[15]*EvaporationFactor)
            value = int(X[15]*value2/somatemporaria)
            X[29] += value    # DPP evaporated
            X[15] -= value
        if X[16] > 0:
            # value = int(X[16]*EvaporationFactor)
            value = int(X[16]*value2/somatemporaria)
            X[30] += value    # Trimer evaporated
            X[16] -= value




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
        # t_temp_thermalinit += t

        factor = 1
        R[48] =  kdmMC * (X[12]*factor) * ((X[12]*factor) - 1) * ((X[12]*factor) - 2)/6.0  # Styrene thermal initiation
        R[49] =  ktc*X[17]*(X[17]-1)/2                          # Termination by combination (ethylbenzene radicals)
        R[50] =  ktc*X[23]*(X[23]-1)/2                          # Termination by combination (dimer radicals)
        R[51] =  ktc*X[23]*X[17]                                # Termination by combination (styrene + dimer radicals)
        R[52] =  0.0 # There are already reactions for ethylbenzene radical # kH*X[22]*(Dsoma-2*X[0])       # HYDROGEN ABSTRACTION by styrene radical attack (forming styrene) (L/mol*seg) X[0]
        R[53] =  kH*X[23]*(Dsoma-2*X[0])       # HYDROGEN ABSTRACTION by dimer radical attack (forming dimer) (L/mol*seg) X[0]
        # -------------------------------------------------------------------------
        t = t + dt
        j, l, i = int(0), int(0), int(0)

        if t >= time_interval[ll]:
            print("time:", np.round(t,3), np.round((SumBonds[0]-sumCCbonds)*100/SumBonds[0], 2), ".X[12] Sty", X[12], "X[26]", X[26], R[48], ". sumCCbonds:", sumCCbonds)
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

            ll += (1)
        elif sumCCbonds == 0:   # (SumBonds[0]-SumBonds[ll])*100/SumBonds[0] > 99.99999:
            print("Ending - Conversion 100%")
            print("SumBonds[0]-SumBonds[ll]*100/SumBonds[0] > 99.0", (SumBonds[0]-SumBonds[ll])*100/SumBonds[0])
            print("SumBonds[0]", SumBonds[0], "SumBonds[ll]", SumBonds[ll])
            print("time:", t)
            print("")
            print("X:", X)
            print("  ")
            break

        # print(t2, dt, t)
        tfim = t

    MWSty = 104.15
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

    return X,P,D,D1Unsat,D2Unsat,Styrene,Dimer,Trimer,DeadPolymer,Rs,Rp,Rm,Mn,Mw,tfim,erro,SumBonds
