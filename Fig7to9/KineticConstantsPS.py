import numpy as np
from numba import jit


@jit(nopython=True)  # @jit(nopython=True, cache=True) 
def KineticConstants(RT):
    # kdm    = 1.314e7 * np.exp(-27.440/(Rcte * T))          # styrene thermal initiation (L2/(mol2 min))
    # kdm    = 7.884e8/60  * np.exp(-27.440/(Rcte * T))
    kdm      = 2.19*10**5*np.exp(-27.440/(RT))
    
    print("kdm:", kdm)
    k11    = 6.36e8 * np.exp(-7067.0/(RT))            # propagation constant (L/(mol min))

    
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
    return [kf0, kfs0, kH0, kH0_Rp, kdp0, kdp_Rp0, kbs0, kbs0_LMWS, kc0, ktd0, k130,k150, k170, k730, kra0, kbra0, kdppa0, kdm, k11]
