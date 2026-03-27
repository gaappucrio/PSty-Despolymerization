import numpy as np
import matplotlib.pyplot as plt
import random
import time
from numba import jit

from  InitialDistribution import Moleculas
# from  McPackage import MC
from  kMC_v2024 import MC

Na   = 6.022e23              # 1/mol --> Avogadro number
T    = 500.0 + 273.15         # K         
Rcte = 1.987/1000          # kcal/mol K
RT   = Rcte*T
MW   = 104.15
V    = 1.0e-16 # 1.0e-16

t = 0.0
if T ==  (500.0 + 273.15):
    tend = 10 # 30  # seconds
    step = 1
elif T ==  (600.0 + 273.15):
    tend = 1.0  # seconds   
    step = 0.001
else: # T ==  (700.0 + 273.15):
    tend = 0.1  # seconds 
    step = 0.0001    
    
    
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
products = np.sum(X[12:17]) + np.sum(X[18:20]) + X[21] + np.sum(X[24:])
print("Monomer:", np.round(X[12]*100/products,2), X[26]*100/products)
print("alpha-Methylstyrene:", np.round(X[13]*100/products,2), X[27]*100/products)
print("DPP:", np.round(X[15]*100/products,2), X[2]*100/products)
print("Dimer:", np.round(X[14]*100/products,2), X[29]*100/products)
print("Trimer:", np.round(X[16]*100/products,2), X[30]*100/products)
print("Toluene:", np.round(X[19]*100/products, 2), X[24]*100/products)
print("Ethylbenzene:", X[18]*100/products, X[25]*100/products)
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


print("Area PS original:",  np.trapz(wx_PS   , x=xPlotar))
print("Area PS degradado:", np.trapz(wx_Dsoma, x=x_Dsoma_Ds))

fig, ax = plt.subplots(figsize=(6,5))
ax.plot(xPlotar,    wx_PS,      color = "grey", label="Original", linewidth=4.0) # POLYMER NOT DEGRADADED
ax.plot(x_Dsoma_Ds, wx_Dsoma,   color= "black", label="Degradated", ls='--', linewidth=3.0)

plt.title(f'Time = {np.round(tfim,4)} s', fontsize=18)
plt.ylabel('$w_n$' + '(%)',fontsize = 22)   # plt.ylabel('$w_{log(n)}$',fontsize = axf)
plt.xlabel('Chain length',fontsize = 20)  # plt.xlabel('$log_{10}(n)$',fontsize = axf)
plt.tick_params(labelsize=18)
plt.legend(frameon=False, fontsize=16)
ax.yaxis.get_offset_text().set_fontsize(16)
ax.set_xlim([0, 5000])
fig.tight_layout()
plt.savefig("MMDegradation_SUMDs_2" + str(int(T-273.15)) + ".png",format='png',dpi = 500,bbox_inches='tight')

# np.savetxt('MWD_500_171s.txt', wx_Dsoma)
# np.savetxt('MWD_500_171s_x.txt', x_Dsoma_Ds)
np.savetxt("500_wx_Dsoma_"+ str(V) + ".txt", wx_Dsoma)

# print(Mn_MC)
# print(Mw_MC)
# Mn_MC[-1] = Mn_MC[-2]
# Mw_MC[-1] = Mw_MC[-2]
Mn_MC[np.argmax(Styrene):] = Mn_MC[np.argmax(Styrene)]
Mw_MC[np.argmax(Styrene):] = Mw_MC[np.argmax(Styrene)]


Mn_MC[0] = Mn0
Mw_MC[0] = Mw0

np.savetxt("time_interval500.txt", time_interval)
np.savetxt("Mw_MC500.txt", Mw_MC)
np.savetxt("Mn_MC500.txt", Mn_MC)

fig, ax = plt.subplots(figsize=(6,4))
ax.plot(time_interval, Mw_MC, lw='3.0', color='deeppink', label='$M_w$')
ax.plot(time_interval, Mn_MC, lw='3.0', label='$M_n$')

# ax2 = ax.twinx()  
# ax2.plot(time_interval, Mn_MC, lw='3.0', label='$M_n$')
# ax2.set_ylabel('$M_n$', color='blue', fontsize = 18) 

ax.set_xlabel('Time (s)', fontsize = 20)
ax.set_ylabel('Molecular weight (g/mol)', fontsize = 20)

# ax.set_ylabel('Styrene (molecules)', color=color, fontsize = 16)
ax.set_xlim([0, tfim])
ax.tick_params(axis='both', which='major', labelsize=20)

plt.legend(frameon=False, fontsize = 20)

plt.savefig("MwMn" + str(int(T-273.15)) + ".png",format='png',dpi = 1000,bbox_inches='tight')


DeadPolymer[np.argmax(Styrene):] = DeadPolymer[np.argmax(Styrene)]
Rs[np.argmax(Styrene):] = Rs[np.argmax(Styrene)]
Rp[np.argmax(Styrene):] = Rp[np.argmax(Styrene)]
Styrene[Styrene == 0] = np.max(Styrene)
Dimer[Dimer == 0] = np.max(Dimer)
Trimer[Trimer == 0] = np.max(Trimer)
Styrene[0] = 0
Dimer[0] = 0
Trimer[0] = 0



fig, ax1 = plt.subplots(figsize=(6,4))
color = 'tab:red'
ax1.set_xlabel('Time (s)', fontsize = 16)
ax1.set_ylabel('Styrene (molecules)', color=color, fontsize = 18)
ax1.plot(time_interval, Styrene, color=color, lw=3.0)
ax1.tick_params(axis='y', labelcolor=color)
ax1.tick_params(axis='both', which='major', labelsize=16)
ax1.set_xlim([0, tfim])
ax1.set_ylim(bottom=0)
ax1.yaxis.get_offset_text().set_fontsize(16)

# plt.plot(time_interval, Styrene, color = "red", label="Styrene")
ax2 = ax1.twinx()  
color = 'tab:blue'
# ax2.set_ylabel('Polymer (molecules)', color=color) 
ax2.set_ylabel('Polymer (molecules)', color=color, fontsize = 18)  # plt.ylabel('$w_{(n)}') 
ax2.plot(time_interval, DeadPolymer, color=color, lw=3.0)
# ax2.get_yaxis().get_major_formatter().set_scientific(True)
ax2.set_ylim(bottom=0)
ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
ax2.tick_params(axis='y', labelcolor=color)
ax2.tick_params(axis='both', which='major', labelsize=16)
ax2.yaxis.get_offset_text().set_fontsize(16)

fig.tight_layout()
# fig.legend()
plt.savefig("Styrene_VS_DeadPolymer" + str(int(T-273.15)) + ".png",format='png',dpi = 500,bbox_inches='tight')


from scipy.interpolate import UnivariateSpline
spl_Rs = UnivariateSpline(time_interval, Rs)
spl_Rp = UnivariateSpline(time_interval, Rp)
spl_Rm = UnivariateSpline(time_interval, Rm)
time_interval_Rs = np.arange(t, tend+1, 1)

fig, ax1 = plt.subplots(figsize=(8,5))
color = 'tab:red'
ax1.set_xlabel('Time (s)', fontsize = 22)
ax1.set_ylabel('Polymer radicals (mol)', color=color, fontsize = 22)
ax1.plot(time_interval, Rs/Na, color=color, label='Rs', lw=3)
ax1.plot(time_interval, Rp/Na, color='deeppink', label='Rp', linestyle='dotted', lw=3)
ax1.plot(time_interval, Rm/Na, color='purple', label='Rm', linestyle='dashed', lw=3) # color=color
# *100/sumaaaar


ax1.tick_params(axis='y', labelcolor=color)
ax1.tick_params(axis='both', which='major', labelsize=22)
ax1.set_xlim([0, tfim])
ax1.tick_params(axis='x', labelsize=22)
# ax1.set_ylim(0, 10)
ax1.yaxis.get_offset_text().set_fontsize(22)

# ax1.set_ylim(bottom=0)
ax2 = ax1.twinx()  
color = 'tab:blue'
ax2.set_ylabel('Products (mol)', color=color, fontsize = 22)
ax2.plot(time_interval, Styrene/Na, color=color, label='Styrene', lw=3)
ax2.plot(time_interval, Dimer/Na, color=color, linestyle='dotted', label='Dimer', lw=5)
ax2.plot(time_interval, Trimer/Na, color=color, linestyle='dashed', label='Trimer', lw=3)

ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
ax2.tick_params(axis='y', labelcolor=color)
ax2.tick_params(axis='both', which='major', labelsize=22)
ax2.yaxis.get_offset_text().set_fontsize(22)
# ax2.set_ylim(0, 1e7)

# fig.legend(frameon=False, fontsize = 16)
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
lines = lines_1 + lines_2
labels = labels_1 + labels_2



# plt.legend(lines, labels,frameon=False, loc=0, fontsize = 16)
plt.legend(lines, labels, bbox_to_anchor=(1.15, 0.8), loc='upper left', frameon=False, fontsize = 22)


# fig.tight_layout()
plt.savefig("RadicalsVSProducts" + str(int(T-273.15)) + ".png",format='png',dpi = 1000,bbox_inches='tight')





C_Distribution     = np.zeros(6)
C_Distribution[0]  += X[19] + X[24]  # X[19] + X[10]   # Toluene = C7H8
C_Distribution[1]  += X[26]          # X[12] + D2[1]   # Monomer = Styrene = C8H8  
# C_Distribution[1]  += X[25] # X[18]        # + Ethylbenzene = C8H10
C_Distribution[2]  += X[27]  + X[13] # X[9]  + X[13]    # Alpha-methylstyrene
C_Distribution[3]  += X[28]          # X[14] + D2[2]   # Dimer   = C16H16
C_Distribution[4]  += X[29]          # X[15] # + X[11]   # DPP     = C16H16
C_Distribution[5]  += X[30]          # X[16] + D2[3]   # Trimer  = C24H26 

print("C_Distribution", C_Distribution)




axis_x_1 = [7, 8, 9, 15, 16, 24]
labels = ['Toluene', 'Styrene', 'aMS', 'DPP', 'Dimer', 'Trimer']


print("ethylbenzene:", (X[18] + X[17])*100/np.sum(X[3:]))           # Ethylbenzene = C8H10

C_Distribution_Percentage = np.zeros(6)
C_Distribution_Percentage[0] = C_Distribution[0]*100/np.sum(X[24:]) # Toluene = C7H8 C_Distribution[0]*100/np.sum(X[3:]) 
C_Distribution_Percentage[1] = C_Distribution[1]*100/np.sum(X[24:]) # Monomer
C_Distribution_Percentage[2] = C_Distribution[2]*100/np.sum(X[24:]) # aMS
C_Distribution_Percentage[3] = C_Distribution[3]*100/np.sum(X[24:]) # Dimer
C_Distribution_Percentage[4] = C_Distribution[4]*100/np.sum(X[24:]) # DPP
C_Distribution_Percentage[5] = C_Distribution[5]*100/np.sum(X[24:]) # Trimer


C_Distribution2 = np.zeros(6)
C_Distribution2[0]  += X[19] + X[10]   # Toluene = C7H8
C_Distribution2[1]  += X[12] + D2[1]   # Monomer = Styrene = C8H8  
# C_Distribution2[1]  += X[18]           # + Ethylbenzene = C8H10
C_Distribution2[2]  += X[9]  + X[13]    # Alpha-methylstyrene
C_Distribution2[3]  += X[14] + D2[2]   # Dimer   = C16H16
C_Distribution2[4]  += X[15] # + X[11]   # DPP     = C16H16
C_Distribution2[5]  += X[16] + D2[3]   # Trimer  = C24H26 
C_Distribution_Percentage2    = np.zeros(6)
C_Distribution_Percentage2[0] = C_Distribution2[0]*100/np.sum(X[3:]) # Toluene = C7H8
C_Distribution_Percentage2[1] = C_Distribution2[1]*100/np.sum(X[3:]) # Monomer
C_Distribution_Percentage2[2] = C_Distribution2[2]*100/np.sum(X[3:]) # aMS
C_Distribution_Percentage2[3] = C_Distribution2[3]*100/np.sum(X[3:]) # Dimer
C_Distribution_Percentage2[4] = C_Distribution2[4]*100/np.sum(X[3:]) # DPP
C_Distribution_Percentage2[5] = C_Distribution2[5]*100/np.sum(X[3:]) # Trimer

print("C_Distribution2", C_Distribution2)


C_Distribution_experimental = np.zeros(6)
C_Distribution_experimental_error = np.zeros(6)

if T == (500.0 + 273.15):
    C_Distribution_experimental[1] = 75.7 # Monomer
    C_Distribution_experimental[4] = 6.1 # Dimer
    C_Distribution_experimental[5] = 12.2  # Trimer
    C_Distribution_experimental[0] = 0.8 # Toluene = C7H8
    C_Distribution_experimental[2] = 0.2 # aMS
    C_Distribution_experimental[3] += 7.4 # DPP

    C_Distribution_experimental_error[1] = 6.1 # Monomer
    C_Distribution_experimental_error[4] = 1.2 # Dimer
    C_Distribution_experimental_error[5] = 4.0  # Trimer
    C_Distribution_experimental_error[0] = 0.8*0.05 # Toluene = C7H8
    C_Distribution_experimental_error[2] = 0.2*0.05 # aMS
    C_Distribution_experimental_error[3] += 6.1 # DPP 
    
elif T == (600.0 + 273.15):
    C_Distribution_experimental[1] = 76.4 # Monomer
    C_Distribution_experimental[4] = 4.2 # Dimer
    C_Distribution_experimental[5] = 8.5  # Trimer
    C_Distribution_experimental[0] = 0.75 # Toluene = C7H8
    C_Distribution_experimental[2] = 0.25 # aMS
    C_Distribution_experimental[3] += 0.55 # DPP

    C_Distribution_experimental_error[1] = 3.7 # Monomer
    C_Distribution_experimental_error[4] = 1.5 # Dimer
    C_Distribution_experimental_error[5] = 3.0  # Trimer
    C_Distribution_experimental_error[0] = 0.75*0.05 # Toluene = C7H8
    C_Distribution_experimental_error[2] = 0.25*0.05 # aMS
    C_Distribution_experimental_error[3] += 0.1 # DPP

else:
    C_Distribution_experimental[1] = 77.2 # Monomer
    C_Distribution_experimental[4] = 3.4  # Dimer
    C_Distribution_experimental[5] = 5.2  # Trimer
    C_Distribution_experimental[0] = 0.7  # Toluene = C7H8
    C_Distribution_experimental[2] = 0.4  # aMS
    C_Distribution_experimental[3] += 1.4 # DPP

    C_Distribution_experimental_error[1] = 1.1      # Monomer
    C_Distribution_experimental_error[4] = 3.4*0.05 # Dimer
    C_Distribution_experimental_error[5] = 5.2*0.5 # 0.3      # Trimer
    C_Distribution_experimental_error[0] = 0.7*0.05 # Toluene = C7H8
    C_Distribution_experimental_error[2] = 0.1      # aMS
    C_Distribution_experimental_error[3] = 0.3     # DPP


C_Distribution_experimental_error[0] /= C_Distribution_experimental[0]
C_Distribution_experimental_error[1] /= C_Distribution_experimental[1]
C_Distribution_experimental_error[2] /= C_Distribution_experimental[2]
C_Distribution_experimental_error[3] /= C_Distribution_experimental[3]
C_Distribution_experimental_error[4] /= C_Distribution_experimental[4]
C_Distribution_experimental_error[5] /= C_Distribution_experimental[5]  




C_Distribution_Percentage[0] *= 92.14
C_Distribution_Percentage[1] *= 104.15
C_Distribution_Percentage[2] *= 118.18
C_Distribution_Percentage[3] *= 210.32
C_Distribution_Percentage[4] *= 210.32
C_Distribution_Percentage[5] *= 312.46

C_Distribution_Percentage2[0] *= 92.14
C_Distribution_Percentage2[1] *= 104.15
C_Distribution_Percentage2[2] *= 118.18
C_Distribution_Percentage2[3] *= 210.32
C_Distribution_Percentage2[4] *= 210.32
C_Distribution_Percentage2[5] *= 312.46

C_Distribution_experimental[0] *= 92.14
C_Distribution_experimental[1] *= 104.15
C_Distribution_experimental[2] *= 118.18
C_Distribution_experimental[3] *= 210.32
C_Distribution_experimental[4] *= 210.32
C_Distribution_experimental[5] *= 312.46

C_Distribution_Percentage = C_Distribution_Percentage*100/np.sum(C_Distribution_Percentage)
C_Distribution_Percentage = np.round(C_Distribution_Percentage, 2)

C_Distribution_Percentage2 = C_Distribution_Percentage2*100/np.sum(C_Distribution_Percentage2)
C_Distribution_Percentage2 = np.round(C_Distribution_Percentage2, 2)

C_Distribution_experimental = C_Distribution_experimental*100/np.sum(C_Distribution_experimental)
C_Distribution_experimental  = np.round(C_Distribution_experimental, 2)


C_Distribution_experimental_error[0] *= C_Distribution_experimental[0]
C_Distribution_experimental_error[1] *= C_Distribution_experimental[1]
C_Distribution_experimental_error[2] *= C_Distribution_experimental[2]
C_Distribution_experimental_error[3] *= C_Distribution_experimental[3]
C_Distribution_experimental_error[4] *= C_Distribution_experimental[4]
C_Distribution_experimental_error[5] *= C_Distribution_experimental[5]  



axis_x_1 = [7, 8.5, 10, 15, 16, 24]    # [7, 8, 9, 16, 24]

fig, ax = plt.subplots(figsize=(10,5))
width = 0.5
axis_x_2 = [p + width for p in axis_x_1]
axis_x_3 = [p + 0.5*width for p in axis_x_1]

plt.bar(axis_x_1, C_Distribution_Percentage, width, alpha=0.5, color='#EE3224', label='Model')
plt.bar(axis_x_2, C_Distribution_experimental, width, alpha=0.5, yerr=C_Distribution_experimental_error, error_kw=dict(ecolor='black', lw=2, capsize=5, capthick=2), color='#F78F1E', label='Experimental')
## plt.bar(axis_x_2[-2], 7.4, width, alpha=0.5, yerr=6.1, color='#FFC222', label='DPP') # experimental

# plt.bar(axis_x_2[-3], 6.1, width, alpha=1, yerr=1.2, error_kw=dict(ecolor='grey', lw=2, capsize=5, capthick=2), color='crimson', label='Dimer Exp.')   # experimental 
# plt.bar(axis_x_1[-3], (X[14] + D2[2])*100/np.sum(X[3:]), width, alpha=1, color='crimson', label='Dimer Model')   # experimental 


ax.set_ylabel('wt %', fontsize=16)
ax.set_xticks(axis_x_3)

ax.set_xticklabels(labels)
ax.annotate('{}'.format(C_Distribution_experimental[0]), xy=(axis_x_2[0], np.maximum(C_Distribution_experimental[0], C_Distribution_Percentage[0])), xytext=(0, 2), textcoords="offset points",  ha='center', va='bottom', fontsize=14)
ax.annotate('{}'.format(np.round(C_Distribution_Percentage[0],1)), xy=(axis_x_1[0], np.maximum(C_Distribution_experimental[0], C_Distribution_Percentage[0])), xytext=(0, 14), textcoords="offset points",  ha='center', va='bottom', fontsize=14)

ax.annotate('{}'.format(C_Distribution_experimental[2]), xy=(axis_x_2[2], np.maximum(C_Distribution_experimental[2], C_Distribution_Percentage[2])), xytext=(0, 8), textcoords="offset points",  ha='center', va='bottom', fontsize=14)
ax.annotate('{}'.format(np.round(C_Distribution_Percentage[2],1)), xy=(axis_x_1[2], np.maximum(C_Distribution_experimental[2], C_Distribution_Percentage[2])), xytext=(0, 20), textcoords="offset points",  ha='center', va='bottom', fontsize=14)

ax.annotate('{}'.format(C_Distribution_experimental[1]), xy=(axis_x_2[1], np.maximum(C_Distribution_experimental[1], C_Distribution_Percentage[1])), xytext=(0, 12), textcoords="offset points",  ha='center', va='bottom', fontsize=14)
ax.annotate('{}'.format(np.round(C_Distribution_Percentage[1],1)), xy=(axis_x_1[1], np.maximum(C_Distribution_experimental[1], C_Distribution_Percentage[1])), xytext=(0, 0), textcoords="offset points",  ha='center', va='bottom', fontsize=14)

ax.annotate('{}'.format(C_Distribution_experimental[3]), xy=(axis_x_2[3], np.maximum(C_Distribution_experimental[3], C_Distribution_Percentage[3])), xytext=(0, 25), textcoords="offset points",  ha='center', va='bottom', fontsize=14)
ax.annotate('{}'.format(np.round(C_Distribution_Percentage[3],1)), xy=(axis_x_1[3], np.maximum(C_Distribution_experimental[3], C_Distribution_Percentage[3])), xytext=(0, 10), textcoords="offset points",  ha='center', va='bottom', fontsize=14)
# ax.annotate('{}'.format("6.1 Dimer + 7.4 DPP"), xy=(axis_x_2[3], np.maximum(C_Distribution_experimental_500[3], C_Distribution_Percentage[3])), xytext=(0, 24), textcoords="offset points",  ha='center', va='bottom', fontsize=14)
# ax.annotate('{}'.format("3.8 Dimer + 3.1 DPP"), xy=(axis_x_1[3], np.maximum(C_Distribution_experimental_500[3], C_Distribution_Percentage[3])), xytext=(0, 2), textcoords="offset points",  ha='center', va='bottom', fontsize=14)

ax.annotate('{}'.format(C_Distribution_experimental[4]), xy=(axis_x_2[4], np.maximum(C_Distribution_experimental[4], C_Distribution_Percentage[4])), xytext=(0, 12), textcoords="offset points",  ha='center', va='bottom', fontsize=14)
ax.annotate('{}'.format(np.round(C_Distribution_Percentage[4],1)), xy=(axis_x_1[4], np.maximum(C_Distribution_experimental[4], C_Distribution_Percentage[4])), xytext=(0, 0), textcoords="offset points",  ha='center', va='bottom', fontsize=14)


ax.annotate('{}'.format(C_Distribution_experimental[5]), xy=(axis_x_2[5], np.maximum(C_Distribution_experimental[5], C_Distribution_Percentage[5])), xytext=(0, 30), textcoords="offset points",  ha='center', va='bottom', fontsize=14)
ax.annotate('{}'.format(np.round(C_Distribution_Percentage[5],1)), xy=(axis_x_1[5], np.maximum(C_Distribution_experimental[5], C_Distribution_Percentage[5])), xytext=(0, 5), textcoords="offset points",  ha='center', va='bottom', fontsize=14)


plt.xticks(fontsize=14, rotation=45)
plt.yticks(fontsize=14)

                
plt.ylim(0, 100)
plt.legend(frameon=False, fontsize=16)
plt.tight_layout()
plt.savefig("CDistributionExp" + str(int(T-273.15)) + ".png",format='png',dpi = 800,bbox_inches='tight')






fig, ax = plt.subplots(figsize=(10,5))
width = 0.5
axis_x_2 = [p + width for p in axis_x_1]
axis_x_3 = [p + 0.5*width for p in axis_x_1]
plt.bar(axis_x_1, C_Distribution_Percentage2, width, alpha=0.5, color='#EE3224', label='Model')
plt.bar(axis_x_2, C_Distribution_experimental, width, alpha=0.5, yerr=C_Distribution_experimental_error, error_kw=dict(ecolor='black', lw=2, capsize=5, capthick=2), color='#F78F1E', label='Experimental')
ax.set_ylabel('wt %', fontsize=16)
ax.set_xticks(axis_x_3)
ax.set_xticklabels(labels)
ax.annotate('{}'.format(C_Distribution_experimental[0]), xy=(axis_x_2[0], np.maximum(C_Distribution_experimental[0], C_Distribution_Percentage[0])), xytext=(0, 2), textcoords="offset points",  ha='center', va='bottom', fontsize=14)
ax.annotate('{}'.format(np.round(C_Distribution_Percentage2[0],1)), xy=(axis_x_1[0], np.maximum(C_Distribution_experimental[0], C_Distribution_Percentage[0])), xytext=(0, 14), textcoords="offset points",  ha='center', va='bottom', fontsize=14)
ax.annotate('{}'.format(C_Distribution_experimental[2]), xy=(axis_x_2[2], np.maximum(C_Distribution_experimental[2], C_Distribution_Percentage[2])), xytext=(0, 8), textcoords="offset points",  ha='center', va='bottom', fontsize=14)
ax.annotate('{}'.format(np.round(C_Distribution_Percentage2[2],1)), xy=(axis_x_1[2], np.maximum(C_Distribution_experimental[2], C_Distribution_Percentage[2])), xytext=(0, 20), textcoords="offset points",  ha='center', va='bottom', fontsize=14)
ax.annotate('{}'.format(C_Distribution_experimental[1]), xy=(axis_x_2[1], np.maximum(C_Distribution_experimental[1], C_Distribution_Percentage[1])), xytext=(0, 12), textcoords="offset points",  ha='center', va='bottom', fontsize=14)
ax.annotate('{}'.format(np.round(C_Distribution_Percentage2[1],1)), xy=(axis_x_1[1], np.maximum(C_Distribution_experimental[1], C_Distribution_Percentage[1])), xytext=(0, 0), textcoords="offset points",  ha='center', va='bottom', fontsize=14)
ax.annotate('{}'.format(C_Distribution_experimental[3]), xy=(axis_x_2[3], np.maximum(C_Distribution_experimental[3], C_Distribution_Percentage[3])), xytext=(0, 25), textcoords="offset points",  ha='center', va='bottom', fontsize=14)
ax.annotate('{}'.format(np.round(C_Distribution_Percentage2[3],1)), xy=(axis_x_1[3], np.maximum(C_Distribution_experimental[3], C_Distribution_Percentage[3])), xytext=(0, 10), textcoords="offset points",  ha='center', va='bottom', fontsize=14)
ax.annotate('{}'.format(C_Distribution_experimental[4]), xy=(axis_x_2[4], np.maximum(C_Distribution_experimental[4], C_Distribution_Percentage[4])), xytext=(0, 12), textcoords="offset points",  ha='center', va='bottom', fontsize=14)
ax.annotate('{}'.format(np.round(C_Distribution_Percentage2[4],1)), xy=(axis_x_1[4], np.maximum(C_Distribution_experimental[4], C_Distribution_Percentage[4])), xytext=(0, 0), textcoords="offset points",  ha='center', va='bottom', fontsize=14)
ax.annotate('{}'.format(C_Distribution_experimental[5]), xy=(axis_x_2[5], np.maximum(C_Distribution_experimental[5], C_Distribution_Percentage[5])), xytext=(0, 30), textcoords="offset points",  ha='center', va='bottom', fontsize=14)
ax.annotate('{}'.format(np.round(C_Distribution_Percentage2[5],1)), xy=(axis_x_1[5], np.maximum(C_Distribution_experimental[5], C_Distribution_Percentage[5])), xytext=(0, 5), textcoords="offset points",  ha='center', va='bottom', fontsize=14)
plt.xticks(fontsize=14, rotation=45)
plt.yticks(fontsize=14) 
plt.ylim(0, 100)
plt.legend(frameon=False, fontsize=16)
plt.tight_layout()
plt.show()