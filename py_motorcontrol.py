import numpy as np
import matplotlib.pyplot as plt

vdc=300
Pb = 3000        
Ibr = 23              
Vbr = Pb/(3*Ibr)      
Poles = 8             
Speed_rated = 1500    # in rpm
w_rated = Speed_rated*2*np.pi/60   # in Hz

Tb = Pb/w_rated       
wb = w_rated*Poles/2  
Vb = 1.414*Vbr        
Ib = 1.414*Ibr       
zb = Vbr/Ibr          
lamb = Vb/wb          
Lb = lamb/Ib          

Lq = 130/100000
Ld = 130/100000
ro = Lq/Ld
lamaf = 98/1000
Rs = 570/1000
B = 0.0005
J = 0.0002
tln = 0.5

f_cmd = 100           
wr_ref = 2*np.pi*1000/60

fc = 8000
Kpi = 10

Kp = 1
Ki = 10

Kp_id = 2
Ki_id = 100
Kp_iq = 2
Ki_iq = 100

theta_r = 0 
wr = 0 
t = 0 
dt = 1e-7 
tfinal = 0.05 
if_ref = -1e-16 
iqs = 0;   ids = 0
idsf = 0;  iqsf = 0
vqs = 0;  vds = 0
Tl= 0.5*Tb 
n = 1 
x = 1 
signe = 1 
carrier = -1
ias=0;   ibs=0;   ics=0;   t1=0
vax1=0;   vbx1=0;   vcx1=0
vao1=0
zia=0;  zib=0;   zic=0
y = 0 
w = 0 
z = 0 
vax_o = 0
time_idx = []
wr_output = []
n1 = 0
while (t<tfinal):
    wr_err = wr_ref - wr
    y = y + wr_err*dt   
    Te_ref = Kp*wr_err + Ki*y   
    iq_ref = Te_ref    
    id_ref = 0          

    if iq_ref > 2*Ib:
        iq_ref = 2*Ib
    if iq_ref < -2*Ib:
        iq_ref = -2*Ib

    is_ref = np.sqrt(id_ref**2 + iq_ref**2)

    iq_err = iq_ref - iqsf 
    w = w + iq_err*dt 
    Vq_ref = Kp_iq*iq_err + Ki_iq*w
    id_err = id_ref - idsf 
    z = z + id_err*dt 
    Vd_ref = Kp_id*id_err + Ki_id*z

    theta_v = np.arctan2(Vq_ref, Vd_ref)
    Vs_ref = np.sqrt(Vd_ref**2 + Vq_ref**2)

    Vas_ref = np.cos(theta_r) * Vd_ref - np.sin(theta_r) * Vq_ref
    Vbs_ref = np.cos(theta_r - 2*np.pi/3) * Vd_ref - np.sin(theta_r - 2*np.pi/3) * Vq_ref
    Vcs_ref = np.cos(theta_r + 2*np.pi/3) * Vd_ref - np.sin(theta_r + 2*np.pi/3) * Vq_ref

    if Vas_ref > 150:
        Vas_ref = 150
    if Vbs_ref > 150:
        Vbs_ref = 150
    if Vcs_ref > 150:
        Vcs_ref = 150

    Vx_max = max([Vas_ref,Vbs_ref,Vcs_ref]) 
    Vx_min = min([Vas_ref,Vbs_ref,Vcs_ref]) 
    vax = Vas_ref-(Vx_max+Vx_min)/2 
    vbx = Vbs_ref-(Vx_max+Vx_min)/2 
    vcx = Vcs_ref-(Vx_max+Vx_min)/2 

    if t1 > 1/fc:
        vax1 = vax 
        vbx1 = vbx 
        vcx1 = vcx 
        t1 = 0 

    if vax1 >= carrier:
        vao = vdc/2   
    elif vax1 < carrier:
        vao = -vdc/2 

    if vbx1 >= carrier:
        vbo = vdc/2 
    elif vbx1 < carrier:
        vbo = -vdc/2 

    if vcx1 >= carrier:
        vco = vdc/2 
    elif vcx1 < carrier:
        vco = -vdc/2 

    vab = vao - vbo 
    vbc = vbo - vco 
    vca = vco - vao 

    vas = (vab - vca)/3 
    vbs = (vbc - vab)/3 
    vcs = (vca - vbc)/3 

    vds = (2/3)*(np.cos(theta_r)*vas + np.cos(theta_r-2*np.pi/3)*vbs + np.cos(theta_r+2*np.pi/3)*vcs)
    vqs = (2/3)*(-np.sin(theta_r)*vas - np.sin(theta_r-2*np.pi/3)*vbs - np.sin(theta_r+2*np.pi/3)*vcs)

    d_iqs = (vqs - Rs*iqs - wr*Ld*ids - wr*lamaf)*dt/Lq 
    iqs = iqs + d_iqs 
    d_ids = (vds + wr*Lq*iqs - Rs*ids)*dt/Ld 
    ids = ids + d_ids 

    stator_curr = np.sqrt(iqs**2 + ids**2)
    #delta = np.arctan(iqs/ids)
    Te = (3/2)*(Poles/2)*iqs*((Ld-Lq)*ids + lamaf)
    d_wr = ((Poles/2) * (Te-Tl) - B*wr) * dt/J
    wr = wr + d_wr 
    d_theta_r = wr * dt  
    theta_r = theta_r + d_theta_r

    ias = ids*np.cos(theta_r) - iqs*np.sin(theta_r)
    ibs = ids*np.cos(theta_r-2*np.pi/3) - iqs*np.sin(theta_r-2*np.pi/3)
    ics = -(ias+ibs) 

    idsf = (2/3)*(np.cos(theta_r)*ias + np.cos(theta_r-2*np.pi/3)*ibs + np.cos(theta_r+2*np.pi/3)*ics)
    iqsf = (2/3)*(-np.sin(theta_r)*ias - np.sin(theta_r-2*np.pi/3)*ibs - np.sin(theta_r+2*np.pi/3)*ics)

    carrier =  150 * signe * (2/(1/(2*fc)))*dt + carrier 
    if carrier > 150:
        signe = -1
    if carrier < -150:
        signe = 1

    t = t + dt 
    t1 = t1 + dt
    x = x + 1
    if x%20000 == 0:
        print(wr)
        time_idx.append(t)
        wr_output.append(wr)

plt.plot(time_idx, wr_output, 'r-', label='plot: wr')
plt.ylabel('wr')
plt.xlabel('time')
plt.legend()
plt.show()