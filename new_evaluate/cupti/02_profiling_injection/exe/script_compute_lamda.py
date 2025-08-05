
import sys
import os
import math

# General definition of the failure rate: lamda_total = lamda_die + lamda_package

# typical die size: 45 mm x 70 mm

# considering a Silicon MOS: Asic Circuits from the IEC 62380-2 standard
# using Standard cell, full custom design, the parameters are:
 
# lamda_1 = 1.2 x 10(-5) = 0.000012
# lamda_2 = 10.0



# Air_flow types:
airflow_factor_K = {
    "natural_convection": 1.4,
    "Slightly_assisted_cooling": 1.2,
    "Fan_assisted_cooling": 1.0,
    "Forced_cooling": 0.7
}

# components of table 14: Thermal expansion coefficients
##
thermal_coefficient_component = {
    "plastic_package" : 21.5,
    "ceramic_package" : 6.5,
    "metallic_package" : 5    
}

thermal_coefficient_substrate = {
    "FR4-G-10" : 16,
    "polytetrafluoroethylene" : 20.0,
    "Polyimide_Aramid" : 6.5,
    "CU-Invar-CU" : 5.4
}
##


# table of climates, Table 8, page 23
##
climate_world = {
    "T_ae_night" : 5,
    "T_ae_day_light" : 15,
    "T_ae_mean" : 14,
    "delta_t_i" : 10
}

climate_france = {
    "T_ae_night" : 6,
    "T_ae_day_light" : 14,
    "T_ae_mean" : 11,
    "delta_t_i" : 8
}
##


die_length = 45
die_width = 70

# value of K:
cooling = {
    "natural_convection" : 1.4,
    "sligh_assis_cooling" : 1.2,
    "fan_assis_cooling" : 1.0,
    "forced_cooling" : 0.7
}


world = 0
france = 1



# Number of transistors per component:
Transistors_per_Component = {
    "DPU_arch1" : 35000,    # check this value
    "DPU_arch2" : 36000,    # check this value
    "DPU_arch3" : 37000,    # check this value
    "DPU_arch4" : 38000,    # check this value
    "DPU_arch5" : 39000,    # check this value
    "DPU_arch6" : 40000,    # check this value
    "ADD_1" : 8844,         #   4535, number of cells
    "MUL_1" : 20942,        #   13486,
    "MAC_1" : 29786,        #   18021,
    "ADD_2" : 8202,         #   1043,
    "MUL_2" : 22010,        #   2004,
    "MAC_2" : 37014         #   3951
}

# Number of ports per component:
Ports_per_Component = {
    "DPU_arch1" : 322,
    "DPU_arch2" : 322,
    "DPU_arch3" : 322,
    "DPU_arch4" : 380,    # check this value
    "DPU_arch5" : 390,    # check this value
    "DPU_arch6" : 400,    # check this value
    "ADD_1"     : 100,
    "MUL_1"     : 98,
    "MAC_1"     : 130,
    "ADD_2"     : 96,
    "MUL_2"     : 96,
    "MAC_2"     : 130
}





def calculate_total_lamda(temperature_standard, t1, t2):
    
    
    # second lamda parameter:
    # -----------------------------------------------------------------------------------------
    # 
    
    # transistor number:
    
    # Power dissipated by the component:
    
    
    # asuming BGA plastic package.

    # number of pins:
    S = 600         # asumming a complete GPU
    # S = 80        # asumming only the accelerator from the GPU
    
    package_thermal_resistance_junction_case = 0.4 * ( 6.6 * (1.1e6 / (S*S)) )
    
    package_thermal_resistance_junction_ambient = (0.4 + 0.6 * cooling["forced_cooling"]) * ( 6.6 * (1.1e6 / (S*S)) )
    
    delta_t_j = package_thermal_resistance_junction_ambient * 0.5
    
    
    
    
    
    #### Thermal expansion coefficients: (Table 14, page 32 in IEC 62380)
    ####
    # alpha_substrate (alpha_s)
    alpha_s = thermal_coefficient_substrate["FR4-G-10"]
    # alpha_component (alpha_c)
    alpha_c = thermal_coefficient_component["plastic_package"]
    # Influence factor related to the thermal expansion coefficients difference, between the mounting substrate and the package material
    pi_a = 0.06 * pow(abs(alpha_s - alpha_c),1.68)
    ####
    
    
    # Junction-case thermal resistance:
    r_jc = 0.4* (6.6 + (1.1e6 / S**2) )
    
    # Airflow factor (forced cooling)
    K = airflow_factor_K["Forced_cooling"]
    
    # Junction-ambient thermal resistance:
    r_ja = (0.4 + (0.6 * K) ) * (6.6 + (1.1e6 / S**2) )
    
    # From Table 17
    # Base fault rate of the integrated circuit package
    D = pow( (die_length**2 + die_width**2 ) , 0.5)
    lamda_3 = 0.048 * pow(D,1.68)
    
    
    # Average outside ambient temperature surrounding the equipment (example: 14 world wide, 11 for french standards)
    if (temperature_standard == world):
    
        t_ae = climate_world["T_ae_mean"]
    
    elif (temperature_standard == france):
    
        t_ae = climate_france["T_ae_mean"]

    # average ambient temperature of the printed circuit board (PCB) near the components
    t_ac = t_ae + 20
    
    
    # Mission profile:
    
    # permanent working: (average per cycle of t_ae)
    delta_t = t_ae      # delta_t = 0, if device in a controlled environment. (no variation).
    
    # Annual number of thermal cycles with amplitude (delta_t)
    # n_i = 365 # For the majority of applications, one day corresponds to one cycle
    n_i = 30            # Assuming data center operation with just a few forced power cycles
    
    
    # Mission profile stages: (1) full operation, (2) idle operation, (3) dormant/unused
    # 85% full operation, 5% idle, 10% dormant/unused
    
    # Simplified Mission Profile Table (from chatgpt):
        # Parameter	Typical Value
        # Ambient Temperature	20–35°C
        # Operating Temperature	40–70°C (component-level)
        # Relative Humidity	30–60% RH
        # Duty Cycle (Active)	90–95%
        # Power-On Hours	8,760 hours/year
        # Power Cycling (Hard Reboot)	2–4 cycles/year
        # Lifespan	3–7 years
        # Vibration	~0.001–0.005 g²/Hz
        # Shock	Negligible
        # Thermal Cycling: Typically ±5°C daily due to small HVAC or workload-induced variations.
        # ### Thao_1 = 0.006, Thao_2 = 0.046, Thao_3 = 0.006
    
    n1 = n_i*0.85
    n2 = n_i*0.05
    n3 = n_i*0.1
    
    # t1 = 55                 # average from chatgpt values
    # t2 = 40
    t3 = 27.5
    
    # Temperature of the junctions:
    
    tj_1 = t1 + delta_t_j
    tj_2 = t2 + delta_t_j
    tj_3 = t3 + delta_t_j
    
    # Delta of temperature among the mission profile stages:
    
    delta_t1 = (t1/3) + 5
    delta_t2 = (t2/3) + 5
    delta_t3 = (t3/3) + 5
    
    
    # Influence factor: (depends on the number of cycles per year)
    
    if (n_i > 8760):
        # Apply expression for several cycles: (ni > 8760 thermal cycles per year)
        pi_n1 = 1.7 * pow(n1, 0.60)
        pi_n2 = 1.7 * pow(n2, 0.60)
        pi_n3 = 1.7 * pow(n3, 0.60)
    
    else:
        # Apply expression for few cycles: (ni <= 8760 thermal cycles per year)
        pi_n1 = pow(n1, 0.60)
        pi_n2 = pow(n2, 0.60)
        pi_n3 = pow(n3, 0.60)
    
    sumatoria = pi_n1 * pow(delta_t1, 0.68) + pi_n2 * pow(delta_t2, 0.68) + pi_n3 * pow(delta_t3, 0.68)
    
    lamda_package = 2.75e-3 * pi_a * sumatoria * lamda_3
    
    # 
    # -----------------------------------------------------------------------------------------
    
    # third lamda parameter:
    # -----------------------------------------------------------------------------------------
    # 
    lamda_over_stress = 0 # the circuits is not an interface
    #
    # -----------------------------------------------------------------------------------------


    # First lamda parameter:
    # -----------------------------------------------------------------------------------------
    # 
    
    ### Number of transistors of the integrated circuit:
    N = ((4535 + 13486 + 13486 + 13486) * 16) + 600
    
    lamda_1 = 0.000012
    lamda_2 = 10.0
    
    eee = math.exp( -0.35 * 1)
    
    t_on  = 0.9
    t_off = 0.1
    
    Thao_1 = 0.006
    Thao_2 = 0.046
    Thao_3 = 0.006
    
    # Temperature factor:
    
    pi_t1 = math.exp( 3480 * ( (1/328) - (1 / (273 + tj_1) ) ) )
    pi_t2 = math.exp( 3480 * ( (1/328) - (1 / (273 + tj_2) ) ) )
    pi_t3 = math.exp( 3480 * ( (1/328) - (1 / (273 + tj_3) ) ) )
    
    lamda_die = ( lamda_1 * N * eee + lamda_2 ) * ( (pi_t1 * Thao_1) + (pi_t2 * Thao_2) + (pi_t3 * Thao_3) )
    
    # 
    # -----------------------------------------------------------------------------------------
    # 
    
    # -----------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------
    # calculating the lamda from the mathematical model (Page 31 IEC TR 62380:2004)
    total_lamda = (lamda_die + lamda_package + lamda_over_stress) * (1e-9) # / hours
    
    print("result:")
    print(total_lamda)
    
    
    # Clearly, at this stage the thermo mechanical effects cannot be computed and are assumed as 1.0 in the equation for the components.
    
def FIT_die(targeted_component_transistors, targeted_component_number_ports):
    
    ### Number of transistors of the integrated circuit:
    N = targeted_component_transistors          # Directly depends on the targeted component, so extracted from list.
    
    # Number of pins for the individual component:
    S = targeted_component_number_ports
    package_thermal_resistance_junction_ambient = (0.4 + 0.6 * cooling["forced_cooling"]) * ( 6.6 * (1.1e6 / (S*S)) )
    
    delta_t_j = package_thermal_resistance_junction_ambient * 0.5
    
    
    # Parameters extracted from Table 16, page 35 of the IEC 62380 standar
    lamda_1 = 0.000012              # standar cell, full custom
    lamda_2 = 10.0                  # standar cell, full custom
    
    eee = math.exp( -0.35 * 1)      # assuming a new component, less than one year of manufacturing., change 1 to evaluate several years of manufacturing
    
    # assuming a 90% of the execution working time.
    t_on  = 0.9
    t_off = 0.1
    
    Thao_1 = 0.006
    Thao_2 = 0.046
    Thao_3 = 0.006
    
    
    # Mission profile stages: (1) full operation, (2) idle operation, (3) dormant/unused
    # 85% full operation, 5% idle, 10% dormant/unused
    
    # Simplified Mission Profile Table (from chatgpt):
        # Parameter	Typical Value
        # Ambient Temperature	20–35°C
        # Operating Temperature	40–70°C (component-level)
        # Relative Humidity	30–60% RH
        # Duty Cycle (Active)	90–95%
        # Power-On Hours	8,760 hours/year
        # Power Cycling (Hard Reboot)	2–4 cycles/year
        # Lifespan	3–7 years
        # Vibration	~0.001–0.005 g²/Hz
        # Shock	Negligible
        # Thermal Cycling: Typically ±5°C daily due to small HVAC or workload-induced variations.
        # ### Thao_1 = 0.006, Thao_2 = 0.046, Thao_3 = 0.006
    
    t1 = 64.97                 # average from chatgpt values
    t2 = 40
    t3 = 27.5
    
    Thao_1 = 0.006
    Thao_2 = 0.046
    Thao_3 = 0.006
    
    # Temperature of the junctions:
    
    tj_1 = t1 + delta_t_j
    tj_2 = t2 + delta_t_j
    tj_3 = t3 + delta_t_j
    
    # Temperature factor:           (Assuming MOS BiCMOS, low voltage die circuit), page 33 of the IEC 62380 estandar
    
    pi_t1 = math.exp( 3480 * ( (1/328) - (1 / (273 + tj_1) ) ) )
    pi_t2 = math.exp( 3480 * ( (1/328) - (1 / (273 + tj_2) ) ) )
    pi_t3 = math.exp( 3480 * ( (1/328) - (1 / (273 + tj_3) ) ) )
    
    lamda_die = ( lamda_1 * N * eee + lamda_2 ) * ( (pi_t1 * Thao_1) + (pi_t2 * Thao_2) + (pi_t3 * Thao_3) ) * (1e-9) # / hours
    
    lamda_die2 = ( lamda_1 * N * eee + lamda_2 ) * ( (pi_t1 * Thao_1) + (pi_t2 * Thao_2) + (pi_t3 * Thao_3) )
    
    return(lamda_die, lamda_die2)




def main():
    
    
    # Other option is france
    temperature_standard = world
    print("overall aprox complete circuit")
    calculate_total_lamda(temperature_standard, 53.87, 46.5)
    
    
    Target_component = "ADD_1"
    print("overall aprox ADD component:")
    result, result2 = FIT_die( Transistors_per_Component[Target_component], Ports_per_Component[Target_component] )
    print( str(result), str(result2) + " FIT" )
    

    Target_component = "MUL_1"    
    print("overall aprox MUL component:")
    result, result2 = FIT_die( Transistors_per_Component[Target_component], Ports_per_Component[Target_component] )
    print( str(result), str(result2) + " FIT" )


    Target_component = "MAC_1"
    print("overall aprox MAC component:")
    result, result2 = FIT_die( Transistors_per_Component[Target_component], Ports_per_Component[Target_component] )
    print( str(result), str(result2) + " FIT" )


    Target_component = "ADD_2"
    print("overall aprox ADD 2 component:")
    result, result2 = FIT_die( Transistors_per_Component[Target_component], Ports_per_Component[Target_component] )
    print( str(result), str(result2) + " FIT" )


    Target_component = "MUL_2"
    print("overall aprox MUL 2 component:")
    result, result2 = FIT_die( Transistors_per_Component[Target_component], Ports_per_Component[Target_component] )
    print( str(result), str(result2) + " FIT" )


    Target_component = "MAC_2"
    print("overall aprox MAC 2 component:")
    result, result2 = FIT_die( Transistors_per_Component[Target_component], Ports_per_Component[Target_component] )
    print( str(result), str(result2) + " FIT" )

    # to calculate the FIT rate on individual components, we compute only the Lamda associated with the die, instead of a complete calculus involving the package.
    



main()


