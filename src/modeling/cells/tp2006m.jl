using Thunderbolt, StaticArrays
#=
   There are a total of 73 entries in the algebraic variable array.
   There are a total of 19 entries in each of the rate and state variable arrays.
   There are a total of 53 entries in the constant variable array.
=#
#=
   VOI is time in component environment (millisecond).
   STATES[1] is V in component membrane (millivolt).
   CONSTANTS[1] is R in component membrane (joule_per_mole_kelvin).
   CONSTANTS[2] is T in component membrane (kelvin).
   CONSTANTS[3] is F in component membrane (coulomb_per_millimole).
   CONSTANTS[4] is Cm in component membrane (picoF).
   CONSTANTS[5] is V_c in component membrane (micrometre3).
   ALGEBRAIC48 is i_K1 in component inward_rectifier_potassium_current (picoA_per_picoF).
   ALGEBRAIC55 is i_to in component transient_outward_current (picoA_per_picoF).
   ALGEBRAIC49 is i_Kr in component rapid_time_dependent_potassium_current (picoA_per_picoF).
   ALGEBRAIC50 is i_Ks in component slow_time_dependent_potassium_current (picoA_per_picoF).
   ALGEBRAIC53 is i_CaL in component L_type_Ca_current (picoA_per_picoF).
   ALGEBRAIC56 is i_NaK in component sodium_potassium_pump_current (picoA_per_picoF).
   ALGEBRAIC51 is i_Na in component fast_sodium_current (picoA_per_picoF).
   ALGEBRAIC52 is i_b_Na in component sodium_background_current (picoA_per_picoF).
   ALGEBRAIC57 is i_NaCa in component sodium_calcium_exchanger_current (picoA_per_picoF).
   ALGEBRAIC54 is i_b_Ca in component calcium_background_current (picoA_per_picoF).
   ALGEBRAIC59 is i_p_K in component potassium_pump_current (picoA_per_picoF).
   ALGEBRAIC58 is i_p_Ca in component calcium_pump_current (picoA_per_picoF).
   ALGEBRAIC13 is i_Stim in component membrane (picoA_per_picoF).
   CONSTANTS[6] is stim_start in component membrane (millisecond).
   CONSTANTS[7] is stim_period in component membrane (millisecond).
   CONSTANTS[8] is stim_duration in component membrane (millisecond).
   CONSTANTS[9] is stim_amplitude in component membrane (picoA_per_picoF).
   ALGEBRAIC26 is E_Na in component reversal_potentials (millivolt).
   ALGEBRAIC34 is E_K in component reversal_potentials (millivolt).
   ALGEBRAIC42 is E_Ks in component reversal_potentials (millivolt).
   ALGEBRAIC44 is E_Ca in component reversal_potentials (millivolt).
   CONSTANTS[10] is P_kna in component reversal_potentials (dimensionless).
   CONSTANTS[11] is K_o in component potassium_dynamics (millimolar).
   CONSTANTS[12] is Na_o in component sodium_dynamics (millimolar).
   STATES[2] is K_i in component potassium_dynamics (millimolar).
   STATES[3] is Na_i in component sodium_dynamics (millimolar).
   CONSTANTS[13] is Ca_o in component calcium_dynamics (millimolar).
   STATES[4] is Ca_i in component calcium_dynamics (millimolar).
   CONSTANTS[14] is g_K1 in component inward_rectifier_potassium_current (nanoS_per_picoF).
   ALGEBRAIC47 is xK1_inf in component inward_rectifier_potassium_current (dimensionless).
   ALGEBRAIC45 is alpha_K1 in component inward_rectifier_potassium_current (dimensionless).
   ALGEBRAIC46 is beta_K1 in component inward_rectifier_potassium_current (dimensionless).
   CONSTANTS[15] is g_Kr in component rapid_time_dependent_potassium_current (nanoS_per_picoF).
   STATES[5] is Xr1 in component rapid_time_dependent_potassium_current_Xr1_gate (dimensionless).
   STATES[6] is Xr2 in component rapid_time_dependent_potassium_current_Xr2_gate (dimensionless).
   ALGEBRAIC1 is xr1_inf in component rapid_time_dependent_potassium_current_Xr1_gate (dimensionless).
   ALGEBRAIC14 is alpha_xr1 in component rapid_time_dependent_potassium_current_Xr1_gate (dimensionless).
   ALGEBRAIC27 is beta_xr1 in component rapid_time_dependent_potassium_current_Xr1_gate (dimensionless).
   ALGEBRAIC35 is tau_xr1 in component rapid_time_dependent_potassium_current_Xr1_gate (millisecond).
   ALGEBRAIC2 is xr2_inf in component rapid_time_dependent_potassium_current_Xr2_gate (dimensionless).
   ALGEBRAIC15 is alpha_xr2 in component rapid_time_dependent_potassium_current_Xr2_gate (dimensionless).
   ALGEBRAIC28 is beta_xr2 in component rapid_time_dependent_potassium_current_Xr2_gate (dimensionless).
   ALGEBRAIC36 is tau_xr2 in component rapid_time_dependent_potassium_current_Xr2_gate (millisecond).
   CONSTANTS[16] is g_Ks in component slow_time_dependent_potassium_current (nanoS_per_picoF).
   STATES[7] is Xs in component slow_time_dependent_potassium_current_Xs_gate (dimensionless).
   ALGEBRAIC3 is xs_inf in component slow_time_dependent_potassium_current_Xs_gate (dimensionless).
   ALGEBRAIC16 is alpha_xs in component slow_time_dependent_potassium_current_Xs_gate (dimensionless).
   ALGEBRAIC29 is beta_xs in component slow_time_dependent_potassium_current_Xs_gate (dimensionless).
   ALGEBRAIC37 is tau_xs in component slow_time_dependent_potassium_current_Xs_gate (millisecond).
   CONSTANTS[17] is g_Na in component fast_sodium_current (nanoS_per_picoF).
   STATES[8] is m in component fast_sodium_current_m_gate (dimensionless).
   STATES[9] is h in component fast_sodium_current_h_gate (dimensionless).
   STATES[10] is j in component fast_sodium_current_j_gate (dimensionless).
   ALGEBRAIC4 is m_inf in component fast_sodium_current_m_gate (dimensionless).
   ALGEBRAIC17 is alpha_m in component fast_sodium_current_m_gate (dimensionless).
   ALGEBRAIC30 is beta_m in component fast_sodium_current_m_gate (dimensionless).
   ALGEBRAIC38 is tau_m in component fast_sodium_current_m_gate (millisecond).
   ALGEBRAIC5 is h_inf in component fast_sodium_current_h_gate (dimensionless).
   ALGEBRAIC18 is alpha_h in component fast_sodium_current_h_gate (per_millisecond).
   ALGEBRAIC31 is beta_h in component fast_sodium_current_h_gate (per_millisecond).
   ALGEBRAIC39 is tau_h in component fast_sodium_current_h_gate (millisecond).
   ALGEBRAIC6 is j_inf in component fast_sodium_current_j_gate (dimensionless).
   ALGEBRAIC19 is alpha_j in component fast_sodium_current_j_gate (per_millisecond).
   ALGEBRAIC32 is beta_j in component fast_sodium_current_j_gate (per_millisecond).
   ALGEBRAIC40 is tau_j in component fast_sodium_current_j_gate (millisecond).
   CONSTANTS[18] is g_bna in component sodium_background_current (nanoS_per_picoF).
   CONSTANTS[19] is g_CaL in component L_type_Ca_current (litre_per_farad_second).
   STATES[11] is Ca_ss in component calcium_dynamics (millimolar).
   STATES[12] is d in component L_type_Ca_current_d_gate (dimensionless).
   STATES[13] is f in component L_type_Ca_current_f_gate (dimensionless).
   STATES[14] is f2 in component L_type_Ca_current_f2_gate (dimensionless).
   STATES[15] is fCass in component L_type_Ca_current_fCass_gate (dimensionless).
   ALGEBRAIC7 is d_inf in component L_type_Ca_current_d_gate (dimensionless).
   ALGEBRAIC20 is alpha_d in component L_type_Ca_current_d_gate (dimensionless).
   ALGEBRAIC33 is beta_d in component L_type_Ca_current_d_gate (dimensionless).
   ALGEBRAIC41 is gamma_d in component L_type_Ca_current_d_gate (millisecond).
   ALGEBRAIC43 is tau_d in component L_type_Ca_current_d_gate (millisecond).
   ALGEBRAIC8 is f_inf in component L_type_Ca_current_f_gate (dimensionless).
   ALGEBRAIC21 is tau_f in component L_type_Ca_current_f_gate (millisecond).
   ALGEBRAIC9 is f2_inf in component L_type_Ca_current_f2_gate (dimensionless).
   ALGEBRAIC22 is tau_f2 in component L_type_Ca_current_f2_gate (millisecond).
   ALGEBRAIC10 is fCass_inf in component L_type_Ca_current_fCass_gate (dimensionless).
   ALGEBRAIC23 is tau_fCass in component L_type_Ca_current_fCass_gate (millisecond).
   CONSTANTS[20] is g_bca in component calcium_background_current (nanoS_per_picoF).
   CONSTANTS[21] is g_to in component transient_outward_current (nanoS_per_picoF).
   STATES[16] is s in component transient_outward_current_s_gate (dimensionless).
   STATES[17] is r in component transient_outward_current_r_gate (dimensionless).
   ALGEBRAIC11 is s_inf in component transient_outward_current_s_gate (dimensionless).
   ALGEBRAIC24 is tau_s in component transient_outward_current_s_gate (millisecond).
   ALGEBRAIC12 is r_inf in component transient_outward_current_r_gate (dimensionless).
   ALGEBRAIC25 is tau_r in component transient_outward_current_r_gate (millisecond).
   CONSTANTS[22] is P_NaK in component sodium_potassium_pump_current (picoA_per_picoF).
   CONSTANTS[23] is K_mk in component sodium_potassium_pump_current (millimolar).
   CONSTANTS[24] is K_mNa in component sodium_potassium_pump_current (millimolar).
   CONSTANTS[25] is K_NaCa in component sodium_calcium_exchanger_current (picoA_per_picoF).
   CONSTANTS[26] is K_sat in component sodium_calcium_exchanger_current (dimensionless).
   CONSTANTS[27] is alpha in component sodium_calcium_exchanger_current (dimensionless).
   CONSTANTS[28] is gamma in component sodium_calcium_exchanger_current (dimensionless).
   CONSTANTS[29] is Km_Ca in component sodium_calcium_exchanger_current (millimolar).
   CONSTANTS[30] is Km_Nai in component sodium_calcium_exchanger_current (millimolar).
   CONSTANTS[31] is g_pCa in component calcium_pump_current (picoA_per_picoF).
   CONSTANTS[32] is K_pCa in component calcium_pump_current (millimolar).
   CONSTANTS[33] is g_pK in component potassium_pump_current (nanoS_per_picoF).
   STATES[18] is Ca_SR in component calcium_dynamics (millimolar).
   ALGEBRAIC69 is i_rel in component calcium_dynamics (millimolar_per_millisecond).
   ALGEBRAIC60 is i_up in component calcium_dynamics (millimolar_per_millisecond).
   ALGEBRAIC61 is i_leak in component calcium_dynamics (millimolar_per_millisecond).
   ALGEBRAIC62 is i_xfer in component calcium_dynamics (millimolar_per_millisecond).
   ALGEBRAIC68 is O in component calcium_dynamics (dimensionless).
   STATES[19] is R_prime in component calcium_dynamics (dimensionless).
   ALGEBRAIC65 is k1 in component calcium_dynamics (per_millimolar2_per_millisecond).
   ALGEBRAIC66 is k2 in component calcium_dynamics (per_millimolar_per_millisecond).
   CONSTANTS[34] is k1_prime in component calcium_dynamics (per_millimolar2_per_millisecond).
   CONSTANTS[35] is k2_prime in component calcium_dynamics (per_millimolar_per_millisecond).
   CONSTANTS[36] is k3 in component calcium_dynamics (per_millisecond).
   CONSTANTS[37] is k4 in component calcium_dynamics (per_millisecond).
   CONSTANTS[38] is EC in component calcium_dynamics (millimolar).
   CONSTANTS[39] is max_sr in component calcium_dynamics (dimensionless).
   CONSTANTS[40] is min_sr in component calcium_dynamics (dimensionless).
   ALGEBRAIC63 is kcasr in component calcium_dynamics (dimensionless).
   CONSTANTS[41] is V_rel in component calcium_dynamics (per_millisecond).
   CONSTANTS[42] is V_xfer in component calcium_dynamics (per_millisecond).
   CONSTANTS[43] is K_up in component calcium_dynamics (millimolar).
   CONSTANTS[44] is V_leak in component calcium_dynamics (per_millisecond).
   CONSTANTS[45] is Vmax_up in component calcium_dynamics (millimolar_per_millisecond).
   ALGEBRAIC64 is ddt_Ca_i_total in component calcium_dynamics (millimolar_per_millisecond).
   ALGEBRAIC70 is ddt_Ca_sr_total in component calcium_dynamics (millimolar_per_millisecond).
   ALGEBRAIC71 is ddt_Ca_ss_total in component calcium_dynamics (millimolar_per_millisecond).
   ALGEBRAIC67 is f_JCa_i_free in component calcium_dynamics (dimensionless).
   ALGEBRAIC72 is f_JCa_sr_free in component calcium_dynamics (dimensionless).
   ALGEBRAIC73 is f_JCa_ss_free in component calcium_dynamics (dimensionless).
   CONSTANTS[46] is Buf_c in component calcium_dynamics (millimolar).
   CONSTANTS[47] is K_buf_c in component calcium_dynamics (millimolar).
   CONSTANTS[48] is Buf_sr in component calcium_dynamics (millimolar).
   CONSTANTS[49] is K_buf_sr in component calcium_dynamics (millimolar).
   CONSTANTS[50] is Buf_ss in component calcium_dynamics (millimolar).
   CONSTANTS[51] is K_buf_ss in component calcium_dynamics (millimolar).
   CONSTANTS[52] is V_sr in component calcium_dynamics (micrometre3).
   CONSTANTS[53] is V_ss in component calcium_dynamics (micrometre3).
   RATES[1] is d/dt V in component membrane (millivolt).
   RATES[5] is d/dt Xr1 in component rapid_time_dependent_potassium_current_Xr1_gate (dimensionless).
   RATES[6] is d/dt Xr2 in component rapid_time_dependent_potassium_current_Xr2_gate (dimensionless).
   RATES[7] is d/dt Xs in component slow_time_dependent_potassium_current_Xs_gate (dimensionless).
   RATES[8] is d/dt m in component fast_sodium_current_m_gate (dimensionless).
   RATES[9] is d/dt h in component fast_sodium_current_h_gate (dimensionless).
   RATES[10] is d/dt j in component fast_sodium_current_j_gate (dimensionless).
   RATES[12] is d/dt d in component L_type_Ca_current_d_gate (dimensionless).
   RATES[13] is d/dt f in component L_type_Ca_current_f_gate (dimensionless).
   RATES[14] is d/dt f2 in component L_type_Ca_current_f2_gate (dimensionless).
   RATES[15] is d/dt fCass in component L_type_Ca_current_fCass_gate (dimensionless).
   RATES[16] is d/dt s in component transient_outward_current_s_gate (dimensionless).
   RATES[17] is d/dt r in component transient_outward_current_r_gate (dimensionless).
   RATES[19] is d/dt R_prime in component calcium_dynamics (dimensionless).
   RATES[4] is d/dt Ca_i in component calcium_dynamics (millimolar).
   RATES[18] is d/dt Ca_SR in component calcium_dynamics (millimolar).
   RATES[11] is d/dt Ca_ss in component calcium_dynamics (millimolar).
   RATES[3] is d/dt Na_i in component sodium_dynamics (millimolar).
   RATES[2] is d/dt K_i in component potassium_dynamics (millimolar).
=#
# CELLML model
module TP2006M

using StaticArrays
using Adapt

function initConsts(CONSTANTS, RATES, STATES)
  STATES[1] = -85.423;
  CONSTANTS[1] = 8.314;
  CONSTANTS[2] = 310;
  CONSTANTS[3] = 96.485;
  CONSTANTS[4] = 185;
  CONSTANTS[5] = 16404;
  CONSTANTS[6] = 10;
  CONSTANTS[7] = 1000;
  CONSTANTS[8] = 1;
  CONSTANTS[9] = -52;
  CONSTANTS[10] = 0.03;
  CONSTANTS[11] = 5.4;
  CONSTANTS[12] = 140;
  STATES[2] = 138.52;
  STATES[3] = 10.132;
  CONSTANTS[13] = 2;
  STATES[4] = 0.000153;
  CONSTANTS[14] = 5.405;
  CONSTANTS[15] = 0.153;
  STATES[5] = 0.0165;
  STATES[6] = 0.473;
  CONSTANTS[16] = 0.098;
  STATES[7] = 0.0174;
  CONSTANTS[17] = 14.838;
  STATES[8] = 0.00165;
  STATES[9] = 0.749;
  STATES[10] = 0.6788;
  CONSTANTS[18] = 0.00029;
  CONSTANTS[19] = 0.0398;
  STATES[11] = 0.00042;
  STATES[12] = 3.288e-5;
  STATES[13] = 0.7026;
  STATES[14] = 0.9526;
  STATES[15] = 0.9942;
  CONSTANTS[20] = 0.000592;
  CONSTANTS[21] = 0.294;
  STATES[16] = 0.999998;
  STATES[17] = 2.347e-8;
  CONSTANTS[22] = 2.724;
  CONSTANTS[23] = 1;
  CONSTANTS[24] = 40;
  CONSTANTS[25] = 1000;
  CONSTANTS[26] = 0.1;
  CONSTANTS[27] = 2.5;
  CONSTANTS[28] = 0.35;
  CONSTANTS[29] = 1.38;
  CONSTANTS[30] = 87.5;
  CONSTANTS[31] = 0.1238;
  CONSTANTS[32] = 0.0005;
  CONSTANTS[33] = 0.0146;
  STATES[18] = 4.272;
  STATES[19] = 0.8978;
  CONSTANTS[34] = 0.15;
  CONSTANTS[35] = 0.045;
  CONSTANTS[36] = 0.06;
  CONSTANTS[37] = 0.005;
  CONSTANTS[38] = 1.5;
  CONSTANTS[39] = 2.5;
  CONSTANTS[40] = 1;
  CONSTANTS[41] = 0.102;
  CONSTANTS[42] = 0.0038;
  CONSTANTS[43] = 0.00025;
  CONSTANTS[44] = 0.00036;
  CONSTANTS[45] = 0.006375;
  CONSTANTS[46] = 0.2;
  CONSTANTS[47] = 0.001;
  CONSTANTS[48] = 10;
  CONSTANTS[49] = 0.3;
  CONSTANTS[50] = 0.4;
  CONSTANTS[51] = 0.00025;
  CONSTANTS[52] = 1094;
  CONSTANTS[53] = 54.68;
end

@inline function computeRates(VOI::T1, CONSTANTS::T2, RATES::T3, STATES::T4) where {T1,T2,T3,T4,T5}
  ALGEBRAIC8 = 1.00000/(1.00000+exp((STATES[1]+20.0000)/7.00000));
  ALGEBRAIC21 =  1102.50*exp(- ^(STATES[1]+27.0000, 2.00000)/225.000)+200.000/(1.00000+exp((13.0000 - STATES[1])/10.0000))+180.000/(1.00000+exp((STATES[1]+30.0000)/10.0000))+20.0000;
  RATES[13] = (ALGEBRAIC8 - STATES[13])/ALGEBRAIC21;
  ALGEBRAIC9 = 0.670000/(1.00000+exp((STATES[1]+35.0000)/7.00000))+0.330000;
  ALGEBRAIC22 =  562.000*exp(- ^(STATES[1]+27.0000, 2.00000)/240.000)+31.0000/(1.00000+exp((25.0000 - STATES[1])/10.0000))+80.0000/(1.00000+exp((STATES[1]+30.0000)/10.0000));
  RATES[14] = (ALGEBRAIC9 - STATES[14])/ALGEBRAIC22;
  ALGEBRAIC10 = 0.600000/(1.00000+^(STATES[11]/0.0500000, 2.00000))+0.400000;
  ALGEBRAIC23 = 80.0000/(1.00000+^(STATES[11]/0.0500000, 2.00000))+2.00000;
  RATES[15] = (ALGEBRAIC10 - STATES[15])/ALGEBRAIC23;
  ALGEBRAIC11 = 1.00000/(1.00000+exp((STATES[1]+20.0000)/5.00000));
  ALGEBRAIC24 =  85.0000*exp(- ^(STATES[1]+45.0000, 2.00000)/320.000)+5.00000/(1.00000+exp((STATES[1] - 20.0000)/5.00000))+3.00000;
  RATES[16] = (ALGEBRAIC11 - STATES[16])/ALGEBRAIC24;
  ALGEBRAIC12 = 1.00000/(1.00000+exp((20.0000 - STATES[1])/6.00000));
  ALGEBRAIC25 =  9.50000*exp(- ^(STATES[1]+40.0000, 2.00000)/1800.00)+0.800000;
  RATES[17] = (ALGEBRAIC12 - STATES[17])/ALGEBRAIC25;
  ALGEBRAIC1 = 1.00000/(1.00000+exp((- 26.0000 - STATES[1])/7.00000));
  ALGEBRAIC14 = 450.000/(1.00000+exp((- 45.0000 - STATES[1])/10.0000));
  ALGEBRAIC27 = 6.00000/(1.00000+exp((STATES[1]+30.0000)/11.5000));
  ALGEBRAIC35 =  1.00000*ALGEBRAIC14*ALGEBRAIC27;
  RATES[5] = (ALGEBRAIC1 - STATES[5])/ALGEBRAIC35;
  ALGEBRAIC2 = 1.00000/(1.00000+exp((STATES[1]+88.0000)/24.0000));
  ALGEBRAIC15 = 3.00000/(1.00000+exp((- 60.0000 - STATES[1])/20.0000));
  ALGEBRAIC28 = 1.12000/(1.00000+exp((STATES[1] - 60.0000)/20.0000));
  ALGEBRAIC36 =  1.00000*ALGEBRAIC15*ALGEBRAIC28;
  RATES[6] = (ALGEBRAIC2 - STATES[6])/ALGEBRAIC36;
  ALGEBRAIC3 = 1.00000/(1.00000+exp((- 5.00000 - STATES[1])/14.0000));
  ALGEBRAIC16 = 1400.00/ ^((1.00000+exp((5.00000 - STATES[1])/6.00000)), 1.0 / 2);
  ALGEBRAIC29 = 1.00000/(1.00000+exp((STATES[1] - 35.0000)/15.0000));
  ALGEBRAIC37 =  1.00000*ALGEBRAIC16*ALGEBRAIC29+80.0000;
  RATES[7] = (ALGEBRAIC3 - STATES[7])/ALGEBRAIC37;
  ALGEBRAIC4 = 1.00000/^(1.00000+exp((- 56.8600 - STATES[1])/9.03000), 2.00000);
  ALGEBRAIC17 = 1.00000/(1.00000+exp((- 60.0000 - STATES[1])/5.00000));
  ALGEBRAIC30 = 0.100000/(1.00000+exp((STATES[1]+35.0000)/5.00000))+0.100000/(1.00000+exp((STATES[1] - 50.0000)/200.000));
  ALGEBRAIC38 =  1.00000*ALGEBRAIC17*ALGEBRAIC30;
  RATES[8] = (ALGEBRAIC4 - STATES[8])/ALGEBRAIC38;
  ALGEBRAIC5 = 1.00000/^(1.00000+exp((STATES[1]+71.5500)/7.43000), 2.00000);
  ALGEBRAIC18 = (STATES[1]<- 40.0000 ?  0.0570000*exp(- (STATES[1]+80.0000)/6.80000) : 0.00000);
  ALGEBRAIC31 = (STATES[1]<- 40.0000 ?  2.70000*exp( 0.0790000*STATES[1])+ 310000.0*exp( 0.348500*STATES[1]) : 0.770000/( 0.130000*(1.00000+exp((STATES[1]+10.6600)/- 11.1000))));
  ALGEBRAIC39 = 1.00000/(ALGEBRAIC18+ALGEBRAIC31);
  RATES[9] = (ALGEBRAIC5 - STATES[9])/ALGEBRAIC39;
  ALGEBRAIC6 = 1.00000/^(1.00000+exp((STATES[1]+71.5500)/7.43000), 2.00000);
  ALGEBRAIC19 = (STATES[1]<- 40.0000 ? (( ( - 25428.0*exp( 0.244400*STATES[1]) -  6.94800e-06*exp( - 0.0439100*STATES[1]))*(STATES[1]+37.7800))/1.00000)/(1.00000+exp( 0.311000*(STATES[1]+79.2300))) : 0.00000);
  ALGEBRAIC32 = (STATES[1]<- 40.0000 ? ( 0.0242400*exp( - 0.0105200*STATES[1]))/(1.00000+exp( - 0.137800*(STATES[1]+40.1400))) : ( 0.600000*exp( 0.0570000*STATES[1]))/(1.00000+exp( - 0.100000*(STATES[1]+32.0000))));
  ALGEBRAIC40 = 1.00000/(ALGEBRAIC19+ALGEBRAIC32);
  RATES[10] = (ALGEBRAIC6 - STATES[10])/ALGEBRAIC40;
  ALGEBRAIC7 = 1.00000/(1.00000+exp((- 8.00000 - STATES[1])/7.50000));
  ALGEBRAIC20 = 1.40000/(1.00000+exp((- 35.0000 - STATES[1])/13.0000))+0.250000;
  ALGEBRAIC33 = 1.40000/(1.00000+exp((STATES[1]+5.00000)/5.00000));
  ALGEBRAIC41 = 1.00000/(1.00000+exp((50.0000 - STATES[1])/20.0000));
  ALGEBRAIC43 =  1.00000*ALGEBRAIC20*ALGEBRAIC33+ALGEBRAIC41;
  RATES[12] = (ALGEBRAIC7 - STATES[12])/ALGEBRAIC43;
  ALGEBRAIC56 = (( (( CONSTANTS[22]*CONSTANTS[11])/(CONSTANTS[11]+CONSTANTS[23]))*STATES[3])/(STATES[3]+CONSTANTS[24]))/(1.00000+ 0.124500*exp(( - 0.100000*STATES[1]*CONSTANTS[3])/( CONSTANTS[1]*CONSTANTS[2]))+ 0.0353000*exp(( - STATES[1]*CONSTANTS[3])/( CONSTANTS[1]*CONSTANTS[2])));
  ALGEBRAIC26 =  (( CONSTANTS[1]*CONSTANTS[2])/CONSTANTS[3])*log(CONSTANTS[12]/STATES[3]);
  ALGEBRAIC51 =  CONSTANTS[17]*^(STATES[8], 3.00000)*STATES[9]*STATES[10]*(STATES[1] - ALGEBRAIC26);
  ALGEBRAIC52 =  CONSTANTS[18]*(STATES[1] - ALGEBRAIC26);
  ALGEBRAIC57 = ( CONSTANTS[25]*( exp(( CONSTANTS[28]*STATES[1]*CONSTANTS[3])/( CONSTANTS[1]*CONSTANTS[2]))*^(STATES[3], 3.00000)*CONSTANTS[13] -  exp(( (CONSTANTS[28] - 1.00000)*STATES[1]*CONSTANTS[3])/( CONSTANTS[1]*CONSTANTS[2]))*^(CONSTANTS[12], 3.00000)*STATES[4]*CONSTANTS[27]))/( (^(CONSTANTS[30], 3.00000)+^(CONSTANTS[12], 3.00000))*(CONSTANTS[29]+CONSTANTS[13])*(1.00000+ CONSTANTS[26]*exp(( (CONSTANTS[28] - 1.00000)*STATES[1]*CONSTANTS[3])/( CONSTANTS[1]*CONSTANTS[2]))));
  RATES[3] =  (- (ALGEBRAIC51+ALGEBRAIC52+ 3.00000*ALGEBRAIC56+ 3.00000*ALGEBRAIC57)/( CONSTANTS[5]*CONSTANTS[3]))*CONSTANTS[4];
  ALGEBRAIC34 =  (( CONSTANTS[1]*CONSTANTS[2])/CONSTANTS[3])*log(CONSTANTS[11]/STATES[2]);
  ALGEBRAIC45 = 0.100000/(1.00000+exp( 0.0600000*((STATES[1] - ALGEBRAIC34) - 200.000)));
  ALGEBRAIC46 = ( 3.00000*exp( 0.000200000*((STATES[1] - ALGEBRAIC34)+100.000))+exp( 0.100000*((STATES[1] - ALGEBRAIC34) - 10.0000)))/(1.00000+exp( - 0.500000*(STATES[1] - ALGEBRAIC34)));
  ALGEBRAIC47 = ALGEBRAIC45/(ALGEBRAIC45+ALGEBRAIC46);
  ALGEBRAIC48 =  CONSTANTS[14]*ALGEBRAIC47* ^((CONSTANTS[11]/5.40000), 1.0 / 2)*(STATES[1] - ALGEBRAIC34);
  ALGEBRAIC55 =  CONSTANTS[21]*STATES[17]*STATES[16]*(STATES[1] - ALGEBRAIC34);
  ALGEBRAIC49 =  CONSTANTS[15]* ^((CONSTANTS[11]/5.40000), 1.0 / 2)*STATES[5]*STATES[6]*(STATES[1] - ALGEBRAIC34);
  ALGEBRAIC42 =  (( CONSTANTS[1]*CONSTANTS[2])/CONSTANTS[3])*log((CONSTANTS[11]+ CONSTANTS[10]*CONSTANTS[12])/(STATES[2]+ CONSTANTS[10]*STATES[3]));
  ALGEBRAIC50 =  CONSTANTS[16]*^(STATES[7], 2.00000)*(STATES[1] - ALGEBRAIC42);
  ALGEBRAIC53 = if STATES[1] ≈ 15.0
    0.0
  else
    ( (( CONSTANTS[19]*STATES[12]*STATES[13]*STATES[14]*STATES[15]*4.00000*(STATES[1] - 15.0000)*^(CONSTANTS[3], 2.00000))/( CONSTANTS[1]*CONSTANTS[2]))*( 0.250000*STATES[11]*exp(( 2.00000*(STATES[1] - 15.0000)*CONSTANTS[3])/( CONSTANTS[1]*CONSTANTS[2])) - CONSTANTS[13]))/(exp(( 2.00000*(STATES[1] - 15.0000)*CONSTANTS[3])/( CONSTANTS[1]*CONSTANTS[2])) - 1.00000);
  end
  ALGEBRAIC44 =  (( 0.500000*CONSTANTS[1]*CONSTANTS[2])/CONSTANTS[3])*log(CONSTANTS[13]/STATES[4]);
  ALGEBRAIC54 =  CONSTANTS[20]*(STATES[1] - ALGEBRAIC44);
  ALGEBRAIC59 = ( CONSTANTS[33]*(STATES[1] - ALGEBRAIC34))/(1.00000+exp((25.0000 - STATES[1])/5.98000));
  ALGEBRAIC58 = ( CONSTANTS[31]*STATES[4])/(STATES[4]+CONSTANTS[32]);
  ALGEBRAIC13 = 0.0# =stim (VOI -  floor(VOI/CONSTANTS[7])*CONSTANTS[7]>=CONSTANTS[6]&&VOI -  floor(VOI/CONSTANTS[7])*CONSTANTS[7]<=CONSTANTS[6]+CONSTANTS[8] ? CONSTANTS[9] : 0.00000);
  RATES[2] =  (- ((ALGEBRAIC48+ALGEBRAIC55+ALGEBRAIC49+ALGEBRAIC50+ALGEBRAIC59+ALGEBRAIC13) -  2.00000*ALGEBRAIC56)/( CONSTANTS[5]*CONSTANTS[3]))*CONSTANTS[4];
  ALGEBRAIC63 = CONSTANTS[39] - (CONSTANTS[39] - CONSTANTS[40])/(1.00000+^(CONSTANTS[38]/STATES[18], 2.00000));
  ALGEBRAIC66 =  CONSTANTS[35]*ALGEBRAIC63;
  RATES[19] =  - ALGEBRAIC66*STATES[11]*STATES[19]+ CONSTANTS[37]*(1.00000 - STATES[19]);
  ALGEBRAIC60 = CONSTANTS[45]/(1.00000+^(CONSTANTS[43], 2.00000)/^(STATES[4], 2.00000));
  ALGEBRAIC61 =  CONSTANTS[44]*(STATES[18] - STATES[4]);
  ALGEBRAIC62 =  CONSTANTS[42]*(STATES[11] - STATES[4]);
  ALGEBRAIC64 = ( - ((ALGEBRAIC54+ALGEBRAIC58) -  2.00000*ALGEBRAIC57)*CONSTANTS[4])/( 2.00000*CONSTANTS[5]*CONSTANTS[3])+( (ALGEBRAIC61 - ALGEBRAIC60)*CONSTANTS[52])/CONSTANTS[5]+ALGEBRAIC62;
  ALGEBRAIC67 = 1.00000/(1.00000+( CONSTANTS[46]*CONSTANTS[47])/^(STATES[4]+CONSTANTS[47], 2.00000));
  RATES[4] =  ALGEBRAIC64*ALGEBRAIC67;
  ALGEBRAIC65 = CONSTANTS[34]/ALGEBRAIC63;
  ALGEBRAIC68 = ( ALGEBRAIC65*^(STATES[11], 2.00000)*STATES[19])/(CONSTANTS[36]+ ALGEBRAIC65*^(STATES[11], 2.00000));
  ALGEBRAIC69 =  CONSTANTS[41]*ALGEBRAIC68*(STATES[18] - STATES[11]);
  ALGEBRAIC70 = ALGEBRAIC60 - (ALGEBRAIC69+ALGEBRAIC61);
  ALGEBRAIC72 = 1.00000/(1.00000+( CONSTANTS[48]*CONSTANTS[49])/^(STATES[18]+CONSTANTS[49], 2.00000));
  RATES[18] =  ALGEBRAIC70*ALGEBRAIC72;
  ALGEBRAIC71 = (( - ALGEBRAIC53*CONSTANTS[4])/( 2.00000*CONSTANTS[53]*CONSTANTS[3])+( ALGEBRAIC69*CONSTANTS[52])/CONSTANTS[53]) - ( ALGEBRAIC62*CONSTANTS[5])/CONSTANTS[53];
  ALGEBRAIC73 = 1.00000/(1.00000+( CONSTANTS[50]*CONSTANTS[51])/^(STATES[11]+CONSTANTS[51], 2.00000));
  RATES[11] =  ALGEBRAIC71*ALGEBRAIC73;
  xIstretch = 0.0
  
  RATES[1] = - (ALGEBRAIC48+ALGEBRAIC55+ALGEBRAIC49+ALGEBRAIC50+ALGEBRAIC53+ALGEBRAIC56+ALGEBRAIC51+ALGEBRAIC52+ALGEBRAIC57+ALGEBRAIC54+ALGEBRAIC59+ALGEBRAIC58+ALGEBRAIC13);

end

function computeVariables(VOI, CONSTANTS, RATES, STATES, ALGEBRAIC)
  ALGEBRAIC8 = 1.00000/(1.00000+exp((STATES[1]+20.0000)/7.00000));
  ALGEBRAIC21 =  1102.50*exp(- ^(STATES[1]+27.0000, 2.00000)/225.000)+200.000/(1.00000+exp((13.0000 - STATES[1])/10.0000))+180.000/(1.00000+exp((STATES[1]+30.0000)/10.0000))+20.0000;
  ALGEBRAIC9 = 0.670000/(1.00000+exp((STATES[1]+35.0000)/7.00000))+0.330000;
  ALGEBRAIC22 =  562.000*exp(- ^(STATES[1]+27.0000, 2.00000)/240.000)+31.0000/(1.00000+exp((25.0000 - STATES[1])/10.0000))+80.0000/(1.00000+exp((STATES[1]+30.0000)/10.0000));
  ALGEBRAIC10 = 0.600000/(1.00000+^(STATES[11]/0.0500000, 2.00000))+0.400000;
  ALGEBRAIC23 = 80.0000/(1.00000+^(STATES[11]/0.0500000, 2.00000))+2.00000;
  ALGEBRAIC11 = 1.00000/(1.00000+exp((STATES[1]+20.0000)/5.00000));
  ALGEBRAIC24 =  85.0000*exp(- ^(STATES[1]+45.0000, 2.00000)/320.000)+5.00000/(1.00000+exp((STATES[1] - 20.0000)/5.00000))+3.00000;
  ALGEBRAIC12 = 1.00000/(1.00000+exp((20.0000 - STATES[1])/6.00000));
  ALGEBRAIC25 =  9.50000*exp(- ^(STATES[1]+40.0000, 2.00000)/1800.00)+0.800000;
  ALGEBRAIC1 = 1.00000/(1.00000+exp((- 26.0000 - STATES[1])/7.00000));
  ALGEBRAIC14 = 450.000/(1.00000+exp((- 45.0000 - STATES[1])/10.0000));
  ALGEBRAIC27 = 6.00000/(1.00000+exp((STATES[1]+30.0000)/11.5000));
  ALGEBRAIC35 =  1.00000*ALGEBRAIC14*ALGEBRAIC27;
  ALGEBRAIC2 = 1.00000/(1.00000+exp((STATES[1]+88.0000)/24.0000));
  ALGEBRAIC15 = 3.00000/(1.00000+exp((- 60.0000 - STATES[1])/20.0000));
  ALGEBRAIC28 = 1.12000/(1.00000+exp((STATES[1] - 60.0000)/20.0000));
  ALGEBRAIC36 =  1.00000*ALGEBRAIC15*ALGEBRAIC28;
  ALGEBRAIC3 = 1.00000/(1.00000+exp((- 5.00000 - STATES[1])/14.0000));
  ALGEBRAIC16 = 1400.00/ ^((1.00000+exp((5.00000 - STATES[1])/6.00000)), 1.0 / 2);
  ALGEBRAIC29 = 1.00000/(1.00000+exp((STATES[1] - 35.0000)/15.0000));
  ALGEBRAIC37 =  1.00000*ALGEBRAIC16*ALGEBRAIC29+80.0000;
  ALGEBRAIC4 = 1.00000/^(1.00000+exp((- 56.8600 - STATES[1])/9.03000), 2.00000);
  ALGEBRAIC17 = 1.00000/(1.00000+exp((- 60.0000 - STATES[1])/5.00000));
  ALGEBRAIC30 = 0.100000/(1.00000+exp((STATES[1]+35.0000)/5.00000))+0.100000/(1.00000+exp((STATES[1] - 50.0000)/200.000));
  ALGEBRAIC38 =  1.00000*ALGEBRAIC17*ALGEBRAIC30;
  ALGEBRAIC5 = 1.00000/^(1.00000+exp((STATES[1]+71.5500)/7.43000), 2.00000);
  ALGEBRAIC18 = (STATES[1]<- 40.0000 ?  0.0570000*exp(- (STATES[1]+80.0000)/6.80000) : 0.00000);
  ALGEBRAIC31 = (STATES[1]<- 40.0000 ?  2.70000*exp( 0.0790000*STATES[1])+ 310000.0*exp( 0.348500*STATES[1]) : 0.770000/( 0.130000*(1.00000+exp((STATES[1]+10.6600)/- 11.1000))));
  ALGEBRAIC39 = 1.00000/(ALGEBRAIC18+ALGEBRAIC31);
  ALGEBRAIC6 = 1.00000/^(1.00000+exp((STATES[1]+71.5500)/7.43000), 2.00000);
  ALGEBRAIC19 = (STATES[1]<- 40.0000 ? (( ( - 25428.0*exp( 0.244400*STATES[1]) -  6.94800e-06*exp( - 0.0439100*STATES[1]))*(STATES[1]+37.7800))/1.00000)/(1.00000+exp( 0.311000*(STATES[1]+79.2300))) : 0.00000);
  ALGEBRAIC32 = (STATES[1]<- 40.0000 ? ( 0.0242400*exp( - 0.0105200*STATES[1]))/(1.00000+exp( - 0.137800*(STATES[1]+40.1400))) : ( 0.600000*exp( 0.0570000*STATES[1]))/(1.00000+exp( - 0.100000*(STATES[1]+32.0000))));
  ALGEBRAIC40 = 1.00000/(ALGEBRAIC19+ALGEBRAIC32);
  ALGEBRAIC7 = 1.00000/(1.00000+exp((- 8.00000 - STATES[1])/7.50000));
  ALGEBRAIC20 = 1.40000/(1.00000+exp((- 35.0000 - STATES[1])/13.0000))+0.250000;
  ALGEBRAIC33 = 1.40000/(1.00000+exp((STATES[1]+5.00000)/5.00000));
  ALGEBRAIC41 = 1.00000/(1.00000+exp((50.0000 - STATES[1])/20.0000));
  ALGEBRAIC43 =  1.00000*ALGEBRAIC20*ALGEBRAIC33+ALGEBRAIC41;
  ALGEBRAIC56 = (( (( CONSTANTS[22]*CONSTANTS[11])/(CONSTANTS[11]+CONSTANTS[23]))*STATES[3])/(STATES[3]+CONSTANTS[24]))/(1.00000+ 0.124500*exp(( - 0.100000*STATES[1]*CONSTANTS[3])/( CONSTANTS[1]*CONSTANTS[2]))+ 0.0353000*exp(( - STATES[1]*CONSTANTS[3])/( CONSTANTS[1]*CONSTANTS[2])));
  ALGEBRAIC26 =  (( CONSTANTS[1]*CONSTANTS[2])/CONSTANTS[3])*log(CONSTANTS[12]/STATES[3]);
  ALGEBRAIC51 =  CONSTANTS[17]*^(STATES[8], 3.00000)*STATES[9]*STATES[10]*(STATES[1] - ALGEBRAIC26);
  ALGEBRAIC52 =  CONSTANTS[18]*(STATES[1] - ALGEBRAIC26);
  ALGEBRAIC57 = ( CONSTANTS[25]*( exp(( CONSTANTS[28]*STATES[1]*CONSTANTS[3])/( CONSTANTS[1]*CONSTANTS[2]))*^(STATES[3], 3.00000)*CONSTANTS[13] -  exp(( (CONSTANTS[28] - 1.00000)*STATES[1]*CONSTANTS[3])/( CONSTANTS[1]*CONSTANTS[2]))*^(CONSTANTS[12], 3.00000)*STATES[4]*CONSTANTS[27]))/( (^(CONSTANTS[30], 3.00000)+^(CONSTANTS[12], 3.00000))*(CONSTANTS[29]+CONSTANTS[13])*(1.00000+ CONSTANTS[26]*exp(( (CONSTANTS[28] - 1.00000)*STATES[1]*CONSTANTS[3])/( CONSTANTS[1]*CONSTANTS[2]))));
  ALGEBRAIC34 =  (( CONSTANTS[1]*CONSTANTS[2])/CONSTANTS[3])*log(CONSTANTS[11]/STATES[2]);
  ALGEBRAIC45 = 0.100000/(1.00000+exp( 0.0600000*((STATES[1] - ALGEBRAIC34) - 200.000)));
  ALGEBRAIC46 = ( 3.00000*exp( 0.000200000*((STATES[1] - ALGEBRAIC34)+100.000))+exp( 0.100000*((STATES[1] - ALGEBRAIC34) - 10.0000)))/(1.00000+exp( - 0.500000*(STATES[1] - ALGEBRAIC34)));
  ALGEBRAIC47 = ALGEBRAIC45/(ALGEBRAIC45+ALGEBRAIC46);
  ALGEBRAIC48 =  CONSTANTS[14]*ALGEBRAIC47* ^((CONSTANTS[11]/5.40000), 1.0 / 2)*(STATES[1] - ALGEBRAIC34);
  ALGEBRAIC55 =  CONSTANTS[21]*STATES[17]*STATES[16]*(STATES[1] - ALGEBRAIC34);
  ALGEBRAIC49 =  CONSTANTS[15]* ^((CONSTANTS[11]/5.40000), 1.0 / 2)*STATES[5]*STATES[6]*(STATES[1] - ALGEBRAIC34);
  ALGEBRAIC42 =  (( CONSTANTS[1]*CONSTANTS[2])/CONSTANTS[3])*log((CONSTANTS[11]+ CONSTANTS[10]*CONSTANTS[12])/(STATES[2]+ CONSTANTS[10]*STATES[3]));
  ALGEBRAIC50 =  CONSTANTS[16]*^(STATES[7], 2.00000)*(STATES[1] - ALGEBRAIC42);
  ALGEBRAIC53 = ( (( CONSTANTS[19]*STATES[12]*STATES[13]*STATES[14]*STATES[15]*4.00000*(STATES[1] - 15.0000)*^(CONSTANTS[3], 2.00000))/( CONSTANTS[1]*CONSTANTS[2]))*( 0.250000*STATES[11]*exp(( 2.00000*(STATES[1] - 15.0000)*CONSTANTS[3])/( CONSTANTS[1]*CONSTANTS[2])) - CONSTANTS[13]))/(exp(( 2.00000*(STATES[1] - 15.0000)*CONSTANTS[3])/( CONSTANTS[1]*CONSTANTS[2])) - 1.00000);
  ALGEBRAIC44 =  (( 0.500000*CONSTANTS[1]*CONSTANTS[2])/CONSTANTS[3])*log(CONSTANTS[13]/STATES[4]);
  ALGEBRAIC54 =  CONSTANTS[20]*(STATES[1] - ALGEBRAIC44);
  ALGEBRAIC59 = ( CONSTANTS[33]*(STATES[1] - ALGEBRAIC34))/(1.00000+exp((25.0000 - STATES[1])/5.98000));
  ALGEBRAIC58 = ( CONSTANTS[31]*STATES[4])/(STATES[4]+CONSTANTS[32]);
  ALGEBRAIC13 = 0.0# =stim (VOI -  floor(VOI/CONSTANTS[7])*CONSTANTS[7]>=CONSTANTS[6]&&VOI -  floor(VOI/CONSTANTS[7])*CONSTANTS[7]<=CONSTANTS[6]+CONSTANTS[8] ? CONSTANTS[9] : 0.00000);
  ALGEBRAIC63 = CONSTANTS[39] - (CONSTANTS[39] - CONSTANTS[40])/(1.00000+^(CONSTANTS[38]/STATES[18], 2.00000));
  ALGEBRAIC66 =  CONSTANTS[35]*ALGEBRAIC63;
  ALGEBRAIC60 = CONSTANTS[45]/(1.00000+^(CONSTANTS[43], 2.00000)/^(STATES[4], 2.00000));
  ALGEBRAIC61 =  CONSTANTS[44]*(STATES[18] - STATES[4]);
  ALGEBRAIC62 =  CONSTANTS[42]*(STATES[11] - STATES[4]);
  ALGEBRAIC64 = ( - ((ALGEBRAIC54+ALGEBRAIC58) -  2.00000*ALGEBRAIC57)*CONSTANTS[4])/( 2.00000*CONSTANTS[5]*CONSTANTS[3])+( (ALGEBRAIC61 - ALGEBRAIC60)*CONSTANTS[52])/CONSTANTS[5]+ALGEBRAIC62;
  ALGEBRAIC67 = 1.00000/(1.00000+( CONSTANTS[46]*CONSTANTS[47])/^(STATES[4]+CONSTANTS[47], 2.00000));
  ALGEBRAIC65 = CONSTANTS[34]/ALGEBRAIC63;
  ALGEBRAIC68 = ( ALGEBRAIC65*^(STATES[11], 2.00000)*STATES[19])/(CONSTANTS[36]+ ALGEBRAIC65*^(STATES[11], 2.00000));
  ALGEBRAIC69 =  CONSTANTS[41]*ALGEBRAIC68*(STATES[18] - STATES[11]);
  ALGEBRAIC70 = ALGEBRAIC60 - (ALGEBRAIC69+ALGEBRAIC61);
  ALGEBRAIC72 = 1.00000/(1.00000+( CONSTANTS[48]*CONSTANTS[49])/^(STATES[18]+CONSTANTS[49], 2.00000));
  ALGEBRAIC71 = (( - ALGEBRAIC53*CONSTANTS[4])/( 2.00000*CONSTANTS[53]*CONSTANTS[3])+( ALGEBRAIC69*CONSTANTS[52])/CONSTANTS[53]) - ( ALGEBRAIC62*CONSTANTS[5])/CONSTANTS[53];
  ALGEBRAIC73 = 1.00000/(1.00000+( CONSTANTS[50]*CONSTANTS[51])/^(STATES[11]+CONSTANTS[51], 2.00000));
end

end

using Adapt

struct TP2006MModel{ConstantsType} <: Thunderbolt.AbstractIonicModel
  CONSTANTS::ConstantsType
end

function get_V(du, u, model::TP2006MModel)
  return u[1]
end

function get_EK(du, u, model::TP2006MModel)
  CONSTANTS = model.CONSTANTS
  ALGEBRAIC34 =  (( CONSTANTS[1]*CONSTANTS[2])/CONSTANTS[3])*log(CONSTANTS[11]/u[2]);
  return ALGEBRAIC34
end

function get_ECa(du, u, model::TP2006MModel)
  CONSTANTS = model.CONSTANTS
  ALGEBRAIC44 =  (( 0.500000*CONSTANTS[1]*CONSTANTS[2])/CONSTANTS[3])*log(CONSTANTS[13]/u[4]);
  return ALGEBRAIC44
end

function get_ENa(du, u, model::TP2006MModel)
  CONSTANTS = model.CONSTANTS
  ALGEBRAIC26 =  (( CONSTANTS[1]*CONSTANTS[2])/CONSTANTS[3])*log(CONSTANTS[12]/u[3]);
  return ALGEBRAIC26
end

Adapt.@adapt_structure TP2006MModel

Thunderbolt.num_states(::TP2006MModel) = 19

function Thunderbolt.default_initial_state(ionic_model::TP2006MModel)
  _1  = zeros(53)
  _2 = zeros(19)
  u₀ = zeros(19)
  TP2006M.initConsts(_1, _2, u₀)
  return u₀
end

@inline function Thunderbolt._pointwise_step_inner_kernel!(cell_model::F, i::I, t::T, Δt::T, cache::C) where {F<:TP2006MModel, C <: Thunderbolt.ForwardEulerCellSolverCache, T <: Real, I <: Integer}
  u_local    = @view cache.uₙmat[i, :]
  du_local   = @view cache.dumat[i, :]
  x          = Thunderbolt.getcoordinate(cache, i)

  # TODO get Cₘ
  Thunderbolt.cell_rhs!(du_local, u_local, x, t, cell_model, i)

  @inbounds for j in 1:length(u_local)
      u_local[j] += Δt*du_local[j]
  end

  return true
end

@inline function Thunderbolt._pointwise_step_inner_kernel!(cell_model::F, i::I, t::T, Δt::T, cache::C) where {F<:TP2006MModel, C <: Thunderbolt.AdaptiveForwardEulerSubstepperCache, T <: Real, I <: Integer}
  u_local    = @view cache.uₙmat[i, :]
  du_local   = @view cache.dumat[i, :]
  x          = Thunderbolt.getcoordinate(cache, i)

  φₘidx = Thunderbolt.transmembranepotential_index(cell_model)

  # TODO get Cₘ
  Thunderbolt.cell_rhs!(du_local, u_local, x, t, cell_model, i)

  if abs(du_local[φₘidx]) < cache.reaction_threshold
      for j in 1:length(u_local)
          u_local[j] += Δt*du_local[j]
      end
  else
      Δtₛ = Δt/cache.substeps
      for j in 1:length(u_local)
          u_local[j] += Δtₛ*du_local[j]
      end

      for substep ∈ 2:cache.substeps
          tₛ = t + substep*Δtₛ
          #TODO Cₘ
          Thunderbolt.cell_rhs!(du_local, u_local, x, t, cell_model, i)

          for j in 1:length(u_local)
              u_local[j] += Δtₛ*du_local[j]
          end
      end
  end

  return true
end
function Thunderbolt.cell_rhs!(du::TD,u::TU,x::TX,t::TT,cell_parameters::TP) where {T,TD <: AbstractVector{T},TU,TX,TT,TP <: TP2006MModel}
  # HOTFIX projection onto solution boundary until rush larsen works
  u[8]  = clamp(u[8],  T(0.0), T(0.99))
  u[9]  = clamp(u[9],  T(0.0), T(0.99))
  u[10] = clamp(u[10], T(0.0), T(0.99))
  u[12] = clamp(u[12], T(0.0), T(0.99))
  u[13] = clamp(u[13], T(0.0), T(0.99))
  u[14] = clamp(u[14], T(0.0), T(0.99))
  u[16] = clamp(u[16], T(0.0), T(0.99))
  u[17] = clamp(u[17], T(0.0), T(0.99))

  # any(isnan.(s)) && error("nan1 $φₘ $s")
  TP2006M.computeRates(t, cell_parameters.CONSTANTS, du, u)
  # any(isnan.(du)) && error("nan2 $du $φₘ $s")
end

function TP2006MModel(::Type{TV}) where TV
  CONSTANTS = zeros(53)
  _1 = zeros(19)
  TP2006M.initConsts(CONSTANTS, _1, _1)
  return TP2006MModel(TV(CONSTANTS))
end

function Thunderbolt.state_symbol(::TP2006MModel, i::Int)
  syms = @SVector [
    :φₘ,
    :Kᵢ,
    :Naᵢ,
    :Caᵢ,
    :Xr₁,
    :Xr₂,
    :Xs,
    :m,
    :h,
    :j,
    :Caₛₛ,
    :d,
    :f,
    :f2,
    :fCaₛₛ,
    :s,
    :r,
    :Caₛᵣ,
    :Rp,
  ]
  return syms[i]
end

Thunderbolt.transmembranepotential_index(::TP2006MModel) = 1
