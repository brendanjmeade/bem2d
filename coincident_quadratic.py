import numpy as np


# g0
g0_quadratic_phi_1_node_1 = -5/144*a*np.log(25/9*a**2)/(np.pi - np.pi*nu) - 17/288*a*np.log(1/9*a**2)/(np.pi - np.pi*nu) + 1/12*a/(np.pi - np.pi*nu)
 
g0_quadratic_phi_2_node_1 = -25/288*a*np.log(25/9*a**2)/(np.pi - np.pi*nu) + 7/288*a*np.log(1/9*a**2)/(np.pi - np.pi*nu) + 1/12*a/(np.pi - np.pi*nu)
 
g0_quadratic_phi_3_node_1 = -25/288*a*np.log(25/9*a**2)/(np.pi - np.pi*nu) - 1/144*a*np.log(1/9*a**2)/(np.pi - np.pi*nu) - 1/6*a/(np.pi - np.pi*nu)
 
g0_quadratic_phi_1_node_2 = -3/16*a*np.log(a)/(np.pi - np.pi*nu) - 1/8*a/(np.pi - np.pi*nu)
 
g0_quadratic_phi_2_node_2 = -1/8*a*np.log(a)/(np.pi - np.pi*nu) + 1/4*a/(np.pi - np.pi*nu)
 
g0_quadratic_phi_3_node_2 = -3/16*a*np.log(a)/(np.pi - np.pi*nu) - 1/8*a/(np.pi - np.pi*nu)
 
g0_quadratic_phi_1_node_3 = -25/288*a*np.log(25/9*a**2)/(np.pi - np.pi*nu) - 1/144*a*np.log(1/9*a**2)/(np.pi - np.pi*nu) - 1/6*a/(np.pi - np.pi*nu)
 
g0_quadratic_phi_2_node_3 = -25/288*a*np.log(25/9*a**2)/(np.pi - np.pi*nu) + 7/288*a*np.log(1/9*a**2)/(np.pi - np.pi*nu) + 1/12*a/(np.pi - np.pi*nu)
 
g0_quadratic_phi_3_node_3 = -5/144*a*np.log(25/9*a**2)/(np.pi - np.pi*nu) - 17/288*a*np.log(1/9*a**2)/(np.pi - np.pi*nu) + 1/12*a/(np.pi - np.pi*nu)


# g1
1/4/(nu - 1)
 
0
 
0
 
0
 
1/4/(nu - 1)
 
0
 
0
 
0
 
1/4/(nu - 1)


# g2
1/8*np.log(25/9*a**2)/(np.pi - np.pi*nu) - 1/8*np.log(1/9*a**2)/(np.pi - np.pi*nu) - 3/4/(np.pi - np.pi*nu)
 
3/4/(np.pi - np.pi*nu)
 
0
 
-3/8/(np.pi - np.pi*nu)
 
0
 
3/8/(np.pi - np.pi*nu)
 
0
 
-3/4/(np.pi - np.pi*nu)
 
-1/8*np.log(25/9*a**2)/(np.pi - np.pi*nu) + 1/8*np.log(1/9*a**2)/(np.pi - np.pi*nu) + 3/4/(np.pi - np.pi*nu)


# g3
-9/16/(a*nu - a)
 
3/4/(a*nu - a)
 
-3/16/(a*nu - a)
 
-3/16/(a*nu - a)
 
0
 
3/16/(a*nu - a)
 
3/16/(a*nu - a)
 
-3/4/(a*nu - a)
 
9/16/(a*nu - a)


# g4
9/32*np.log(25/9*a**2)/(np.pi*a*nu - np.pi*a) - 9/32*np.log(1/9*a**2)/(np.pi*a*nu - np.pi*a) + 27/80/(np.pi*a*nu - np.pi*a)
 
-3/8*np.log(25/9*a**2)/(np.pi*a*nu - np.pi*a) + 3/8*np.log(1/9*a**2)/(np.pi*a*nu - np.pi*a) + 9/8/(np.pi*a*nu - np.pi*a)
 
3/32*np.log(25/9*a**2)/(np.pi*a*nu - np.pi*a) - 3/32*np.log(1/9*a**2)/(np.pi*a*nu - np.pi*a) - 9/16/(np.pi*a*nu - np.pi*a)
 
-9/16/(np.pi*a*nu - np.pi*a)
 
13/8/(np.pi*a*nu - np.pi*a)
 
-9/16/(np.pi*a*nu - np.pi*a)
 
3/32*np.log(25/9*a**2)/(np.pi*a*nu - np.pi*a) - 3/32*np.log(1/9*a**2)/(np.pi*a*nu - np.pi*a) - 9/16/(np.pi*a*nu - np.pi*a)
 
-3/8*np.log(25/9*a**2)/(np.pi*a*nu - np.pi*a) + 3/8*np.log(1/9*a**2)/(np.pi*a*nu - np.pi*a) + 9/8/(np.pi*a*nu - np.pi*a)
 
9/32*np.log(25/9*a**2)/(np.pi*a*nu - np.pi*a) - 9/32*np.log(1/9*a**2)/(np.pi*a*nu - np.pi*a) + 27/80/(np.pi*a*nu - np.pi*a)


# g5
9/32*np.log(25/9*a**2)/(np.pi*a**2*nu - np.pi*a**2) - 9/32*np.log(1/9*a**2)/(np.pi*a**2*nu - np.pi*a**2) + 621/100/(np.pi*a**2*nu - np.pi*a**2)
 
-9/16*np.log(25/9*a**2)/(np.pi*a**2*nu - np.pi*a**2) + 9/16*np.log(1/9*a**2)/(np.pi*a**2*nu - np.pi*a**2) - 27/5/(np.pi*a**2*nu - np.pi*a**2)
 
9/32*np.log(25/9*a**2)/(np.pi*a**2*nu - np.pi*a**2) - 9/32*np.log(1/9*a**2)/(np.pi*a**2*nu - np.pi*a**2) + 27/20/(np.pi*a**2*nu - np.pi*a**2)
 
3/4/(np.pi*a**2*nu - np.pi*a**2)
 
0
 
-3/4/(np.pi*a**2*nu - np.pi*a**2)
 
-9/32*np.log(25/9*a**2)/(np.pi*a**2*nu - np.pi*a**2) + 9/32*np.log(1/9*a**2)/(np.pi*a**2*nu - np.pi*a**2) - 27/20/(np.pi*a**2*nu - np.pi*a**2)
 
9/16*np.log(25/9*a**2)/(np.pi*a**2*nu - np.pi*a**2) - 9/16*np.log(1/9*a**2)/(np.pi*a**2*nu - np.pi*a**2) + 27/5/(np.pi*a**2*nu - np.pi*a**2)
 
-9/32*np.log(25/9*a**2)/(np.pi*a**2*nu - np.pi*a**2) + 9/32*np.log(1/9*a**2)/(np.pi*a**2*nu - np.pi*a**2) - 621/100/(np.pi*a**2*nu - np.pi*a**2)


# g6
-9/16/(a**2*nu - a**2)
 
9/8/(a**2*nu - a**2)
 
-9/16/(a**2*nu - a**2)
 
-9/16/(a**2*nu - a**2)
 
9/8/(a**2*nu - a**2)
 
-9/16/(a**2*nu - a**2)
 
-9/16/(a**2*nu - a**2)
 
9/8/(a**2*nu - a**2)
 
-9/16/(a**2*nu - a**2)

