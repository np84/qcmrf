import sys
import numpy as np
from scipy.linalg import expm
np.set_printoptions(threshold=sys.maxsize,linewidth=1024)

import itertools
from colored import fg, bg, attr

from qiskit.opflow import I, X, Z, MatrixOp
from qiskit.compiler import transpile
from qiskit.quantum_info.states.statevector import Statevector
from qiskit.quantum_info import partial_trace, DensityMatrix

####################################### input structure

C = [[0,1],[1,2]] # clique structure, 2-var chain
#C = [[0,1],[1,2],[2,3],[0,3]] # clique structure, 4-var circle

n = len(np.unique(np.array(C).flatten())) # number of (qu)bits
d = 0
for c in C:
	m = len(c)
	d = d + (2**m)
	
dim = 2**n

####################################### utility funcs

h = MatrixOp((1.0/np.sqrt(2.0)) * np.array([[1,1],[1,-1]])) # Hadamard gate

def grouped(iterable, n):
	return zip(*[iter(iterable)]*n)
	
def merge(L):
	R = []
	for (M0,M1) in grouped(L,2):
		R.append((((I+Z)/2)^M0) + (((I-Z)/2)^M1))
	return R
	
def merge_all(L):
	R = L
	while len(R)>1:
		R = merge(R)
	return R[0]
	
def format_head(h,t):
	return fg('#179c7d')+attr('bold')+h+attr('reset')+': '+fg('white')+attr('bold')+t+attr('reset')
	
def format_val(val):
	if type(val) is int:
		return fg('white')+attr('bold')+str(val)+attr('reset')
	else:
		return fg('white')+attr('bold')+np.format_float_scientific(val, precision=4)+attr('reset')
		
def format_math(math):
	brackets  = ['(',')']
	operators = ['+','-','*','/','_','^']
	mcol = '#1f82c0'
	bcol = 'white'
	ocol = '#eb6a0a'
	for b in brackets:
		math = math.replace(b,fg(bcol)+b+fg(mcol))
	for o in operators:
		math = math.replace(o,fg(ocol)+o+fg(mcol))
	return fg(mcol)+math+attr('reset')

####################################### model funcs

def genPhi(c,y):
	result = 1
	
	plus  = [v for i,v in enumerate(c) if not y[i]] # 0
	minus = [v for i,v in enumerate(c) if     y[i]] # 1

	for i in range(n):
		f = I
		if i in minus:
			f = (I-Z)/2
		elif i in plus:
			f = (I+Z)/2

		result = result^f
		
	return result

def genHamiltonian():
	L = []
	H = 0
	i = 0
	for l,c in enumerate(C):
		Ly = []
		for r,y in enumerate(list(itertools.product([0, 1], repeat=len(c)))):
			Phi = genPhi(c,y)
			theta = -0.1*(i+1) #np.random.uniform() # we need a negative MRF
			H += Phi * -theta
			Ly.append((theta,Phi)) # list of all factors that belong to same clique
			i = i + 1
		L.append(Ly) # list of factor-lists, one factor-list per clique

	return H,L

def genPhaseFactors(ey):
	t = np.sqrt((-1-ey)/(-1+ey))
	
	phi1 = np.angle( ( (1+t*1j) + ey*(1-t*1j) ) / 2 )
	phi2 = np.real(np.angle( np.sqrt(0.5) * np.sqrt(1 - ey) * (t + 1j) ))
	
	return np.array([phi1, phi2])
	
def genUphi(U,phi):
	M  = Z^(I^n) # (2PI-(I^n))
	F1 = (-phi[0] * M).exp_i()
	F2 = (-phi[1] * M).exp_i()
	return F1 @ (~U) @ F2 @ U

# works only if Eigenvalues of A are bounded by 1	
def uniEmbedding(A):
	return (X^((I^n)-A)) + (Z^A)

# returns unitary if A is unitary	
def conjugateBlocks(A):
	return (((I+Z)/2)^A) + (((I-Z)/2)^(~A))

# works only if number of cliques is a power of two
def expH_block_factors_from_list(beta, L0, lnZ=0):
	R = []
	for Ly in L0:
		L = [genUphi(uniEmbedding(Phi0), genPhaseFactors(np.exp(beta*(w0 - (lnZ/len(C)))))) for w0,Phi0 in Ly]
		R.append(merge_all(L))
	M = merge_all(R)
	O = h^(I^int(n+1+np.log2(d)))
	return O @ conjugateBlocks(M) @ (~O)
	
def expH_from_list(beta, L0, lnZ=0, forceReal=False):
	RESULT = I^(n+1)
	for L in L0:
		for (w0,Phi0) in L:
			U = genUphi(uniEmbedding(Phi0), genPhaseFactors(np.exp(beta*(w0 - (lnZ/len(C))))))
			if forceReal:
				RESULT = ((U+(~U))/2) @ RESULT # non-unitary extraction of real part
			else:
				RESULT = U @ RESULT # unitary extraction of real part
	return RESULT
	
#######################################

H,L  = genHamiltonian() # L is list of factors
beta = 1

R0   = expm(-beta*H.to_matrix()) # exp(-βH) via numpy for debugging

#######################################

print("NODES ("+format_math('n')+"):",n," PARAMETERS ("+format_math('d')+"):",d)

#######################################
print(format_head("CHECK 1","expH_from_list(.., forceReal=True);"))
print("  * This shows that Eq.(3) from the draft works.")

R1   = expH_from_list(beta, L, forceReal=True)
r1   = R1.to_matrix()

print("  * "+format_math('l2')+"-Error between",format_math('exp(-βH)'),"and upper left")
print("    block of",format_math('PROD_j REAL(U^j_(γ^y))')+":", format_val(np.linalg.norm(R0 - r1[:dim,:dim], ord=2)))

#######################################
print()
print(format_head("CHECK 2","expH_from_list(.., forceReal=False);"))
print("  * This shows that Eq. (3) from the draft does *NOT*")
print("    work when we do not extract real parts.")

R2   = expH_from_list(beta, L, forceReal=False)
r2   = R2.to_matrix()

print("  * "+format_math('l2')+"-Error between",format_math('exp(-βH)'),"and upper left")
print("    block of",format_math('PROD_j U^j_(γ^y)')+":", format_val(np.linalg.norm(R0 - r2[:dim,:dim], ord=2)))

#p = np.diag(R0)
#q = np.diag(r2)[:2**n]
#
#print(p)
#print(q)

#######################################
print()
print(format_head("CHECK 3","expH_block_factors_from_list(..);"))
print("  * This shows that the new block factorization with ("+format_math('1+log_2(d)'))
print("    extra ancillas) has correct values of each factor on the diagonal.")

R3   = expH_block_factors_from_list(beta, L)
r3   = R3.to_matrix()

r = np.eye(2*dim) # 2*dim = 2**(n+1)
for i in range(d):
	a = (i)   * (2*dim)
	b = (i+1) * (2*dim)
	B = r3[a:b,a:b]
	r = B @ r

print("  * "+format_math('l2')+"-Error between",format_math('exp(-βH)'),"and the product over")
print("    first",format_math('d'),"diagonal blocks of block factorization:", format_val(np.linalg.norm(R0 - r[:dim,:dim], ord=2)))

#######################################
print()
print(format_head("CHECK 4","uniEmbedding(genPhi(..));"))
print("  * This shows that the transpiled circuit for a rather simple")
print("    unitary embedding of a",format_math('Φ')+"-matrix requires a \"large\" circuit.")

Q = uniEmbedding(genPhi(C[0],[1,0]))
UU = transpile(Q.to_circuit(), basis_gates=['cx','id','rz','sx','x'], optimization_level=3)

print("  * Number of gates:", format_val(len(UU)))
print("  * Depth:", format_val(UU.depth()))

#######################################
print()
print(format_head("CHECK 5","genUphi(uniEmbedding(genPhi(..)), genPhaseFactors(..));"))
print("  * This shows that the transpiled circuit for any",format_math('U^j_(γ^y)'))
print("    unitary operator requires an \"even larger\" circuit.")

Q = genUphi(uniEmbedding(genPhi(C[0],[1,0])), genPhaseFactors(np.exp(-0.5)))
UU = transpile(Q.to_circuit(), basis_gates=['cx','id','rz','sx','x'], optimization_level=3)

print("  * Number of gates:", format_val(len(UU)))
print("  * Depth:", format_val(UU.depth()))

