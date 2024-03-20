import numpy as np
def SelfSVD(A, tol=1e-8):
    Lambda, V = np.linalg.eigh(A.T @ A)
    Sigma = np.sqrt(Lambda)
    idx = np.argsort(Sigma)[::-1]
    Sigma = Sigma[idx]
    V = V[:, idx]
    U = np.zeros((A.shape[0], len(Sigma)))
    for i in range(len(Sigma)):
        U[:, i] = A @ V[:, i] / Sigma[i]

    valid_indices = np.where(Sigma >= tol)[0]
    U = U[:, valid_indices]
    Sigma = Sigma[valid_indices]
    V = V[:, valid_indices]

    return U, Sigma, V

A = np.random.rand(4, 3)
tol = 1e-8

U_custom, Sigma_custom, V_custom = SelfSVD(A, tol)

U_builtin, Sigma_builtin, V_builtin = np.linalg.svd(A, full_matrices=False)
print('Custom SVD:')
print('U:')
print(U_custom)
print('Sigma:')
print(Sigma_custom)
print('V:')
print(V_custom)

print('Built-in svd:')
print('U:')
print(U_builtin)
print('Sigma:')
print(Sigma_builtin)
print('V:')
print(V_builtin)
