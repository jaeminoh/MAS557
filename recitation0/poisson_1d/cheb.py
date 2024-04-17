"""
NumPy version of the following MATLAB function:

function [D,x] = cheb(N)
% CHEB compute D = differentiation matrix, x = Chebyshev grid, from
% Trefethen

if N==0, D=0; x=1; return, end
x = cos(pi*(0:N)/N)';
c = [2; ones(N-1,1); 2].*(-1).^(0:N)';
X = repmat(x,1,N+1);
dX = X-X';
D = (c*(1./c)')./(dX+eye(N+1));
D = D-diag(sum(D'));
"""

import numpy as np


def cheb(N):
    if N == 0:
        return 0, 1

    x = np.cos(np.pi * np.linspace(0, 1, N + 1))
    c = np.array([2] + [1] * (N - 1) + [2]) * (-1 * np.ones(N + 1)) ** np.arange(N + 1)
    dX = x[:, None] - x
    D = c[:, None] * (1 / c) / (dX + np.eye(N + 1))
    D = D - np.diag(D.sum(1))
    return D, x
