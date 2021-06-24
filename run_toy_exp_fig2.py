#!/usr/bin/env python3
# Copyright 2021 Alexander Meulemans, Matilde Tristany Farinha, Javier Garcia Gordonez
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import numpy as np
import os
from networks.tpdi_networks import DFCNetwork
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FormatStrFormatter

mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'


out_dir = './logs/toy_experiments/fig2'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

np.random.seed(42)
torch.manual_seed(42)

n_in=5
n_hidden=[2]
n_out=2
nb_Q = 2000
nb_J_damped = 100
fit_on = 'total' # 'J', 'total', 'Q'

def rescale(matrix, scale=1.):
    matrix_magnitude = np.linalg.norm(matrix)
    return scale/matrix_magnitude * matrix

def all_positive_eig(A):
    lamb = np.linalg.eigvals(A)
    return sum(lamb.real<0) == 0

def all_negative_eig(A):
    lamb = np.linalg.eigvals(A)
    return sum(lamb.real>0) == 0

def generate_random_Q(jac):
    while True:
        permutation = np.random.randn(n_out,n_out)
        Q_rand = np.matmul(jac.T, permutation)
        if all_positive_eig(np.matmul(jac, Q_rand)):
            return rescale(Q_rand.flatten())

def compute_damped_jac(jac, damping):
    curv = np.matmul(jac, jac.T)
    return rescale(np.matmul(jac.T,
                np.linalg.inv(curv + damping * np.eye(curv.shape[0]))).flatten())

net = DFCNetwork(n_in=n_in, n_hidden=n_hidden, n_out=n_out,
                 activation='tanh', initialization='xavier_normal')

x = torch.randn((1, n_in))

net.forward(x)

jac = net.compute_full_jacobian(linear=True).squeeze(0).numpy()

Qs_vectorized = np.zeros((nb_Q, jac.size))

for i in range(nb_Q):
    Qs_vectorized[i,:] = generate_random_Q(jac)


J_damped_pinv = np.zeros((nb_J_damped, jac.size))
damping_values = np.logspace(-5, 2, num=nb_J_damped)

for i, damping in enumerate(damping_values):
    J_damped_pinv[i, :] = compute_damped_jac(jac, damping)

J_pinv = np.expand_dims(rescale(np.linalg.pinv(jac).flatten()), 0)
J_trans = np.expand_dims(rescale(jac.T.flatten()), 0)

QJ_combined = np.concatenate((Qs_vectorized, J_damped_pinv, J_pinv, J_trans), axis=0)

pca = PCA(n_components=2)
if fit_on == 'Q':
    pca.fit(Qs_vectorized)
elif fit_on == 'J':
    pca.fit(J_damped_pinv)
elif fit_on == 'total':
    pca.fit(QJ_combined)

Qs_pca = pca.transform(Qs_vectorized)
J_damped_pinv_pca = pca.transform(J_damped_pinv)
J_pinv_pca = pca.transform(J_pinv)
J_trans_pca = pca.transform(J_trans)

min_x_axis = np.min([np.min(Qs_pca[:, 0]), np.min(J_damped_pinv_pca[:,0]), np.min(J_pinv_pca[:,0]), np.min(J_trans_pca[:,0])])
min_y_axis = np.min([np.min(Qs_pca[:, 1]), np.min(J_damped_pinv_pca[:,1]), np.min(J_pinv_pca[:,1]), np.min(J_trans_pca[:,1])])
max_x_axis = np.max([np.max(Qs_pca[:, 0]), np.max(J_damped_pinv_pca[:,0]), np.max(J_pinv_pca[:,0]), np.max(J_trans_pca[:,0])])
max_y_axis = np.max([np.max(Qs_pca[:, 1]), np.max(J_damped_pinv_pca[:,1]), np.max(J_pinv_pca[:,1]), np.max(J_trans_pca[:,1])])

min_axis = np.min([min_x_axis,min_y_axis])
max_axis = np.max([max_x_axis,max_y_axis])

fig, ax = plt.subplots(figsize=(17,15))
sns.set(font_scale=4)

a = 0.9

my_cmap = plt.cm.Blues(np.arange(plt.cm.Blues.N))
my_cmap[:,0:3] *= a 
my_cmap = ListedColormap(my_cmap)
cmap = plt.cm.get_cmap(my_cmap)
first_color = cmap(0.1)
ax.set_facecolor(first_color)

kdeplot = sns.kdeplot(Qs_pca[:, 0], Qs_pca[:, 1], cmap=my_cmap, shade=True, cbar=True, levels=5, cbar_kws={'label': 'probability density'})
sns.scatterplot(Qs_pca[:,0], Qs_pca[:,1], color='midnightblue', edgecolor='midnightblue', marker='.', s=100, alpha=0.3)
sns.scatterplot(J_trans_pca[:,0], J_trans_pca[:,1], color='midnightblue', marker='o', s=700, alpha=0.3, label=r'$Q$')
sns.scatterplot(J_pinv_pca[:,0], J_pinv_pca[:,1], color='orange', marker='X', s=700, alpha=1, label=r'$J^{T}(JJ^{T}+\gamma I)^{-1}$')
sns.scatterplot(J_damped_pinv_pca[:,0], J_damped_pinv_pca[:,1], color='orange', marker='X', s=200, alpha=1)
sns.scatterplot(J_pinv_pca[:,0], J_pinv_pca[:,1], color='green', marker='X', s=700, alpha=1, label=r'$J^{\dagger}$')
sns.scatterplot(J_trans_pca[:,0], J_trans_pca[:,1], color='red', marker='X', s=700, alpha=1, label=r'$J^{T}$')
ax.set_xlim(min_axis-0.05,max_axis+0.05)
ax.set_ylim(min_y_axis-0.05,max_y_axis+0.05)
ax.set_xticks([-0.5, 0, 0.5])
ax.set_yticks([-0.5, 0, 0.5])
plt.yticks(rotation=90)
ax.set_xticklabels(["-0.5", "0", "0.5"])
ax.set_yticklabels(["-0.5", "0", "0.5"])

plt.xlabel('PC1', fontsize=50, labelpad=1)
plt.ylabel('PC2', fontsize=50, labelpad=1)
plt.tick_params(labelsize=50)
plt.locator_params(nbins=3)
sns.despine()
plt.legend(bbox_to_anchor=(0.5, 1.2), fontsize=50, loc="upper center", frameon=False, ncol=2, handletextpad=0.01, labelspacing=0, framealpha=0.7)

if fit_on == 'Q':
    plt.savefig(os.path.join(out_dir + str("pca_fitted_Q") + ".png"))
elif fit_on == 'J':
    plt.savefig(os.path.join(out_dir + str("pca_fitted_J") + ".png"))
elif fit_on == 'total':
    plt.savefig(os.path.join(out_dir + str("pca_fitted_total") + ".png"))