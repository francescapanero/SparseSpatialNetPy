from utils.MCMC import *
from utils.CheckLikelParams import *
from utils.GraphSampler import *
from utils.TruncPois import *
from utils.SimpleGraph import *
from utils.AuxiliaryInference import *
import utils.Updates as up

# Set parameters for simulating data
alpha = 100
sigma = 0.2
tau = 1
beta = 0
size_x = 1
c = 1

K = 100  # number of layers, for layers sampler
T = 0.00001  # threshold simulations weights for GGP and doublepl (with w0 from GGP)
L = 10000  # tot number of nodes in exptiltBFRY

# prior parameters of alpha \sim gamma(a_alpha, b_alpha)
a_alpha = 200
b_alpha = 1

# prior for weights and type of sampler
prior = 'exptiltBFRY'  # can be 'exptiltBFRY', 'GGP', 'doublepl'
w_type = 'exptiltBFRY'  # for prior='doublepl' it is 'exptiltBFRY' or 'GGP' (we are talking about w0). I will change
# this is to make it clearer
sampler = 'layers'  # can be 'layers' or 'naive'

compute_distance = True  # you need distances if you are performing inference on w, n, u
reduce = False  # reduce graph G, locations x and weights w to active nodes. Usually not necessary.
check = True  # to check the loglikelihood of the parameters sigma, tau, alpha given w and u

# ----------------------
# SIMULATE DATA
# ----------------------

output = GraphSampler(prior, sampler, alpha, sigma, tau, beta, size_x, T=T, c=c, K=K, L=L, w_type=w_type)
w, x, G, size, deg = output[0:5]
if prior == 'doublepl':
    betaw, w0 = output[5:7]

# compute distances
if compute_distance is True and beta != 0:
    p_ij = space_distance(x, beta)
    n = up.update_n(w, G, size, p_ij)
if compute_distance is True and beta == 0:
    p_ij = np.ones((size, size))
    n = up.update_n(w, G, size, p_ij)

# compute auxiliary variables and quantities
if prior == 'doublepl':
    alpha_dpl = alpha * c ** (sigma * tau - sigma) / (tau * sigma)
    t = (sigma * size / alpha_dpl) ** (1 / sigma)
    u = tpoissrnd(t * w0)
if prior == 'GGP' or prior == 'exptiltBFRY':
    t = (sigma * size / alpha) ** (1 / sigma)
    u = tpoissrnd(t * w)

if reduce is True:
    G, isol = SimpleGraph(G)
    x_red = np.delete(x, isol)
    w_red = np.delete(w, isol)

if prior == 'GGP' or prior == 'exptiltBFRY' and check is True:
    check = check_sample_loglik(prior, sigma, tau, alpha, size_x, beta, w, u, t)
    sigma_max = check[0]
    tau_max = check[1]
    alpha_max = check[2]
    t_max = (sigma_max * size / alpha_max) ** (1 / sigma_max)
    log_post = log_post(prior, sigma, tau, alpha, t, w, u, a_alpha, b_alpha)
if prior == 'doublepl' and check is True:
    check = check_sample_loglik(prior, sigma, tau, alpha, size_x, beta, w, u, t, c=c, betaw=betaw, w0=w0)
    sigma_max = check[0]
    tau_max = check[1]
    alpha_max = check[2]
    t_max = (sigma_max * size / alpha_max) ** (1 / sigma_max)
    log_post = log_post(prior, sigma, tau, alpha, t, w, u, a_alpha, b_alpha, c=c, betaw=betaw, w0=w0)


# ------------------------
# ESTIMATE W
# ------------------------

# set number iterations and burn in
iter = 1000
nburn = int(0.25*iter)

# type of inference can be:
# - 'w_HMC' inference on w with HMC, all the rest fixed
# - 'w_gibbs' inference on w with gibbs, all the rest fixed
# - 'wnu' (the rest of the params are fixed)
# - 'params' (inference on alpha, sigma, tau, c and fix w, n, u)
# - single parameters (ex 'sigma', 'tau'...)
type = 'w_HMC'

# when type == 'wnu', w_inference is 'HMC' or 'gibbs'.
w_inference = 'HMC'

# set HMC parameters
epsilon = 0.0001
R = 5

# these are now in the setting of type = 'w_HMC' or 'w_gibbs'. For different types you should modify this, but I kept it
# like this to avoid confusion in the tests.
if prior == 'GGP' or prior == 'exptiltBFRY':
    output = MCMC(type, prior, iter, nburn, G, w_init=w, u=u, n=n, w_true=w,
                  tau=tau, sigma=sigma, alpha=alpha, p_ij=p_ij, R=R, epsilon=epsilon,
                  beta=beta, w_inference=w_inference, plot=True)
if prior == 'doublepl':
    output = MCMC(type, prior, iter, nburn, G, w_init=w, u=u, n=n, betaw=betaw, w0=w0, w_true=w,
                  tau=tau, sigma=sigma, alpha=alpha, c=c, p_ij=p_ij, R=R, epsilon=epsilon,
                  beta=beta, w_inference=w_inference, plot=True)


# # -----------------------------
# # ESTIMATE PARAMS
# # -----------------------------

# type of inference can be:
# - 'wnu' (the rest of the params are fixed)
# - 'params' (inference on alpha, sigma, tau, c and fix w, n, u)
# - there are also single inferences (for example sigma, or w_gibbs), but I am focusing on the other two now
type = 'params'

# set number iterations and burnin
iter = 200000
nburn = int(0.25*iter)

# set variances of the MH proposals for sigma, tau, alpha, c
sigma_tau = 0.01
sigma_sigma = 0.01
sigma_alpha = 0.01
sigma_c = 0.01

if prior == 'GGP' or prior == 'exptiltBFRY':
    # output = MCMC(type, prior, iter, nburn, G, beta=beta, w=w, u=u,
    #               sigma_init=sigma, tau_init=tau, alpha_init=alpha, c_init=c,
    #               sigma_sigma=sigma_sigma, sigma_tau=sigma_tau, sigma_alpha=sigma_alpha, a_alpha=a_alpha, b_alpha=b_alpha,
    #               plot=True, sigma_true=sigma, tau_true=tau, alpha_true=alpha, log_post=log_post)
    output = MCMC(type, prior, iter, nburn, G, beta=beta, w=w, u=u,
                  sigma_init=.5, tau_init=.1, alpha_init=2, c_init=1,
                  sigma_sigma=sigma_sigma, sigma_tau=sigma_tau, sigma_alpha=sigma_alpha, a_alpha=a_alpha, b_alpha=b_alpha,
                  plot=True, sigma_true=sigma, tau_true=tau, alpha_true=alpha, log_post=log_post)
    print('acceptance rate = ', (output[4][-1] - output[4][nburn]) / (iter - nburn))
if prior == 'doublepl':
    output = MCMC(type, prior, iter, nburn, G, beta=beta, w=w, u=u, w0=w0, betaw=betaw,
                  sigma_init=sigma, tau_init=tau, alpha_init=alpha, c_init=c,
                  sigma_sigma=sigma_sigma, sigma_tau=sigma_tau, sigma_alpha=sigma_alpha, a_alpha=a_alpha,
                  b_alpha=b_alpha, sigma_true=sigma, tau_true=tau, alpha_true=alpha, c_true=c, log_post=log_post,
                  plot=True)
    print('acceptance rate = ', (output[6][-1] - output[6][nburn]) / (iter - nburn))


#  what follows here is an attempt to compare the performance of the inference for different L, i.e. total number of
#  nodes. We are expecting to see that the converges becomes better as L increases.
#  For this part you need prior exptiltBFRY at the moment.
active_w = w[np.where(deg > 0)]
inactive_w = w[np.where(deg == 0)]
len_inactive = len(inactive_w)
inactive_ind = np.argsort(inactive_w)
len_active = len(active_w)

len_inactive0 = int((L - len_active) / 10)
L0 = len_active + len_inactive0
inactive_w0 = inactive_w[inactive_ind[range(len_inactive - len_inactive0, len_inactive)]]
w0 = np.concatenate((active_w, inactive_w0))
t0 = (sigma*L0/alpha)**(1/sigma)
u0 = tpoissrnd(t0*w0)
output0 = MCMC(type, prior, iter, iter-nburn, G, beta=beta, w=w0, u=u0, sigma_init=sigma, tau_init=tau, alpha_init=alpha, c_init=c,
                  sigma_sigma=sigma_sigma, sigma_tau=sigma_tau, sigma_alpha=sigma_alpha, a_alpha=a_alpha, b_alpha=b_alpha,
                  plot=False, sigma_true=sigma, tau_true=tau, alpha_true=alpha, log_post=log_post)

len_inactive1 = int((L - len_active) / 5)
L1 = len_active + len_inactive1
inactive_w1 = inactive_w[inactive_ind[range(len_inactive - len_inactive1, len_inactive)]]
w1 = np.concatenate((active_w, inactive_w1))
t1 = (sigma*L1/alpha)**(1/sigma)
u1 = tpoissrnd(t1*w1)
output1 = MCMC(type, prior, iter, iter-nburn, G, beta=beta, w=w1, u=u1, sigma_init=sigma, tau_init=tau, alpha_init=alpha, c_init=c,
                  sigma_sigma=sigma_sigma, sigma_tau=sigma_tau, sigma_alpha=sigma_alpha, a_alpha=a_alpha, b_alpha=b_alpha,
                  plot=False, sigma_true=sigma, tau_true=tau, alpha_true=alpha, log_post=log_post)

len_inactive3 = int((L - len_active) / 2)
L3 = len_active + len_inactive3
inactive_w3 = inactive_w[inactive_ind[range(len_inactive - len_inactive3, len_inactive)]]
w3 = np.concatenate((active_w, inactive_w3))
t3 = (sigma*L3/alpha)**(1/sigma)
u3 = tpoissrnd(t3*w3)
output3 = MCMC(type, prior, iter, iter-nburn, G, beta=beta, w=w3, u=u3, sigma_init=sigma, tau_init=tau, alpha_init=alpha, c_init=c,
                  sigma_sigma=sigma_sigma, sigma_tau=sigma_tau, sigma_alpha=sigma_alpha, a_alpha=a_alpha, b_alpha=b_alpha,
                  plot=False, sigma_true=sigma, tau_true=tau, alpha_true=alpha, log_post=log_post)

len_inactive4 = int((L - len_active) / 1.05)
L4 = len_active + len_inactive4
inactive_w4 = inactive_w[inactive_ind[range(len_inactive - len_inactive4, len_inactive)]]
w4 = np.concatenate((active_w, inactive_w4))
t4 = (sigma*L4/alpha)**(1/sigma)
u4 = tpoissrnd(t4*w4)
output4 = MCMC(type, prior, iter, iter-nburn, G, beta=beta, w=w4, u=u4, sigma_init=sigma, tau_init=tau, alpha_init=alpha, c_init=c,
                  sigma_sigma=sigma_sigma, sigma_tau=sigma_tau, sigma_alpha=sigma_alpha, a_alpha=a_alpha, b_alpha=b_alpha,
                  plot=False, sigma_true=sigma, tau_true=tau, alpha_true=alpha, log_post=log_post)

# this is with the orginal w, to compare
output2 = MCMC(type, prior, iter, iter-nburn, G, beta=beta, w=w, u=u, sigma_init=sigma, tau_init=tau, alpha_init=alpha, c_init=c,
               sigma_sigma=sigma_sigma, sigma_tau=sigma_tau, sigma_alpha=sigma_alpha, a_alpha=a_alpha, b_alpha=b_alpha,
               plot=False, sigma_true=sigma, tau_true=tau, alpha_true=alpha, log_post=log_post)

# acceptance rates
print('acceptance rate = ', output0[4][-1]/iter*100)
print('acceptance rate = ', output1[4][-1]/iter*100)
print('acceptance rate = ', output2[4][-1]/iter*100)
print('acceptance rate = ', output3[4][-1]/iter*100)
print('acceptance rate = ', output4[4][-1]/iter*100)

# diagnostic plots, comparing the different trajectories
sigma_est0 = output0[0]
sigma_fin0 = [sigma_est0[i] for i in range(iter-nburn,iter)]
sigma_est1 = output1[0]
sigma_fin1 = [sigma_est1[i] for i in range(iter-nburn,iter)]
sigma_est2 = output2[0]
sigma_fin2 = [sigma_est2[i] for i in range(iter-nburn,iter)]
sigma_est3 = output3[0]
sigma_fin3 = [sigma_est3[i] for i in range(iter-nburn,iter)]
sigma_est4 = output4[0]
sigma_fin4 = [sigma_est4[i] for i in range(iter-nburn,iter)]

plt.figure()
plt.plot(sigma_fin0, label='L/10', color='lightsteelblue')
plt.plot(sigma_fin1, label='L/5', color='cornflowerblue')
plt.plot(sigma_fin3, label='L/2', color='royalblue')
plt.plot(sigma_fin4, label='L/1.1', color='mediumblue')
plt.plot(sigma_fin2, label='L=original', color='black')
plt.axhline(y=sigma, label='true', color='r')
plt.xlabel('iter')
plt.ylabel('sigma')
plt.legend()
#plt.savefig('images/sigma_diff_L')

tau_est0 = output0[1]
tau_fin0 = [tau_est0[i] for i in range(iter-nburn,iter)]
tau_est1 = output1[1]
tau_fin1 = [tau_est1[i] for i in range(iter-nburn,iter)]
tau_est2 = output2[1]
tau_fin2 = [tau_est2[i] for i in range(iter-nburn,iter)]
tau_est3 = output3[1]
tau_fin3 = [tau_est3[i] for i in range(iter-nburn,iter)]
tau_est4 = output4[1]
tau_fin4 = [tau_est4[i] for i in range(iter-nburn,iter)]

plt.figure()
plt.plot(tau_fin0, label='L/10', color='lightsteelblue')
plt.plot(tau_fin1, label='L/5', color='cornflowerblue')
plt.plot(tau_fin3, label='L/2', color='royalblue')
plt.plot(tau_fin4, label='L/1.1', color='mediumblue')
plt.plot(tau_fin2, label='L=original', color='black')
plt.axhline(y=tau, label='true', color='r')
plt.xlabel('iter')
plt.ylabel('tau')
plt.legend()
#plt.savefig('images/tau_diff_L')

alpha_est0 = output0[2]
alpha_fin0 = [alpha_est0[i] for i in range(iter-nburn,iter)]
alpha_est1 = output1[2]
alpha_fin1 = [alpha_est1[i] for i in range(iter-nburn,iter)]
alpha_est2 = output2[2]
alpha_fin2 = [alpha_est2[i] for i in range(iter-nburn,iter)]
alpha_est3 = output3[2]
alpha_fin3 = [alpha_est3[i] for i in range(iter-nburn,iter)]
alpha_est4 = output4[2]
alpha_fin4 = [alpha_est4[i] for i in range(iter-nburn,iter)]

plt.figure()
plt.plot(alpha_fin0, label='L/10', color='lightsteelblue')
plt.plot(alpha_fin1, label='L/5', color='cornflowerblue')
plt.plot(alpha_fin3, label='L/2', color='royalblue')
plt.plot(alpha_fin4, label='L/1.1', color='mediumblue')
plt.plot(alpha_fin2, label='L=original', color='black')
plt.axhline(y=alpha, label='true', color='r')
plt.xlabel('iter')
plt.ylabel('alpha')
plt.legend()
#plt.savefig('images/alpha_diff_L')

# log_post0 = output0[5]
# log_post_fin0 = [log_post0[i] for i in range(iter-nburn,iter)]
# log_post1 = output1[5]
# log_post_fin1 = [log_post1[i] for i in range(iter-nburn,iter)]
# log_post2 = output2[5]
# log_post_fin2 = [log_post2[i] for i in range(iter-nburn,iter)]
# log_post3 = output3[5]
# log_post_fin3 = [log_post3[i] for i in range(iter-nburn,iter)]
# log_post4 = output4[5]
# log_post_fin4 = [log_post4[i] for i in range(iter-nburn,iter)]
# plt.figure()
# plt.plot(log_post0, label='L/10', color='royalblue')
# plt.plot(log_post1, label='L/5', color='mediumblue')
# plt.plot(log_post2, label='L=original', color='darkblue')
# plt.plot(log_post3, label='L/2', color='lightsteelblue')
# plt.plot(log_post4, label='L/1.2', color='cornflowerblue')
# plt.axhline(y=log_post, color='red')
# plt.xlabel('iter')
# plt.ylabel('log posterior')
# plt.legend()
# #plt.savefig('images/logpost_diff_L')
#
# plt.figure()
# plt.plot(log_post_fin0, label='L/10', color='royalblue')
# plt.plot(log_post_fin1, label='L/5', color='mediumblue')
# plt.plot(log_post_fin2, label='L=original', color='darkblue')
# plt.plot(log_post_fin3, label='L/2', color='lightsteelblue')
# plt.plot(log_post_fin4, label='L/1.2', color='cornflowerblue')
# plt.axhline(y=log_post, color='red')
# plt.xlabel('iter')
# plt.ylabel('log posterior final')
# plt.legend()
# # plt.savefig('images/logpostfin_diff_L')


# # -----------------------------
# # ESTIMATE SIGMA
# # -----------------------------
#
#
# # output = MCMC("sigma", "doublepl", iter, w=w, tau=tau, alpha=alpha, u=u, w0=w0, betaw=betaw, c=c)
# output = MCMC("sigma", "GGP", iter, w=w, tau=tau, alpha=alpha, u=u, sigma_tau=0.0001)
#
# sigma_est = output[0]
# sigma_fin = [sigma_est[i] for i in range(iter-nburn,iter)]
#
# emp0_CI_95 = scipy.stats.mstats.mquantiles(sigma_fin, prob=[0.025, 0.975])
# print(emp0_CI_95[0] <= sigma <= emp0_CI_95[1])
#
# plt.plot(sigma_fin, label='estimate', color='cornflowerblue', linewidth=0.7)
# # plt.axhline(y=np.mean(sigma_est), label='mean', color='b')
# plt.axhline(y=sigma, label='true', color='navy', linewidth=2)
# plt.axhline(y=emp0_CI_95[0], label='95% CI', color='royalblue', linestyle='--', linewidth=2)
# plt.axhline(y=emp0_CI_95[1], color='royalblue', linestyle='--', linewidth=2)
# # plt.axhline(y=sigma_maxloglik, label='maxloglik', color='g')
# plt.xlabel('iter')
# plt.ylabel('sigma')
# # plt.ylim((sigma-0.0001, sigma+0.0001))
# plt.legend()
#
# plt.savefig('images/sigma_300_04_5_1_1_02')
#
# accept = output[2]
# print(accept[-1])
#
#
# # -----------------------------
# # ESTIMATE TAU
# # -----------------------------
#
# output = MCMC("tau", "GGP", iter, w=w, sigma=sigma, alpha=alpha, u=u, sigma_tau=0.01)
# # output = MCMC("tau", "doublepl", iter, w=w, sigma=sigma, alpha=alpha, u=u, w0=w0, betaw=betaw, c=c, sigma_tau=0.5)
#
# tau_est = output[0]
# tau_fin = [tau_est[i] for i in range(iter-nburn,iter)]
#
# emp0_CI_95 = scipy.stats.mstats.mquantiles(tau_fin, prob=[0.025, 0.975])
# print(emp0_CI_95[0] <= tau <= emp0_CI_95[1])
#
#
# plt.plot(tau_fin, label='estimate', color='cornflowerblue', linewidth=0.7)
# # plt.axhline(y=np.mean(tau_est), label='mean', color='b')
# plt.axhline(y=tau, label='true', color='navy', linewidth=2)
# plt.axhline(y=emp0_CI_95[0], label='95% CI', color='royalblue', linestyle='--', linewidth=2)
# plt.axhline(y=emp0_CI_95[1], color='royalblue', linestyle='--', linewidth=2)
#
# # plt.axhline(y=tau_maxloglik, label='maxloglik', color='r')
# plt.xlabel('iter')
# plt.ylabel('tau')
# plt.legend()
#
# plt.savefig('images/tau_300_04_5_1_1_02')
#
# accept = output[1]
# print(accept[-1])
#
#
# # -----------------------------
# # ESTIMATE ALPHA
# # -----------------------------
#
# output = MCMC("alpha", "GGP", iter, w=w, sigma=sigma, tau=tau, u=u, a_alpha=a_alpha, b_alpha=b_alpha,
#               sigma_alpha=0.01)
# # output = MCMC("alpha", "doublepl", iter, w=w, sigma=sigma, tau=tau, u=u, a_alpha=a_alpha, b_alpha=b_alpha,
# #               sigma_alpha=0.1, w0=w0, betaw=betaw, c=c, alpha_init=alpha)
#
# alpha_est = output[0]
# alpha_fin = [alpha_est[i] for i in range(iter-nburn,iter)]
#
# emp0_CI_95 = scipy.stats.mstats.mquantiles(alpha_fin, prob=[0.025, 0.975])
# print(emp0_CI_95[0] <= alpha <= emp0_CI_95[1])
#
# plt.plot(alpha_fin, label='estimate', color='cornflowerblue', linewidth=0.7)
# # plt.axhline(y=np.mean(alpha_fin), label='mean', color='b')
# plt.axhline(y=alpha, label='true', color='navy', linewidth=2)
# plt.axhline(y=emp0_CI_95[0], label='95% CI', color='royalblue', linestyle='--', linewidth=2)
# plt.axhline(y=emp0_CI_95[1], color='royalblue', linestyle='--', linewidth=2)
# # plt.axhline(y=sigma_maxloglik, label='maxloglik', color='g')
# plt.xlabel('iter')
# plt.ylabel('alpha')
# # plt.ylim((alpha-0.3,alpha+0.3))
# plt.legend()
# plt.savefig('images/alpha_300_04_5_1_1_02')
#
# accept = output[2]
# print(accept[-1])
#
# # -----------------------------
# # ESTIMATE C
# # -----------------------------
#
# # output = MCMC("c", "doublepl", iter, w=s_true, sigma=sigma_true, alpha=alpha_true, tau=tau_true, u=u_true, w0=w0,
# #               betaw=betaw, sigma_c=sigma_tau)
#
