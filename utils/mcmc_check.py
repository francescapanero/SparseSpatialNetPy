# lags = np.arange(1, 100)
# plt.subplot(1, 4, 1)
# plt.plot(lags, [pm3.autocorr(np.array(output1[5][nburn:iter]), l) for l in lags])
# plt.subplot(1, 4, 2)
# plt.plot(lags, [pm3.autocorr(np.array(output2[5][nburn:iter]), l) for l in lags])
# plt.subplot(1, 4, 3)
# plt.plot(lags, [pm3.autocorr(np.array(output3[5][nburn:iter]), l) for l in lags])
# plt.subplot(1, 4, 4)
# plt.plot(lags, [pm3.autocorr(np.array(output4[5][nburn:iter]), l) for l in lags])
#
# n = iter - nburn + 1
# W = (np.array(output2[3][nburn:nburn+iter]).std() ** 2 + np.array(output3[3][nburn:nburn+iter]).std() ** 2 +
#      np.array(output4[3][nburn:nburn+iter]).std() ** 2) / 3
# mean1 = np.array(output2[3][nburn:nburn+iter]).mean()
# mean2 = np.array(output3[3][nburn:nburn+iter]).mean()
# mean3 = np.array(output4[3][nburn:nburn+iter]).mean()
# mean = (mean1 + mean2 + mean3) / 3
# B = n / 2 * ((mean1 - mean) ** 2 + (mean2 - mean) ** 2 + (mean3 - mean) ** 2)
# var_theta = (1 - 1/n) * W + 1 / n * B
# print("Gelmen-Rubin Diagnostic sigma: ", np.sqrt(var_theta/W))
# W = (np.array(output2[4][nburn:nburn+iter]).std() ** 2 + np.array(output3[4][nburn:nburn+iter]).std() ** 2 +
#      np.array(output4[4][nburn:nburn+iter]).std() ** 2) / 3
# mean1 = np.array(output2[4][nburn:nburn+iter]).mean()
# mean2 = np.array(output3[4][nburn:nburn+iter]).mean()
# mean3 = np.array(output4[4][nburn:nburn+iter]).mean()
# mean = (mean1 + mean2 + mean3) / 3
# B = n / 2 * ((mean1 - mean) ** 2 + (mean2 - mean) ** 2 + (mean3 - mean) ** 2)
# var_theta = (1 - 1/n) * W + 1 / n * B
# print("Gelmen-Rubin Diagnostic c: ", np.sqrt(var_theta/W))
# W = (np.array(output2[5][nburn:nburn+iter]).std() ** 2 + np.array(output3[5][nburn:nburn+iter]).std() ** 2 +
#      np.array(output4[5][nburn:nburn+iter]).std() ** 2) / 3
# mean1 = np.array(output2[5][nburn:nburn+iter]).mean()
# mean2 = np.array(output3[5][nburn:nburn+iter]).mean()
# mean3 = np.array(output4[5][nburn:nburn+iter]).mean()
# mean = (mean1 + mean2 + mean3) / 3
# B = n / 2 * ((mean1 - mean) ** 2 + (mean2 - mean) ** 2 + (mean3 - mean) ** 2)
# var_theta = (1 - 1/n) * W + 1 / n * B
# print("Gelmen-Rubin Diagnostic t: ", np.sqrt(var_theta/W))
#
# plt.figure()
# plt.acorr(output1[3], detrend=plt.mlab.detrend_mean, maxlags=100)
# plt.xlim(0, 100)
# plt.acorr(output1[4], detrend=plt.mlab.detrend_mean, maxlags=100)
# plt.xlim(0, 100)
# plt.acorr(output1[5], detrend=plt.mlab.detrend_mean, maxlags=100)
# plt.xlim(0, 100)
#
# plt.acorr(output2[3], detrend=plt.mlab.detrend_mean, maxlags=100)
# plt.xlim(0, 100)
# plt.acorr(output2[4], detrend=plt.mlab.detrend_mean, maxlags=100)
# plt.xlim(0, 100)
# plt.acorr(output2[5], detrend=plt.mlab.detrend_mean, maxlags=100)
# plt.xlim(0, 100)
#
# plt.acorr(output3[3], detrend=plt.mlab.detrend_mean, maxlags=100)
# plt.xlim(0, 100)
# plt.acorr(output3[4], detrend=plt.mlab.detrend_mean, maxlags=100)
# plt.xlim(0, 100)
# plt.acorr(output3[5], detrend=plt.mlab.detrend_mean, maxlags=100)
# plt.xlim(0, 100)
