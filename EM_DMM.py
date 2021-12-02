import numpy as np
import dirichlet
from scipy.misc import derivative #derivative(func,x,dx=1e-6,n=1).n is the order of derivative
from scipy.special import gamma #gamma(alpha)
from scipy.special import gammaln, polygamma, psi
import scipy.stats as stats
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import os
import sys
import matplotlib.tri as tri
from draw_dirichlet import Dirichlet, draw_pdf_contours, plot_points
from sklearn.metrics import roc_auc_score

# For drawing dirichlet distribution on simplex triangle.
_corners = np.array([[0, 0], [1, 0], [0.5, 0.75**0.5]])
_AREA = 0.5 * 1 * 0.75**0.5
_triangle = tri.Triangulation(_corners[:, 0], _corners[:, 1])

# For each corner of the triangle, the pair of other corners
_pairs = [_corners[np.roll(range(3), -i)[1:]] for i in range(3)]
# The area of the triangle formed by point xy and another pair or points
tri_area = lambda xy, pair: 0.5 * np.linalg.norm(np.cross(*(pair - xy)))
#Generate artificial samples
sample_size = 5000
a_origin = [3, 3, 3]
tau_origin = 20
e = np.eye(3, dtype=int)
pi_origin = [0.25,0.5,0.25]
a0 = a_origin + tau_origin*e[0] 
a1 = a_origin + tau_origin*e[1] 
a2 = a_origin + tau_origin*e[2] 
np.random.seed(12345)
x0 = np.random.dirichlet(a0, size=sample_size)
y0 = np.array([0]*sample_size)
x1 = np.random.dirichlet(a1, size=2*sample_size)
y1 = np.array([1]*(2*sample_size))
x2 = np.random.dirichlet(a2, size=sample_size)
y2 = np.array([2]*sample_size)

x_pretrain = np.concatenate((x0[0:100],x1[0:100],x2[0:100]),axis=0)
y_pretrain = np.concatenate((y0[0:100],y1[0:100],y2[0:100]))

x_test = np.random.dirichlet([30]*3, size=500)
y_test = np.array([3]*500)

x_train = np.concatenate((x0[100:4500],x1[100:9000],x2[100:4500]),axis=0)
y_train = np.concatenate((y0[100:4500],y1[100:9000],y2[100:4500]))
index = np.random.permutation(np.arange(x_train.shape[0]))
x = x_train[index]
y = y_train[index]

x_test = np.concatenate((x0[4500::],x1[9000::],x2[4500::],x_test),axis=0)
y_test = np.concatenate((y0[4500::],y1[9000::],y2[4500::],y_test))


alpha0 = dirichlet.mle(x0[0:100])
alpha1 = dirichlet.mle(x1[0:100])
alpha2 = dirichlet.mle(x2[0:100])

print(alpha0,alpha1,alpha2)

def init_a(X,y):
    """Initial guess for Dirichlet alpha parameters given data X and label y

    Parameters
    ----------
    X : (N, K) shape array
        ``N`` is the number of observations, ``K`` is the number of
        parameters for the Dirichlet distribution.
    y : (1,N) shape array
    Returns
    -------
    alpha : (1, K) shape array
    tau : scalar
    pi : (1, K) shape array. 
       Return guess for parameters of Flexible Dirichlet mixture distribution."""
    Alpha = []
    Tau = []
    Pi = []
    for i in range(X.shape[1]):
        alpha = dirichlet.mle(X[y==i])
        pi = (np.sum(y==i)+0.0)/len(y)
        Alpha.append(alpha)        
        Pi.append(pi)
        if i != 0:
            tau = alpha[i]-Alpha[i-1][i]
            Tau.append(tau)       
    alpha_matrix = np.array(Alpha)
    alpha = (alpha_matrix.sum(axis = 0) - np.diag(alpha_matrix))/(X.shape[1]-1)
    tau = np.mean(np.diag(alpha_matrix) - alpha)
    return alpha, tau, Pi

iter_num = 10
component_num = 3
a_0,tau_0, pi_0 = init_a(x_pretrain,y_pretrain)  
print(a_0, tau_0, pi_0)


def post_prob(x,a,pi,tau,dim): 
    sum_p = np.float64(0.)
    e_dim =np.zeros((a.shape))
    for i in range(component_num):
        e_j = np.copy(e_dim)
        e_j[i] = 1
        sum_p += np.exp(np.log(pi[i]) + gammaln(a[i]) + tau*np.log(x[i]) - gammaln(a[i]+tau))
    e_dim[dim] =1
    if sum_p != 0.0:
        post_p = np.exp(np.log(pi[dim])+gammaln(a[dim])+tau*np.log(x[dim]) - gammaln(a[dim]+tau) - np.log(sum_p))
    else:
        post_p = np.exp(np.log(pi[dim])+gammaln(a[dim])+tau*np.log(x[dim]) - gammaln(a[dim]+tau) + 400)
    return post_p

def log_flexible_dir(x,a):
    sum_log = np.float64(0.)
    for i in range(component_num):
        sum_log += (a[i]-1)*np.log(x[i])-gammaln(a[i])
    log_flex_dir =  sum_log + gammaln(np.sum(a))
    return log_flex_dir
    
#EM step
def EM_FDMM(x,y,a,pi,tau,x_ori,y_ori,a_origin,tau_origin,pi_origin):
    a_update = np.zeros((a.shape))
    log_likelihood_old = np.float64(-0.1)
    log_likelihood = np.float64(0.0)
    e_dim =np.zeros((a.shape))
    t = 0
    f = open('./dirichlet-master/result/train_result_params.txt', 'a+')
    f.write("alpha, Tau, Pi \n")
    f.write("{},{},{} \n".format(a_origin, tau_origin, pi_origin))

    while log_likelihood > log_likelihood_old:
        # plt.cla()
        t+=1
        log_likelihood_old = log_likelihood
        g = np.zeros((component_num,1))
        H = np.ones((component_num,component_num))
        H_trace = 0.
        H_other = 0.
        g_tau = 0.
        H_tau = 0.
        log_likelihood = np.float64(0.0)
        sum_post_p =np.array([0.0]*component_num)
        y_logit = np.zeros((x.shape[0],component_num))
        y_prediction = np.zeros((x.shape[0],1))
        for h in range(component_num):
            for j in range(component_num):
                e_j = np.copy(e_dim)
                e_j[j] = 1
                for i in range(x.shape[0]):
                    sample_likelihood = post_prob(x[i],a,pi,tau,dim=j)
                    g[h,0] += sample_likelihood*(psi(np.sum(a)+tau)-psi(a[h]+tau*e_j[h])+np.log(x[i][h]))
                    H_trace += sample_likelihood*(polygamma(1,np.sum(a)+tau)-polygamma(1,a[h]+tau*e_j[h]))
                    if h==0:
                        H_other += sample_likelihood*polygamma(1,np.sum(a)+tau)
                        y_logit[i][j] = sample_likelihood
                        a_new = np.copy(a)
                        a_new[j] = a_new[j] + tau
                        log_likelihood += sample_likelihood*(np.log(pi[j])+log_flexible_dir(x[i],a_new))
                        sum_post_p[j] += sample_likelihood
                        g_tau += sample_likelihood*(psi(np.sum(a)+tau)-psi(a[j]+tau)+np.log(x[i][j]))
                        H_tau += sample_likelihood*(polygamma(1,np.sum(a)+tau)-polygamma(1,a[j]+tau))
                if h==0:
                    sum_post_p[j] = sum_post_p[j]/x.shape[0]                   
            if h==0:
                H = H*H_other
            H[h][h] = H_trace

        sum_post_p[component_num-1] = 1-np.sum(sum_post_p[0:component_num-1])
        a_update = a.reshape(component_num,1) - np.dot(np.linalg.inv(H),g) 

        if log_likelihood>log_likelihood_old:
            a = a_update.reshape(1,component_num).squeeze()
            pi = sum_post_p
            tau = tau - tau/H_tau

        ff = plt.figure(figsize=(16, 12))
        alphas = [[a[0]+tau,a[1],a[2]],
                [a[0],a[1]+tau,a[2]] ,
                [a[0],a[1],a[2]+tau]]
        alphas_ori = a_origin + tau_origin * e 
        alphas_ori = [alphas_ori[0].tolist(),alphas_ori[1].tolist(),alphas_ori[2].tolist()]
        
        origin_sample = x_ori
        colors =['red','black','green']

        for (i, alpha) in enumerate(alphas):
            # plt.subplot(2, len(alphas), i + 1)
            dist = Dirichlet(alpha)
            dist_ori = Dirichlet(alphas_ori[i])
            alpha[i] -= tau 
            draw_pdf_contours(dist,linewidths=0.5, colors=colors[i])
            draw_pdf_contours(dist_ori,linewidths=0.5, colors='blue')
            title = r'$\alpha$ = (%.3f, %.3f, %.3f),$\tau$ = %.3f' % (alpha[0],alpha[1],alpha[2],tau)
            plt.title(title, fontdict={'fontsize': 8})
            # plt.subplot(2, len(alphas), i + 1 )#+ len(alphas))
            plot_points(origin_sample[i],color=colors[i],marker='o')s
        
        plt.savefig('./dirichlet-master/result/dirichlet_plots{}.png'.format(t))
        print('Wrote plots to "dirichlet_plots.png".')

        # plt.pause(0.1)

    
        f.write("{},{},{},\n".format(a, tau, pi))

        print('loglikelihood after %d iteration is %.2f '%(t, log_likelihood))
        print('parameters after %d iteration is:'%(t))
        print(a,tau,pi)
    print("Traing finished")
    print("Trained parameters are: alpha {}, tau {}, pi {}".format(a,tau,pi))
    return a, tau, pi
#initializated parameter
a = a_0
pi = pi_0
tau = tau_0
x_ori = [x0,x1,x2] 
y_ori = [y0,y1,y2]
a, tau, pi = EM_FDMM(x,y,a,pi,tau,x_ori,y_ori,a_origin,tau_origin,pi_origin)

def get_f1(true_positive, false_positive, false_negative):
    if true_positive == 0:
        return 0.0
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    return 2.0 * precision * recall / (precision + recall)

def find_maximum(f, min_x, max_x, epsilon=1e-5):
    def binary_search(l, r, fl, fr, epsilon):
        mid = l + (r - l) / 2
        fm = f(mid)
        binary_search.eval_count += 1
        if (fm == fl and fm == fr) or r - l < epsilon:
            return mid, fm
        if fl > fm >= fr:
            return binary_search(l, mid, fl, fm, epsilon)
        if fl <= fm < fr:
            return binary_search(mid, r, fm, fr, epsilon)
        p1, f1 = binary_search(l, mid, fl, fm, epsilon)
        p2, f2 = binary_search(mid, r, fm, fr, epsilon)
        if f1 > f2:
            return p1, f1
        else:
            return p2, f2

    binary_search.eval_count = 0

    best_th, best_value = binary_search(min_x, max_x, f(min_x), f(max_x), epsilon)
    # print("Found maximum %f at x = %f in %d evaluations" % (best_value, best_th, binary_search.eval_count))
    return best_th, best_value

def compute_threshold(y_scores, y_true):        
        minP = min(y_scores) 
        maxP = max(y_scores)
        y_false = np.logical_not(y_true)

        def evaluate(e):
            y = np.greater(y_scores, e)
            true_positive = np.sum(np.logical_and(y, y_true))
            false_positive = np.sum(np.logical_and(y, y_false))
            false_negative = np.sum(np.logical_and(np.logical_not(y), y_true))
            return get_f1(true_positive, false_positive, false_negative)

        best_th, best_f1 = find_maximum(evaluate, minP, maxP, 1e-3)

        print("Best e: %f best f1: %f" % (best_th, best_f1))
        return best_th

def evaluate(inliner_classes, threshold, prediction, gt_inlier,softmax_out,label):
    y = np.greater(prediction, threshold)
   
    gt_outlier = np.logical_not(gt_inlier)

    true_positive = np.sum(np.logical_and(y, gt_inlier))
    true_negative = np.sum(np.logical_and(np.logical_not(y), gt_outlier))
    false_positive = np.sum(np.logical_and(y, gt_outlier))
    false_negative = np.sum(np.logical_and(np.logical_not(y), gt_inlier))
    total_count = true_positive + true_negative + false_positive + false_negative

    accuracy = 100 * (true_positive + true_negative) / total_count
    y_all = np.argmax(softmax_out,1).squeeze()
    sf_all = np.asarray(label)
    y_in = y_all[y]
    sf_in = sf_all[y]
    acc = ((y_in==sf_in).astype(float).sum()+true_negative)/len(label)  
    y_true = gt_inlier
    y_scores = prediction

    try:
        auc = roc_auc_score(y_true, y_scores)
    except:
        auc = 0

    # logger.info("Percentage %f" % percentage_of_outliers)
    # logger.info("Accuracy %f" % accuracy)
    # logger.info("Multi_Acc %f" % acc)
    f1 = get_f1(true_positive, false_positive, false_negative)
    # logger.info("F1 %f" % get_f1(true_positive, false_positive, false_negative))
    # logger.info("AUC %f" % auc)

      
    
    # return dict(auc=auc, f1=f1)

    # inliers
    X1 = [x[1] for x in zip(gt_inlier, prediction) if x[0]]

    # outliers
    Y1 = [x[1] for x in zip(gt_inlier, prediction) if not x[0]]
    print(len(Y1),len(X1))

    minP = min(prediction) 
    maxP = max(prediction) 

    ##################################################################
    # FPR at TPR 95
    ##################################################################
    fpr95 = 0.0
    clothest_tpr = 1.0
    dist_tpr = 1.0
    for threshold in np.arange(minP, maxP, 1e-3):
        tpr = np.sum(np.greater_equal(X1, threshold)) / np.float(len(X1))
        fpr = np.sum(np.greater_equal(Y1, threshold)) / np.float(len(Y1))
        if abs(tpr - 0.99) < dist_tpr:
            dist_tpr = abs(tpr - 0.99)
            clothest_tpr = tpr
            print("tpr:{}, threshold:{}".format(clothest_tpr,threshold))
            fpr95 = fpr

    # logger.info("tpr: %f" % clothest_tpr)
    # logger.info("fpr95: %f" % fpr95)

    ##################################################################
    # Detection error
    ##################################################################
    error = 1.0
    for threshold in np.arange(minP, maxP, 1e-3):
        fnr = np.sum(np.less(X1, threshold)) / np.float(len(X1))
        fpr = np.sum(np.greater_equal(Y1, threshold)) / np.float(len(Y1))
        error = np.minimum(error, (fnr + fpr) / 2.0)

    # logger.info("Detection error: %f" % error)

    ##################################################################
    # AUPR IN
    ##################################################################
    auprin = 0.0
    recallTemp = 1.0
    for threshold in np.arange(minP, maxP, 1e-3):
        tp = np.sum(np.greater_equal(X1, threshold))
        fp = np.sum(np.greater_equal(Y1, threshold))
        if tp + fp == 0:
            continue
        precision = tp / (tp + fp)
        recall = tp / np.float(len(X1))
        auprin += (recallTemp - recall) * precision
        recallTemp = recall
    auprin += recall * precision

    # logger.info("auprin: %f" % auprin)

    ##################################################################
    # AUPR OUT
    ##################################################################
    minP, maxP = -maxP, -minP
    X1 = [-x for x in X1]
    Y1 = [-x for x in Y1]
    auprout = 0.0
    recallTemp = 1.0
    for threshold in np.arange(minP, maxP, 1e-3):
        tp = np.sum(np.greater_equal(Y1, threshold))
        fp = np.sum(np.greater_equal(X1, threshold))
        if tp + fp == 0:
            continue
        precision = tp / (tp + fp)
        recall = tp / np.float(len(Y1))
        auprout += (recallTemp - recall) * precision
        recallTemp = recall
    auprout += recall * precision

    # logger.info("auprout: %f" % auprout)

    # with open(os.path.join("results.txt"), "a") as file:
    #     file.write(
    #         "Class: %s\n Percentage: %d\n"
    #         "Error: %f\n F1: %f\n AUC: %f\nfpr95: %f"
    #         "\nDetection: %f\nauprin: %f\nauprout: %f\n Multi_acc: %f\n\n" %
    #         ("_".join([str(x) for x in inliner_classes]), percentage_of_outliers, error, f1, auc, fpr95, error, auprin, auprout, acc))

    # return dict(auc=auc, acc = accuracy, f1=f1, fpr95=fpr95, error=error, auprin=auprin, auprout=auprout, multi_acc=acc)
    return auc, f1, fpr95, error, auprin, auprout

def test_FDMM(x_test,y_test,a,pi,tau):
    y_test_logit = np.zeros((x_test.shape[0],component_num))
    for j in range(component_num):
        for i in range(x_test.shape[0]):
            y_test_logit[i][j] = post_prob(x_test[i],a,pi,tau,dim=j)

    ff = plt.figure(figsize=(16, 12))
    alphas = [[a[0]+tau,a[1],a[2]],
            [a[0],a[1]+tau,a[2]] ,
            [a[0],a[1],a[2]+tau]]
 
    colors =['blue','black','green']

    for (i, alpha) in enumerate(alphas):
        # plt.subplot(2, len(alphas), i + 1)
        dist = Dirichlet(alpha)
        # dist_ori = Dirichlet(alphas_ori[i])
        alpha[i] -= tau 
        draw_pdf_contours(dist,linewidths=0.5, colors=colors[i])
        # draw_pdf_contours(dist_ori,linewidths=0.5, colors='blue')
        title = r'$\alpha$ = (%.3f, %.3f, %.3f),$\tau$ = %.3f' % (alpha[0],alpha[1],alpha[2],tau)
        plt.title(title, fontdict={'fontsize': 8})
        # plt.subplot(2, len(alphas), i + 1 )#+ len(alphas))
        plot_points(x_test[y_test==i],color=colors[i],marker='o')
    plot_points(x_test[y_test==3],color='red',marker='o')
    
    plt.savefig('./dirichlet-master/result/dirichlet_plots_test.png')
    print('Wrote plots to "dirichlet_plots.png".')
    y_true = np.isin(y_test,[0,1,2])
    sf_score = np.max(x_test,axis=1)
    print(sf_score[-10::])
    dir_score = np.max(y_test_logit,axis=1)
    print(dir_score[-10::])
    sf_threshold = compute_threshold(sf_score,y_true )
    dir_threshold = compute_threshold(dir_score, y_true)
    sf_dir_threshold = compute_threshold(dir_score*sf_score, y_true)
    print((dir_score*sf_score)[-10::])
    auc, f1, fpr95, error, auprin, auprout = evaluate([0,1,2], sf_threshold, sf_score, y_true, x_test,y_test)
    print("softmax method : auc:{}, f1:{}, fpr95:{}, error:{}, auprin:{}, auprout:{}".format(auc, f1, fpr95, error, auprin, auprout))
    auc, f1, fpr95, error, auprin, auprout = evaluate([0,1,2], dir_threshold, dir_score, y_true,x_test,y_test)
    print("dirichlet method :auc:{}, f1:{}, fpr95:{}, error:{}, auprin:{}, auprout:{}".format(auc, f1, fpr95, error, auprin, auprout))
    auc, f1, fpr95, error, auprin, auprout = evaluate([0,1,2], sf_dir_threshold, dir_score*sf_score, y_true, x_test,y_test)
    print("sf+dir method : auc:{}, f1:{}, fpr95:{}, error:{}, auprin:{}, auprout:{}".format(auc, f1, fpr95, error, auprin, auprout))

Print("testing")
test_FDMM(x_test,y_test,a,pi,tau)

    


