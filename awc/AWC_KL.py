__author__ = 'kirill, larisa'

import scipy.spatial.distance as sci
import matplotlib.pylab as plt
import matplotlib as mpl
import numpy as np
import copy, time
import scipy.special as spec
from pylab import rcParams
from sklearn import decomposition
mpl.rcParams['axes.color_cycle'] = ['darkkhaki', 'grey', 'rosybrown', 'coral', 'deeppink', 'darkblue', 'deepskyblue', 'aqua', 'sienna']

def onclick(event, X, weights, f, ax2, h_1, h_2):
    ax2.clear()
    p = [event.button, event.xdata, event.ydata]
    picked = [p[1], p[2]]
    massiv = []
    for i in xrange(len(X)) :
        massiv.append(sci.euclidean(X[i, 0:2], picked))
    i = np.argmin(massiv)
    #i = 234
    print 'clicked point', i
    ax2.scatter(X[:, 0], X[:, 1], c=weights[i], cmap = 'YlOrRd', vmin=0, vmax=1.)
    circle2=plt.Circle( X[i, :], h_1, color='b',fill=False)
    circle3=plt.Circle( X[i, :], h_2, color='b',fill=False)
    ax2.add_artist(circle2)
    ax2.add_artist(circle3)
    #ax2.axis('equal')
    f.canvas.draw()

def k_loc(x):
    return x <= 1

def k_stat(x):
    return 1. * (x <= 0)
    return (2.- x).clip(min=0, max=1)

def distance_matrix(X):
    return sci.squareform(sci.pdist(X, 'euclidean'))

def get_neighbour_numbers(h, dist_matrix, weights):
    return np.sum(weights * k_loc(dist_matrix / h), axis = 1)

h_ratio = 1.95

def get_lambda_hash(h, d, L_T):
    global h_ratio
    #print 'a', np.max(np.min((d / h / h_ratio * 10000 ).astype(int), 10000-1))
    return np.take(L_T, np.minimum((d / h / h_ratio * (len(L_T) - 1) ).astype(int), (len(L_T) - 1)))


def get_lambda_table(n, m):
    global h_ratio
    x = np.linspace(0 + 1./m, h_ratio + 1./m, m+1)
    a = spec.betainc((n+1.) / 2, 0.5, 1. - (x/2.)**2)
    return a / (2-a)# - 1.


n_0 = 10

def initialisation(x, H):
    global n_0
    #x = x / H[0]
    dist_matrix = np.sort(x, axis=1)
    neighbor_number = copy.deepcopy(n_0)
    v = dist_matrix[:, neighbor_number - 1]
    n = np.size(x,0)
    a = np.zeros((n,n))
    for i in xrange(n):
        #a[i,:] = np.minimum(np.exp(-(x[i,:]/max(v[i], h_0)-1.) / 0.002), 1.)
        h_closest = H[-1]
        for h in H:
            if h >= v[i]:
                h_closest = h
                break
        a[i, :] = 1 * (x[i,:] <= h_closest) 
    a = np.maximum(a, a.T)
    #a = a.astype(bool)
    #return np.ones((n,n))
    return a

def draw_step(weights, X1, true_clusters, clustering, h_1, h_2, true_weights=None):
    weights = 1 * weights
    rcParams['figure.figsize'] = 10, 10
    cluster_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'gold', 'firebrick', 'olive', 'springgreen', 'palevioletred', 'hotpink','lightgreen']
    #cluster_colors = ['r','lightgreen','b', 'hotpink', 'yellow', 'g', 'c', 'm', 'gold', 'firebrick', 'olive', 'springgreen', 'palevioletred']
    #plt.ylim([-0.3,3.3])
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    ax1.imshow(weights, cmap=plt.cm.gray, interpolation='nearest')
    
    #if np.size(X1, 1) > 2:
    #    X = X1[:, [1, 3]]
    #el
    if np.size(X1, 1) > 2:
        pca = decomposition.FastICA(n_components=2)
        pca.fit(X1)
        X = pca.transform(X1)
    else:
        X = X1
    #X = X1
    #X = X1
    n = np.size(X, 0)
            
    adjmatrix = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        for j in range(n):
            if weights[i, j] >= 0.5:
                adjmatrix[i, j] = 1
                
    colors = np.zeros((n,))
    for cluster_i in range(len(true_clusters)):
        #print 'h', len(true_clusters)
        cluster = true_clusters[cluster_i]
        x = X[cluster, :]
        colors[cluster] = cluster_i
        #print 'o' + cluster_colors[cluster_i]
        ax4.plot(x[:, 0], x[:, 1],  marker='o', linestyle='None', color=cluster_colors[cluster_i])      
    if clustering:
        U = copy.deepcopy(X)
        points = range(len(X))
        not_used_colors = range(len(cluster_colors))
        while(np.size(U, 0) != 0):
            neigbohours = np.sum(adjmatrix, axis=1)
            candidates = np.argsort(neigbohours)
            #candidates = np.argsort(X[points, 0])
            for cluster_generater in reversed(candidates):#xrange(len(U)):
                neigbohours_i = neigbohours[adjmatrix[cluster_generater, :] == 1]
                if np.sum(neigbohours_i > (neigbohours[cluster_generater] - 5)) > 0.9 * np.size(neigbohours_i,0):
                    break
            all_cliques = ()
            for i in range(len(U)):
                if adjmatrix[cluster_generater, i] == 1:
                    all_cliques += tuple([i])
            all_cliques = [all_cliques]
            if len(not_used_colors) > 0:
                colors_n = np.zeros((len(not_used_colors),))
                for k in range(len(colors_n)):
                    colors_n[k] = len(colors[colors[list(all_cliques[0]),] == not_used_colors[k]])
                true_color = np.argmax(colors_n)
                #print 'o' + cluster_colors[not_used_colors[true_color]]
                ax3.plot(U[all_cliques[0], 0], U[all_cliques[0], 1],  marker='o', linestyle='None', color=cluster_colors[not_used_colors[true_color]])
                #ax.scatter(U[all_cliques[0], 0], U[all_cliques[0], 1], U[all_cliques[0], 2],zdir='z', c=cluster_colors[not_used_colors[true_color]])
                del not_used_colors[true_color]
                colors = np.delete(colors, all_cliques[0], 0)
            else:
                ax3.plot(U[all_cliques[0], 0], U[all_cliques[0], 1], 'o')
                #ax.scatter(U[all_cliques[0], 2], U[all_cliques[0], 0], U[all_cliques[0], 1],zdir='z')
            U = np.delete(U, all_cliques[0], 0)  
            adjmatrix = np.delete(adjmatrix, all_cliques[0], 0)
            adjmatrix = np.delete(adjmatrix, all_cliques[0], 1)
            #print all_cliques[0]
            #print points
            points = [points[i] for i in range(len(points)) if i not in all_cliques[0]]
    #plt.show()
    if true_weights == None:
        ax2.plot(X[:, 0], X[:, 1], 'go')
        f.canvas.callbacks.connect('button_press_event', lambda event: onclick(event, X, weights, f, ax2, h_1, h_2))
        ax2.axis('equal')
    else:
        ax2.imshow(true_weights, cmap=plt.cm.gray, interpolation='nearest')
    
    ax3.axis('equal')
    ax4.axis('equal')
    plt.show()
    return 0

lower_bound = 0.0

def ball_volume(r, dim):
    if dim % 2 == 0:
        return  (math.pi ** (dim / 2) ) / math.factorial(dim / 2) * r**dim
    else:
        return 2 * math.factorial((dim - 1) / 2) * ((4 * math.pi) ** ((dim - 1) / 2) ) * (r ** dim) / math.factorial(dim)
    

def correct_weights(weights):
    global weights_init
    a = np.sum(weights, axis=1)
    A = a < 2
    weights[A, :] = weights_init[A, :]
    weights[:, A] = np.maximum(weights_init[A, :].T, weights[:, A])
    return weights

def connect_alone(weights):
    a = np.sum(weights, axis=1)
    A = a < 2
    weights[A, :][:, A] = 1
    return weights

old_weights = -1

def find_unchanged_weights(old_weights, weights):
    dif_weights = weights - old_weights
    n = np.size(dif_weights, 0)
    #print dif_weights
    zeros = (dif_weights == np.zeros((n,n ))) * 1
    #print 'zeros', zeros
    unchanged = np.sum(zeros, axis=1)
    #print np.sum(unchanged == n)
    
    return unchanged == n
    
def cluster_step(X, l, weights, v, n, k, L_T, T, KL, dist_matrix, H, log_show, method, show_step, true_clusters, T_stat_show, propagation=False):
    #global old_weights
    #if k != 1:
    #    K = find_unchanged_weights(old_weights, weights)
    #old_weights = copy.deepcopy(weights)
    if log_show:
        print len(H) - k, k+1
    neighbour_numbers = np.sum(weights * (dist_matrix <= H[k-1]), axis = 1) - 1
    D2 = dist_matrix <= H[k-1]
    np.fill_diagonal(D2, False)
    P =  D2 * weights
    #max_dist = np.max(dist_matrix, axis=1)
    
    max_dist = np.max(dist_matrix, axis=1)
    x = 246
    y = 264
    
    
    t_1 = (neighbour_numbers[np.newaxis, :] - P).T
    
    t_12 = np.inner(P, P)
    t_12x = np.inner(P, D2)
    #print 't_1=', t_1[x, y]
    gg1 = (t_1 == t_12x) * (t_12 < 0.5 * t_12x)
    #gg2 = (t_1.T == t_12x.T) * (t_12 < 0.5 * t_12x.T)
    
    t_1 = t_1 - t_12x + t_12
    q = get_lambda_hash(H[k-1], dist_matrix, L_T)
    #E = (max_dist[i] < H[k-1]) * ( max_dist[i+1:] < H[k-1])
    #q[E] = 1. / get_lambda_hash(np.maximum(max_dist[i], max_dist[i+1:][E]), dist_matrix[i, i+1:][E], L_T)
    
    E = max_dist < H[k-1]
    F = np.repeat([max_dist], n, axis = 0)
    R = np.maximum(F.T, F)
    q[E, :][:, E] = get_lambda_hash(R[E, :][:, E], dist_matrix[E, :][:, E], L_T)
    
    
    t = t_1 + t_1.T - t_12
    e = t_12 / t
    #print 't_12x=', t_12x[x, y], 't_21x=', t_12x[y, x], 't_1=', t_1[x, y]
    #print 't_2=', t_1[y, x], 't_12=', t_12[x, y], 'e=', e[x, y], 'q=', q[x, y]
    
    aa = e >= 0.95
    e[t == 0] = 0
    e = e.clip(min=0.05, max=0.9)
    q = q.clip(min=0.05, max=0.9)
    bb = e <= 0.05
    e *= 2000
    q *= 2000
    
    e = e.astype(int)
    q = q.astype(int)
    
    T = t * KL[q, e]
    T[np.logical_or(bb, t_12 == 0)] = np.nan
    T[aa] = l
    sum_v = v > H[k-1]
    T[sum_v, :] = float("inf")
    T[:, sum_v] = float("inf")
    T[np.logical_or(gg1, gg1.T)] = np.nan
    #T[gg2] = np.nan
    
    #print 'T', T[x, y], T[y,x]
    ###print 'T[' + str(x) + ',' + str(y) + ']', T[x, y]
    #start_time = time.time()
    I = (dist_matrix <= H[k]) * (dist_matrix > 0) * (T != float("inf")) * (np.isnan(T) == False)
    weights[I] = 1 * (T[I] <= l)
    start_time = time.time()
    weights[np.isnan(T)] = 0
    np.fill_diagonal(weights, 1)
    
    if show_step==1  and k > 1000000:
        draw_step(weights, X, true_clusters, clustering,  H[k-1], H[k])
    
    #weights = connect_alone(weights)
    
    #weights = correct_weights(weights)
    #print weights[276, :][I[276, :]]
    #print 'T[203, 302]', T[203, 302], weights[203, 302]#, np.isnan(T[203, 302])
    #weights[352, 240] = 0
    #if k != 1:
    #    weights[K, :][:, K] = old_weights[K, :][:, K]
    return 0

def KL_init():
    m = 2000
    e1 = np.linspace(0, 1, m + 1)
    e = np.repeat([e1], m + 1, axis=0)
    q = e.T
    #print e[0, :]
    #print q[0, :]
    KL = (-1) ** (e > q) * (e-q) * np.log((e * (1. - q) / q / (1. - e)))
    #KL = ((e-q) * np.log((e / q)) + ((q - e) ) * np.log(  (1 - e) / (1 - q)  )  )
    #KL = (-1) ** (e > q) * (e * np.log(e / q) + (1 - e) * np.log(  (1 - e) / (1 - q)  )  )
    #KL = (-1) ** (e > q) * ((-q) * np.log((e / q)) + ((-1 + q) ) * np.log(  (1 - e) / (1 - q)  )  )
    #KL[np.isnan(KL)] = 0 
    KL = np.nan_to_num(KL)
    #print KL[0,1]
    print 'sss', KL[-1, 0]
    return KL 

weights_init = 0    
    
def init(X, n_neigh):
    global n_0#, weights_init
    if n_neigh == -1:
        n_0 = 2 * np.size(X, 1) + 3
        #n_0 = 12
    print 'lll', n_0
    rcParams['figure.figsize'] = 10, 10
    rcParams['figure.figsize'] = 8, 6
    n = len(X)
    L_T = get_lambda_table(np.size(X,1), 10000)
    
    dist_matrix = distance_matrix(X)
    H, dist_ordered = get_h_intervals(dist_matrix)
    v = dist_ordered[:, n_0-1].clip(min=H[0])
    #weights = np.zeros((n, n))
    weights = initialisation(dist_matrix, H)  
    #weights_init = copy.deepcopy(weights)  
    #flag = np.zeros((n,n), dtype=np.int8)
    T = np.zeros((n, n))
    KL = KL_init()
    return n, L_T, dist_matrix, H, dist_ordered, v, weights, T, KL

    
def cluster(X, l, true_clusters = [], true_weights=None,  show_step = False, show_finish = False, T_stat_show = False, clustering = False, log_show = True, n_neigh = -1, method = 1, step=None):
    n, L_T, dist_matrix, H, dist_ordered, v, weights, T, KL = init(X, n_neigh)
    ###KL = KL_init(1000)
    if show_step:
        draw_step(weights, X, true_clusters, clustering,  H[0], H[1], true_weights=true_weights)
    for k in range(1, len(H)):
        print 'k=', k, '/', len(H)
        cluster_step(X, l, weights, v, n, k, L_T, T, KL, dist_matrix, H, log_show, method, show_step, true_clusters, T_stat_show)
        if show_step and k < len(H) - 1:
            draw_step(weights, X, true_clusters, clustering,  H[k-1], H[k], true_weights=true_weights)
        #correct_weights(weights)
    if show_finish:
        draw_step(weights, X, true_clusters, clustering,  H[k-1], H[k], true_weights=true_weights)
    if log_show:
        print '' 
    return weights
    
def get_h_intervals(dist_matrix, log_show=False):
    global n_0, h_ratio
    #print 'n_0=', n_0
    print '1', dist_matrix[0, :]
    dist_matrix = np.sort(dist_matrix)
    h_intervals = [np.percentile(dist_matrix[:, n_0 - 1], 30)]
    neighbor_number = copy.deepcopy(n_0)
    neighbor_number *= 2 ** 0.5
    #plt.plot(range(len(h_intervals)), h_intervals)
    #plt.show()
    #return h_intervals, dist_matrix
    ### New idea  
    
    ### Another Idea
    neighbor_number_seq = [n_0]
    while(1):
        a = int(neighbor_number_seq[-1] * 2 ** 0.3)
        if a < np.size(dist_matrix, 0)-1:
            neighbor_number_seq.append(a)
        else:
            a = np.size(dist_matrix, 0) - 1
            neighbor_number_seq.append(a)
            break
    #print 'NNN', neighbor_number_seq
    
    h_intervals = np.reshape(dist_matrix[:, neighbor_number_seq], (-1))
    
    indexes = copy.deepcopy((dist_matrix))
    for i in xrange(np.size(indexes,0)):
        indexes[i, :] = i
    indexes_h = np.reshape(indexes[:, neighbor_number_seq], (-1))
    permutation_sort = np.argsort(h_intervals)
    
    #h_intervals = h_intervals[permutation_sort]
    indexes_h = indexes_h[permutation_sort]
    indexes_h = indexes_h.astype(int)
    
    print 'RRRRRRRRR'
    
    h_final = []
    stack = []
    n_break_points = 0
    break_level = np.size(dist_matrix, 0) / 10
    for i in range(len(h_intervals)):
        nn = stack.count(indexes_h[i])
        if nn < 2 and n_break_points < break_level:
            stack.append(indexes_h[i])
            if nn == 1:
                n_break_points += 1
        else:
            h_final.append(h_intervals[permutation_sort[i]])
            stack = []
            n_break_points = 0
        if i == len(h_intervals)-1:
            h_final.append(h_intervals[permutation_sort[i]])
            break
    print 'RRRRRRRRR'
    print len(h_final), len(h_intervals), h_final
    #return h_final, dist_matrix
    a = [h_final[0]]
    for i in xrange(len(h_final)-1):
        if h_final[i+1] - h_final[i] != 0:
            a.append(h_final[i+1])
    #a.append(a[-1])
    #a = a + a
    if a[0] == 0:
        del a[0]
    for i in range(6):
        a.append(a[-1]* 1.5)
    print 'a', a
    return a, dist_matrix
 

def get_error(weights, true_weights, separate_errors = False):
    error_1 = np.sum(np.abs(weights) * (true_weights == 0)) / np.sum(np.abs(1 - true_weights) * (true_weights == 0))
    error_2 = np.sum(np.abs(weights - 1) * (true_weights == 1)) / np.sum(np.abs(np.identity(np.size(weights,0)) - 1) * (true_weights == 1))
        
    if separate_errors:
        return error_1, error_2
    else:
        return (np.sum(np.abs(weights) * (true_weights == 0)) + np.sum(np.abs(weights - 1) * (true_weights == 1))) / np.size(weights,0) / (np.size(weights,0)-1)


def show_distances(X):
    dist_matrix = distance_matrix(X)
    n = len(X)
    A = np.zeros(((n**2 - n) / 2, 1))
    #A = np.zeros((n-1, 1))
    A = dist_matrix[np.triu_indices(n)]
    plt.hist(A, bins = 100, normed=1)
    #plt.show()
