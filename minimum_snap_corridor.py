import osqp
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import sparse
from mpl_toolkits.mplot3d import axes3d

show_animation = False

'''
根据路径总长平均分配时间
'''
def ArrangeT(waypts, T):
    ts = np.zeros(len(waypts))
    delta_xy = waypts[1:,:] - waypts[:-1,:]
    dist = np.sqrt((delta_xy**2).sum(axis=1))    
    ts[1:] = np.cumsum(dist * T / sum(dist))
    return ts

'''
计算目标函数中 min F = X^T * H * X + q * X 的 H 
# n:polynormial order
# r:derivertive order, 1:minimum vel 2:minimum acc 3:minimum jerk 4:minimum snap
# ts:start timestamp for polynormial
# te:end timestap for polynormial
'''

def ComputeH(n, r, ts, te):
    
    C = np.zeros((n+1,n+1))
    T_mat = np.zeros((n+1,n+1))
    # ( i! / (i-r)! ) * ( j! / (j-r)! ) / (i - r + j - r + 1) * ( t(i)^k - t(i-1)^k )
    for i in range(r, n+1):
        C[i,i] = math.factorial(i) / math.factorial(i-r)
    
    for i in range(r, n+1):
        for j in range(r, n+1):
            k = (i - r) + (j - r) + 1
            T_mat[i,j] = 1.0 / k * (te**k-ts**k)
    H = np.matmul(C, np.matmul(T_mat, C))
    return H

'''
计算多项式 r 阶导后 t^i 与 对应系数 Ci  向量
'''

def CalculateTVec(t, n_order, r):
    TVec = np.zeros(n_order+1)
    for i in range(r, n_order+1):
        TVec[i] = math.factorial(i) / math.factorial(i-r) * t**(i-r)
    return TVec

def minimum_snap_single_axis_simple(waypts, ts, n_order, v0, a0, ve, ae):
    p0 = waypts[0]
    pe = waypts[-1]
    # print(p0, pe)
    
    n_poly = len(waypts) - 1
    n_coef = n_order + 1

    H = np.zeros((n_poly*(n_coef), n_poly*(n_coef)))

    for i in range(n_poly):
        Hi = ComputeH(n_order, 3, ts[i], ts[i+1])
        # print(i)
        H[n_coef*i:n_coef*i+n_coef, n_coef*i:n_coef*i+n_coef] = Hi
    np.set_printoptions(suppress=True)
    # print(H)
    
    # 等式约束
    Aeq = np.zeros((4*n_poly+2, n_coef*n_poly))
    beq = np.zeros(4*n_poly+2)
    # 起点终点约束
    Aeq[0, 0:n_coef] = CalculateTVec(ts[0], n_order, 0)
    Aeq[1, 0:n_coef] = CalculateTVec(ts[0], n_order, 1)
    Aeq[2, 0:n_coef] = CalculateTVec(ts[0], n_order, 2)

    Aeq[3, n_coef*(n_poly-1):n_coef*n_poly] = CalculateTVec(ts[-1], n_order, 0)
    Aeq[4, n_coef*(n_poly-1):n_coef*n_poly] = CalculateTVec(ts[-1], n_order, 1)
    Aeq[5, n_coef*(n_poly-1):n_coef*n_poly] = CalculateTVec(ts[-1], n_order, 2)

    beq[0:6] = np.array([p0,v0,a0,pe,ve,ae])
    
    neq = 5
    # 位置约束
    for i in range(n_poly-1):
        neq = neq + 1
        Aeq[neq, n_coef*(i+1):n_coef*(i+2)] = CalculateTVec(ts[i+1], n_order, 0)
        beq[neq] = waypts[i+1]
    # 连续性约束
    for i in range(n_poly-1):
        tvec_p = CalculateTVec(ts[i+1], n_order, 0)
        tvec_v = CalculateTVec(ts[i+1], n_order, 1)
        tvec_a = CalculateTVec(ts[i+1], n_order, 2)
        neq = neq + 1
        Aeq[neq, n_coef*i:n_coef*(i+1)] = tvec_p
        Aeq[neq, n_coef*(i+1):n_coef*(i+2)] = -tvec_p
        neq = neq + 1
        Aeq[neq, n_coef*i:n_coef*(i+1)] = tvec_v
        Aeq[neq, n_coef*(i+1):n_coef*(i+2)] = -tvec_v
        neq = neq + 1
        Aeq[neq, n_coef*i:n_coef*(i+1)] = tvec_a
        Aeq[neq, n_coef*(i+1):n_coef*(i+2)] = -tvec_a

    P = sparse.csc_matrix(H)
    q = np.zeros(n_poly*n_coef)
    A = sparse.csc_matrix(Aeq)
    L = beq
    U = beq

    new_prob = osqp.OSQP()
    new_prob.setup(P,q,A,L,U,alpha=1.6)
    res = new_prob.solve()
    polys = np.array(res.x)
    polys = np.reshape(polys,(n_poly, n_coef))

    return polys

def minimum_snap_single_axis_corridor(waypts, ts, n_order, v0, a0, ve, ae, corridor_r):
    p0 = waypts[0]
    pe = waypts[-1]
    
    n_poly = len(waypts) - 1
    n_coef = n_order + 1

    H = np.zeros((n_poly*(n_coef), n_poly*(n_coef)))
    # np.set_printoptions(suppress=True, threshold=np.inf)
    for i in range(n_poly):
        Hi = ComputeH(n_order, 3, ts[i], ts[i+1])
        H[n_coef*i:n_coef*i+n_coef, n_coef*i:n_coef*i+n_coef] = Hi
    
    Aeq = np.zeros((3*n_poly+3, n_coef*n_poly))
    beq = np.zeros(3*n_poly+3)
    '''
    设置初始条件  起点约束  终点约束  
    *** 起始速度加速度对轨迹平滑影响非常大,不能直接设置为0 ***
    ''' 
    # start/terminal pva constraints  (6 equations)
    Aeq[0, 0:n_coef] = CalculateTVec(ts[0], n_order, 0)
    # Aeq[1, 0:n_coef] = CalculateTVec(ts[0], n_order, 1)
    # Aeq[2, 0:n_coef] = CalculateTVec(ts[0], n_order, 2)

    Aeq[3, n_coef*(n_poly-1):n_coef*n_poly] = CalculateTVec(ts[-1], n_order, 0)
    # Aeq[4, n_coef*(n_poly-1):n_coef*n_poly] = CalculateTVec(ts[-1], n_order, 1)
    # Aeq[5, n_coef*(n_poly-1):n_coef*n_poly] = CalculateTVec(ts[-1], n_order, 2)

    beq[0:6] = np.array([p0,v0,a0,pe,ve,ae])
    
    neq = n_order
    
    # 连续性约束  ((n_poly-1)*3 equations)
    for i in range(n_poly-1):
        tvec_p = CalculateTVec(ts[i+1], n_order, 0)
        tvec_v = CalculateTVec(ts[i+1], n_order, 1)
        tvec_a = CalculateTVec(ts[i+1], n_order, 2)
        neq = neq + 1
        Aeq[neq, n_coef*i:n_coef*(i+1)] = tvec_p
        Aeq[neq, n_coef*(i+1):n_coef*(i+2)] = -tvec_p
        neq = neq + 1
        Aeq[neq, n_coef*i:n_coef*(i+1)] = tvec_v
        Aeq[neq, n_coef*(i+1):n_coef*(i+2)] = -tvec_v
        neq = neq + 1
        Aeq[neq, n_coef*i:n_coef*(i+1)] = tvec_a
        Aeq[neq, n_coef*(i+1):n_coef*(i+2)] = -tvec_a
    
    a = np.zeros((n_poly-1, n_coef*n_poly))
    l = np.zeros(n_poly-1)
    u = np.zeros(n_poly-1)

    for i in range(n_poly - 1):
        tvec_p = CalculateTVec(ts[i+1], n_order, 0)
        a[i, n_coef*(i):n_coef*(i+1)] = tvec_p
        l[i] = waypts[i+1] - corridor_r
        u[i] = waypts[i+1] + corridor_r

    A = np.concatenate((a, Aeq), axis = 0)
    L = np.concatenate((l, beq), axis = 0)
    U = np.concatenate((u, beq), axis = 0)

    P = sparse.csc_matrix(H)
    q = np.zeros(n_poly*n_coef)
    A = sparse.csc_matrix(A)

    new_prob = osqp.OSQP()
    new_prob.setup(P,q,A,L,U,alpha=1.8)
    res = new_prob.solve()
    
    polys = np.array(res.x)
    polys = np.reshape(polys,(n_poly, n_coef))

    return polys


def minimum_snap_single_axis_guiding_path(waypts, ts, n_order, v0, a0, ve, ae, corridor_r, lamda):
    p0 = waypts[0]
    pe = waypts[-1]

    n_poly = len(waypts) - 1
    n_coef = n_order + 1

    Q_all = np.zeros((n_poly*(n_coef), n_poly*(n_coef)))

    np.set_printoptions(suppress=True, threshold=np.inf)

    for i in range(n_poly):
        Q_i = ComputeH(n_order, 3, ts[i], ts[i+1])
        Q_all[n_coef*i:n_coef*i+n_coef, n_coef*i:n_coef*i+n_coef] = Q_i

    b_all = np.zeros(n_poly*n_coef)

    H_guide = np.zeros((n_poly*(n_coef), n_poly*(n_coef)))
    b_guide = np.zeros(n_poly*n_coef)
    
    for i in range(n_poly):
        t1 = ts[i]
        t2 = ts[i+1]
        Q_i = ComputeH(n_order, 0, ts[i], ts[i+1])
        H_guide[n_coef*i:n_coef*i+n_coef, n_coef*i:n_coef*i+n_coef] = Q_i

        p1 = waypts[i]
        p2 = waypts[i+1]
        a1 = (p2-p1) / (t2-t1)
        aa0 = p1 - a1*t1
        ci = np.zeros(n_coef)
        ci[:2] = [aa0, a1]
        bi = -np.dot(Q_i, ci)
        b_guide[n_coef*i:n_coef*i+n_coef] = np.transpose(bi)

    Q_all = Q_all + lamda * H_guide
    b_all = b_all + lamda * b_guide

    Aeq = np.zeros((3*n_poly+3, n_coef*n_poly))
    beq = np.zeros(3*n_poly+3)

    # start/terminal pva constraints  (6 equations)
    Aeq[0, 0:n_coef] = CalculateTVec(ts[0], n_order, 0)
    # Aeq[1, 0:n_coef] = CalculateTVec(ts[0], n_order, 1)
    # Aeq[2, 0:n_coef] = CalculateTVec(ts[0], n_order, 2)

    Aeq[3, n_coef*(n_poly-1):n_coef*n_poly] = CalculateTVec(ts[-1], n_order, 0)
    # Aeq[4, n_coef*(n_poly-1):n_coef*n_poly] = CalculateTVec(ts[-1], n_order, 1)
    # Aeq[5, n_coef*(n_poly-1):n_coef*n_poly] = CalculateTVec(ts[-1], n_order, 2)

    beq[0:6] = np.array([p0,v0,a0,pe,ve,ae])

    neq = n_order
    
    # continuous constraints  ((n_poly-1)*3 equations)
    for i in range(n_poly-1):
        tvec_p = CalculateTVec(ts[i+1], n_order, 0)
        tvec_v = CalculateTVec(ts[i+1], n_order, 1)
        tvec_a = CalculateTVec(ts[i+1], n_order, 2)
        neq = neq + 1
        Aeq[neq, n_coef*i:n_coef*(i+1)] = tvec_p
        Aeq[neq, n_coef*(i+1):n_coef*(i+2)] = -tvec_p
        neq = neq + 1
        Aeq[neq, n_coef*i:n_coef*(i+1)] = tvec_v
        Aeq[neq, n_coef*(i+1):n_coef*(i+2)] = -tvec_v
        neq = neq + 1
        Aeq[neq, n_coef*i:n_coef*(i+1)] = tvec_a
        Aeq[neq, n_coef*(i+1):n_coef*(i+2)] = -tvec_a

    # corridor constraints (n_ploy-1 iequations)
    Aieq = np.zeros((2*(n_poly - 1), n_coef*n_poly))
    bieq = np.zeros(2*(n_poly - 1))
    for i in range(n_poly - 1):
        tvec_p = CalculateTVec(ts[i+1], n_order, 0)
        Aieq[2*i, n_coef*(i+1):n_coef*(i+2)] = tvec_p
        Aieq[2*i+1, n_coef*(i+1):n_coef*(i+2)] = -tvec_p
        bieq[2*i] = waypts[i+1] + corridor_r
        bieq[2*i + 1] = corridor_r - waypts[i+1]
    
    a = np.zeros((n_poly-1, n_coef*n_poly))
    l = np.zeros(n_poly-1)
    u = np.zeros(n_poly-1)

    for i in range(n_poly - 1):
        tvec_p = CalculateTVec(ts[i+1], n_order, 0)
        a[i, n_coef*(i):n_coef*(i+1)] = tvec_p
        l[i] = waypts[i+1] - corridor_r
        u[i] = waypts[i+1] + corridor_r

    A = np.concatenate((a, Aeq), axis = 0)
    L = np.concatenate((l, beq), axis = 0)
    U = np.concatenate((u, beq), axis = 0)

    P = sparse.csc_matrix(Q_all)
    q = b_all
    A = sparse.csc_matrix(A)

    new_prob = osqp.OSQP()
    new_prob.setup(P,q,A,L,U,alpha=0.8)
    res = new_prob.solve()
    
    polys = np.array(res.x)
    polys = np.reshape(polys,(n_poly, n_coef))

    return polys


def minimum_snap_single_axis_close_form(waypts, ts, n_order, v0, a0, ve, ae):
    # p0 = waypts[0]
    # pe = waypts[-1]
    # print(p0, pe)
    
    n_poly = len(waypts) - 1
    n_coef = n_order + 1

    H = np.zeros((n_poly*(n_coef), n_poly*(n_coef)))

    for i in range(n_poly):
        Hi = ComputeH(n_order, 3, ts[i], ts[i+1])
        # print(i)
        H[n_coef*i:n_coef*i+n_coef, n_coef*i:n_coef*i+n_coef] = Hi
    np.set_printoptions(suppress=True)
    # print(H)
    
    # 等式约束
    Aeq = np.zeros((4*n_poly+2, n_coef*n_poly))
    beq = np.zeros(4*n_poly+2)
    # 起点终点约束
    Aeq[0, 0:n_coef] = CalculateTVec(ts[0], n_order, 0)
    Aeq[1, 0:n_coef] = CalculateTVec(ts[0], n_order, 1)
    Aeq[2, 0:n_coef] = CalculateTVec(ts[0], n_order, 2)

    Aeq[3, n_coef*(n_poly-1):n_coef*n_poly] = CalculateTVec(ts[-1], n_order, 0)
    Aeq[4, n_coef*(n_poly-1):n_coef*n_poly] = CalculateTVec(ts[-1], n_order, 1)
    Aeq[5, n_coef*(n_poly-1):n_coef*n_poly] = CalculateTVec(ts[-1], n_order, 2)

    beq[0:6] = np.array([p0,v0,a0,pe,ve,ae])
    
    neq = 5
    # 位置约束
    for i in range(n_poly-1):
        neq = neq + 1
        Aeq[neq, n_coef*(i+1):n_coef*(i+2)] = CalculateTVec(ts[i+1], n_order, 0)
        beq[neq] = waypts[i+1]
    # 连续性约束
    for i in range(n_poly-1):
        tvec_p = CalculateTVec(ts[i+1], n_order, 0)
        tvec_v = CalculateTVec(ts[i+1], n_order, 1)
        tvec_a = CalculateTVec(ts[i+1], n_order, 2)
        neq = neq + 1
        Aeq[neq, n_coef*i:n_coef*(i+1)] = tvec_p
        Aeq[neq, n_coef*(i+1):n_coef*(i+2)] = -tvec_p
        neq = neq + 1
        Aeq[neq, n_coef*i:n_coef*(i+1)] = tvec_v
        Aeq[neq, n_coef*(i+1):n_coef*(i+2)] = -tvec_v
        neq = neq + 1
        Aeq[neq, n_coef*i:n_coef*(i+1)] = tvec_a
        Aeq[neq, n_coef*(i+1):n_coef*(i+2)] = -tvec_a

    P = sparse.csc_matrix(H)
    q = np.zeros(n_poly*n_coef)
    A = sparse.csc_matrix(Aeq)
    L = beq
    U = beq

    new_prob = osqp.OSQP()
    new_prob.setup(P,q,A,L,U,alpha=1.6)
    res = new_prob.solve()
    polys = np.array(res.x)
    polys = np.reshape(polys,(n_poly, n_coef))

    return polys

def poly_val(poly, t, r):
    val = 0
    n = len(poly)
    c = np.zeros(n)
    for i in range(r,n):
        c[i] = math.factorial(i) / math.factorial(i-r) * (t**(i-r))
    val = np.sum(poly*c)
    return val

def polys_vals(polys, ts, tt, r):
    idx = 0
    N = len(tt)
    vals = np.zeros(N)
    for i in range(N):
        t = tt[i]
        if t<ts[idx]:
            vals[i] = 0
        else:
            while idx < len(ts) and t>ts[idx+1] + 0.0001:
                idx = idx+1
            vals[i] = poly_val(polys[idx,:], t, r)
    return vals

def plot_rect(center, r):
    p1 = center + np.array([-r, -r])
    p2 = center + np.array([-r, r])
    p3 = center + np.array([r, r])
    p4 = center + np.array([r, -r])
    plot_line(p1,p2)
    plot_line(p2,p3)
    plot_line(p3,p4)
    plot_line(p4,p1)

def plot_line(p1, p2):
    ps = np.array([p1, p2])
    plt.plot(ps[:,0], ps[:,1],'lightskyblue')

def sample_points(points, dis):
    n = len(points)
    spts = np.array([points[0,:]])
    for i in range(1, n):
        x1 = spts[-1, 0]
        y1 = spts[-1, 1]
        x2 = points[i, 0]
        y2 = points[i, 1]
        point_dis = np.hypot(x1-x2, y1-y2)
        if point_dis - dis < 1e-6 and i != n-1:
            continue
        elif point_dis - dis > 1e-6 and i != n-1:
            sx = x1 + (x2-x1) * dis / point_dis
            sy = y1 + (y2-y1) * dis / point_dis
            sp = spts[-1:] + (points[i,:]-spts[-1,:])* dis / point_dis
            spts = np.concatenate((spts, sp), axis = 0)#np.array([[sx, sy]])
        elif point_dis - dis == 1e-6 and i != n-1:
            spts = np.concatenate((spts, points[i,:]), axis = 0)
        else:
            spts = np.concatenate((spts, [points[i,:]]), axis = 0)
    return spts

def main(file_name):
    global_path = np.loadtxt(file_name)
    global_path = np.transpose(global_path)
    # delta_xy = global_path[1:,:] - global_path[:-1,:]
    # squre_sum = delta_xy[:,0]**2 + delta_xy[:,1]**2
    # distance = np.sum(squre_sum**0.5)
    # waypts = np.array([[0,0],[0,20],[2,22],[4.0,22],[6.0,22],[8,20],[8,0]])
    # waypts = np.array([[0,0],[1,2],[2,0],[4,5],[5,2]])
    waypts = np.array([[0,0],[1,2],[2,-1],[4,8],[5,2]])
    # waypts = np.array([[0,0],[1,0],[2,0],[4,0],[15,0]])
    # waypts = global_path
    # print(waypts)
    # print(global_path)
    sample_distance = 1.0
    spts = sample_points(global_path, sample_distance)
    # print(spts)
    waypts = spts
    v0 = np.array([0,0])
    a0 = np.array([0,0])
    v1 = np.array([0,0])
    a1 = np.array([0,0])

    delta_xy = waypts[1:,:] - waypts[:-1,:]
    squre_sum = delta_xy[:,0]**2 + delta_xy[:,1]**2
    distance = np.sum(squre_sum**0.5)
    k = 1.0
    T = distance * k # 5
    pt_num = len(waypts)

    n_order = 5
    lamda = 0  #
    r = 0.3
    step = 0.3
    new_waypts = waypts
    # new_waypts = np.array([waypts[0,:]])

    # for i in range(pt_num - 1):
    #     x1 = waypts[i, 0]
    #     y1 = waypts[i, 1]
    #     x2 = waypts[i+1, 0]
    #     y2 = waypts[i+1, 1]
    #     n = math.ceil(np.hypot(x1-x2, y1-y2) / step) + 1
    #     sample_pts = np.array([np.linspace(x1, x2, n),np.linspace(y1, y2, n)])
    #     sample_pts = np.transpose(sample_pts)
    #     new_waypts = np.concatenate((new_waypts, sample_pts[1:, :]), axis = 0)

    TS = ArrangeT(new_waypts, T)

    # Polys_X = minimum_snap_single_axis_simple(new_waypts[:,0], TS, n_order, v0[0], a0[0], v1[0], a1[0])
    # Polys_Y = minimum_snap_single_axis_simple(new_waypts[:,1], TS, n_order, v0[1], a0[1], v1[1], a1[1])
    Polys_X = minimum_snap_single_axis_corridor(new_waypts[:,0], TS, n_order, v0[0], a0[0], v1[0], a1[0], r)
    Polys_Y = minimum_snap_single_axis_corridor(new_waypts[:,1], TS, n_order, v0[1], a0[1], v1[1], a1[1], r)

    # Polys_X = minimum_snap_single_axis_guiding_path(new_waypts[:,0], TS, n_order, v0[0], a0[0], v1[0], a1[0], r, lamda)
    # Polys_Y = minimum_snap_single_axis_guiding_path(new_waypts[:,1], TS, n_order, v0[1], a0[1], v1[1], a1[1], r, lamda)

    # plt.figure(0)

    # plt.plot(new_waypts[:,0],new_waypts[:,1],'.', color = 'grey')
    # plt.plot(waypts[:,0],waypts[:,1],'*r')
    # for i in range(len(new_waypts)):
    #     plot_rect(new_waypts[i,:], r)

    # for i in range(len(polys_test)):
    # tt = np.arange(0,T, 0.01)
    # print(Polys_X, Polys_Y)
    # xx = polys_vals(Polys_X, TS, tt, 0)
    # yy = polys_vals(Polys_Y, TS, tt, 0)
    # plt.plot(xx,yy,'r')

    # plt.show()
    # print(Polys_X)
    # print(Polys_Y)
    # '''
    if show_animation:
        plt.figure(1)
        plt.plot(new_waypts[:,0],new_waypts[:,1],'.', color = 'grey')
        Ti = np.arange(0, T, 0.01)
        Xi = polys_vals(Polys_X, TS, Ti, 0)
        Yi = polys_vals(Polys_Y, TS, Ti, 0)
        plt.plot(Xi,Yi,'r')
        plt.axis('scaled')

        plt.figure(2)
        Ti = np.arange(0,T, 0.01)
        Xi = polys_vals(Polys_X, TS, Ti, 0)
        Yi = polys_vals(Polys_Y, TS, Ti, 0)
        vXi = polys_vals(Polys_X, TS, Ti, 1)
        vYi = polys_vals(Polys_Y, TS, Ti, 1)
        aXi = polys_vals(Polys_X, TS, Ti, 2)
        aYi = polys_vals(Polys_Y, TS, Ti, 2)
        k = abs(vXi * aYi - aXi * vYi) / (vXi*vXi+vYi*vYi)**(1.5)
        # print(k)
        plt.plot(k[1:-1])

        plt.figure(3)
        plt.subplot(3,2,1)
        plt.plot(Ti,Xi)
        plt.subplot(3,2,2)
        plt.plot(Ti,Yi)
        plt.subplot(3,2,3)
        plt.plot(Ti,vXi)
        plt.subplot(3,2,4)
        plt.plot(Ti,vYi)
        plt.subplot(3,2,5)
        plt.plot(Ti,aXi)
        plt.subplot(3,2,6)
        plt.plot(Ti,aYi)

        plt.figure(4)
        ax = plt.gca(projection = '3d')
        ax.plot(Xi,Yi,k)
        plt.plot(Xi,Yi)

        plt.show()
    # '''

if __name__ == '__main__':
    show_animation = True
    file_name = "temp.txt"
    main(file_name)