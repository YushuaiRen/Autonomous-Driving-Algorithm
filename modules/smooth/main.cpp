#include <iostream>

#include "osqp/osqp.h"
#include "modules/common/include/vec2d.h"

int main(int argc, char **argv) {
    common::math::Vec2d p(1,2);
    std::cout << p.x() << " " << p.y() << std::endl;
    // Load problem data
    c_float P_x[3] = {4.0, 1.0, 2.0, };
    c_float P_x_new[3] = {5.0, 1.5, 1.0, };
    c_int P_nnz = 3;
    c_int P_i[3] = {0, 0, 1, };
    c_int P_p[3] = {0, 1, 3, };
    c_float q[2] = {1.0, 1.0, };
    c_float q_new[2] = {2.0, 3.0, };
    c_float A_x[4] = {1.0, 1.0, 1.0, 1.0, };
    c_float A_x_new[4] = {1.2, 1.5, 1.1, 0.8, };
    c_int A_nnz = 4;
    c_int A_i[4] = {0, 1, 0, 2, };
    c_int A_p[3] = {0, 2, 4, };
    c_float l[3] = {1.0, 0.0, 0.0, };
    c_float l_new[3] = {2.0, -1.0, -1.0, };
    c_float u[3] = {1.0, 0.7, 0.7, };
    c_float u_new[3] = {2.0, 2.5, 2.5, };
    c_int n = 2;
    c_int m = 3;

    // Exitflag
    c_int exitflag = 0;

    // Workspace structures
    OSQPWorkspace *work;
    OSQPSettings  *settings = (OSQPSettings *)c_malloc(sizeof(OSQPSettings));
    OSQPData      *data     = (OSQPData *)c_malloc(sizeof(OSQPData));

    // Populate data
    if (data) {
        data = (OSQPData *)c_malloc(sizeof(OSQPData));
        data->n = n;
        data->m = m;
        data->P = csc_matrix(data->n, data->n, P_nnz, P_x, P_i, P_p);
        data->q = q;
        data->A = csc_matrix(data->m, data->n, A_nnz, A_x, A_i, A_p);
        data->l = l;
        data->u = u;
    }

    // Define Solver settings as default
    if (settings) osqp_set_default_settings(settings);

    // Setup workspace
    exitflag = osqp_setup(&work, data, settings);

    // Solve problem
    osqp_solve(work);

    // Update problem
    // NB: Update only upper triangular part of P
    osqp_update_P(work, P_x_new, OSQP_NULL, 3);
    osqp_update_A(work, A_x_new, OSQP_NULL, 4);

    // Solve updated problem
    osqp_solve(work);

    // Cleanup
    if (data) {
        if (data->A) c_free(data->A);
        if (data->P) c_free(data->P);
        c_free(data);
    }
    if (settings) c_free(settings);

    return exitflag;
};