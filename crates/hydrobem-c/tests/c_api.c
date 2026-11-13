/*
 * A C program on the header and the DLL alone (tests/python/test_c_abi.py builds and runs it):
 * the analytic cylinder made, solved with a progress callback and its added mass read; an error
 * with its message; the columns of a body; the tables freed. Exit status 0 when every check
 * holds.
 */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "hydrobem.h"

static int failures = 0;

#define CHECK(cond)                                                                    \
    do {                                                                               \
        if (!(cond)) {                                                                 \
            fprintf(stderr, "%s:%d: %s failed (%s)\n", __FILE__, __LINE__, #cond,      \
                    hb_last_error_message());                                          \
            failures++;                                                                \
        }                                                                              \
    } while (0)

static int32_t on_progress(const HbProgress *p, void *user) {
    int *calls = (int *)user;
    (*calls)++;
    return p->frequencies_done <= p->frequencies_total;
}

int main(void) {
    CHECK(hb_abi_version() == HB_ABI_VERSION);

    HbModel *m = NULL;
    CHECK(hb_model_new(&m) == HB_OK);

    const char *names[] = {"radius", "draft"};
    const double values[] = {1.0, 0.5};
    const int32_t rows[] = {4, 4, 3, 3, 3, 4, 4, 2, 3, 3, 3, 4};
    HbAnalyticShape shape;
    memset(&shape, 0, sizeof shape);
    shape.shape = "circular_cylinder";
    shape.dimension_names = names;
    shape.dimension_values = values;
    shape.dimensions = 2;
    shape.mapping = "uniform";
    shape.discretization = rows;
    shape.discretization_rows = 2;
    shape.reference_length = 1.0;
    shape.symmetry = "xz_and_yz";
    uint64_t mesh = 0, root = 0, body = 0, options = 0, environment = 0;
    CHECK(hb_model_analytic_mesh(m, "cyl", &shape, &mesh) == HB_OK);
    CHECK(hb_object(m, "", &root) == HB_OK);
    CHECK(hb_add(m, root, "bodies", NULL, "b", -1, &body) == HB_OK);
    CHECK(hb_set_text(m, body, "mesh", "cyl") == HB_OK);
    CHECK(hb_object(m, "options", &options) == HB_OK);
    CHECK(hb_set_bool(m, options, "higher_order", 1) == HB_OK);
    CHECK(hb_object(m, "environment", &environment) == HB_OK);
    const double omegas[] = {1.0, 2.0};
    const double headings[] = {0.0};
    size_t two = 2, one = 1;
    CHECK(hb_set_reals(m, environment, "omegas", omegas, &two, 1) == HB_OK);
    CHECK(hb_set_reals(m, environment, "headings", headings, &one, 1) == HB_OK);

    /* an error: a negative frequency, refused with a message naming the item */
    const double negative[] = {-1.0};
    CHECK(hb_set_reals(m, environment, "omegas", negative, &one, 1) == HB_ERR_VALUE);
    CHECK(strstr(hb_last_error_message(), "omegas") != NULL);
    /* a null path, refused at the boundary; a negative place */
    uint64_t fs = 0;
    CHECK(hb_model_load_fdf(m, "fs", NULL, NULL, &fs) == HB_ERR_ARGUMENT);
    CHECK(hb_move(m, body, -1) == HB_ERR_ARGUMENT);

    int calls = 0;
    HbResults *r = NULL;
    CHECK(hb_model_solve(m, on_progress, &calls, &r) == HB_OK);
    CHECK(calls > 0);

    HbColumn a;
    CHECK(hb_results_quantity(r, "added_mass", &a) == HB_OK);
    CHECK(a.dtype == HB_F32 && a.ndim == 3 && a.shape[0] == 2 && a.shape[1] == 6 && a.shape[2] == 6);
    /* the heave added mass of the cylinder: positive, of the order of its displaced mass */
    const float *am = (const float *)a.data;
    double a33 = am[2 * 6 + 2];
    CHECK(a33 > 100.0 && a33 < 5000.0);

    HbTable *t = NULL;
    HbColumn v;
    CHECK(hb_body_hydrostatics(m, body, &t) == HB_OK);
    CHECK(hb_table_find(t, "volumes", &v) == HB_OK);
    CHECK(v.dtype == HB_F64 && v.count == 3 && fabs(((const double *)v.data)[2] - 3.14159265 * 0.5) < 1e-6);
    CHECK(hb_table_find(t, "no_such_column", &v) == HB_ERR_ARGUMENT);
    hb_table_free(t);

    /* the six rigid coordinates of the body, and its six radiation columns */
    CHECK(hb_results_body_columns(r, 0, &t) == HB_OK);
    CHECK(hb_table_find(t, "coordinates", &v) == HB_OK && v.dtype == HB_I64 && v.count == 6);
    CHECK(hb_table_find(t, "radiation_columns", &v) == HB_OK && v.count == 6);
    hb_table_free(t);

    char *route = NULL;
    CHECK(hb_results_route(r, &route) == HB_OK && route != NULL && strlen(route) > 0);
    hb_string_free(route);

    hb_results_free(r);
    CHECK(hb_model_free(m) == HB_OK);
    if (failures == 0) {
        printf("c_api: ok (a33 = %.3f, %d progress reports)\n", a33, calls);
    }
    return failures == 0 ? 0 : 1;
}
