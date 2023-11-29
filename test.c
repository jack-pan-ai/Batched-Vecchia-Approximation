#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_fft_complex.h>
#include <gsl/gsl_cblas.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/time.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
double matern_kernel(double effective_range, double variance, double smoothness) {
    gsl_set_error_handler_off () ;
    double i;
    double expr = 0.0;
    double con = 0.0;
    double value = 0.0;
    int int_val = 0;
    double sigma_square = variance * variance;

    con = pow(2,(smoothness-1)) * tgamma(smoothness);
    con = 1.0/con;
    con = sigma_square * con;

    for (i=0.0;i<10.0;i+=0.0000001)
    {

        expr = effective_range/i;
        value = (double)(con * pow(expr, smoothness)*gsl_sf_bessel_Knu(smoothness,expr)); // Matern Function
        // printf("i: %f - value: %f\n", i, value);
        int_val = value * 100000;	
        if (int_val == 5000)
            return i;
    }
    return -1;
}




double pow_exp_kernel(double effective_range, double variance, double smoothness) {
    double i;
    double expr  = 0.0;
    double expr1 = 0.0;
    double value = 0.0;
    int int_val = 0;
    double sigma_square = variance * variance;


    for (i=0.0;i<10.0;i+=0.0000001)
    {

        expr = effective_range;
        expr1 = pow(expr, smoothness);
        value = (double)(sigma_square *  exp(-(expr1/i)));        //power-exp kernel

        //printf("i: %f - value: %f\n", i, value);
        int_val = value * 100000;
        if (int_val == 5000)
            return i;
    }
    return -1;
}


double pow_exp_nuggets_kernel(double effective_range, double variance, double smoothness, double noise) {
    double i;
    double expr  = 0.0;
    double expr1 = 0.0;
    double value = 0.0;
    int int_val = 0;
    double sigma_square = variance * variance;


    for (i=0.0;i<10.0;i+=0.0000001)
    {

        expr = effective_range;
        expr1 = pow(expr, smoothness);
        value = (double)(sigma_square *  exp(-(expr1/i))) + noise;        //power-exp kernel

        //printf("i: %f - value: %f\n", i, value);
        int_val = value * 100000;
        if (int_val == 5000)
            return i;
    }
    return -1;
}


int main(int argc, char **argv) {

    double range_matern = 0.0;
    double delta = 0.0;
    double delta2 = 0.0;

    range_matern = matern_kernel(0.1, 1.5, 0.5);
    printf ("effective range (matern) : (eff_range = 0.1, var = 1.5, smoothness = 0.5) -> range %f\n", range_matern);

    range_matern = matern_kernel(0.1, 1.5, 1.5);
    printf ("effective range (matern) : (eff_range = 0.1, var = 1.5, smoothness = 1.5) -> range %f\n", range_matern);
    
    range_matern = matern_kernel(0.1, 1.5, 2.5);
    printf ("effective range (matern) : (eff_range = 0.1, var = 1.5, smoothness = 2.5) -> range %f\n", range_matern);

    printf("======================================\n");

    range_matern = matern_kernel(0.3, 1.5, 0.5);
    printf ("effective range (matern) : (eff_range = 0.3, var = 1.5, smoothness = 0.5) -> range %f\n", range_matern);

    range_matern = matern_kernel(0.3, 1.5, 1.5);
    printf ("effective range (matern) : (eff_range = 0.3, var = 1.5, smoothness = 1.5) -> range %f\n", range_matern);

    range_matern = matern_kernel(0.3, 1.5, 2.5);
    printf ("effective range (matern) : (eff_range = 0.3, var = 1.5, smoothness = 2.5) -> range %f\n", range_matern);
    
    printf("======================================\n");

    range_matern = matern_kernel(0.8, 1.5, 0.5);
    printf ("effective range (matern) : (eff_range = 0.8, var = 1.5, smoothness = 0.5) -> range %f\n", range_matern);
    
    range_matern = matern_kernel(0.8, 1.5, 1.5);
    printf ("effective range (matern) : (eff_range = 0.8, var = 1.5, smoothness = 1.5) -> range %f\n", range_matern);
    
    range_matern = matern_kernel(0.8, 1.5, 2.5);
    printf ("effective range (matern) : (eff_range = 0.8, var = 1.5, smoothness = 2.5) -> range %f\n", range_matern);
    
}