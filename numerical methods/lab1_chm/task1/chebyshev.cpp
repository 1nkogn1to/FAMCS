#include <iostream>
#include <iomanip>
#include <time.h>
#include <cmath>

using namespace std;

double f(double x) { return pow(sin(x), 2) * exp(x) - 3 * x + 1.; }
double df(double x) {
    double _2x = 2. * x;
    return -3. + exp(x) * (sin(_2x) + 0.5 - 0.5 * cos(_2x));
}
double ddf(double x) {
    double _2x = 2. * x;
    return exp(x) * (2 * sin(_2x) + 0.5 + 1.5 * cos(_2x));
}
double g(double x) {
    double _f = f(x), _df = df(x);
    return x - _f / _df - pow(_f, 2) * ddf(x) / (2 * pow(_df, 3));
}

unsigned int Solution(double accuracy = 1e-6) {
    double x0 = 0.3;
    int counter = 0;
    unsigned int t1 = clock();
    for ( ; ; ) {
        double x_curr = g(x0);
        cout << ++counter << " iteration: " << setprecision(15) << x_curr << "\n";
        if (abs(x_curr - x0) <= accuracy) {
            x0 = x_curr;
            break;
        }
        x0 = x_curr;
    }
    unsigned int t2 = clock();
    cout << "Solution: " << setprecision(15) << x0 << "\n";
    cout << "Number of iterations: " << counter << "\n";
    return t2 - t1;
}

int main() {
    unsigned int sum = 0;
    for (int i = 0; i < 100; ++i) {
        sum += Solution();
    }
    cout << sum / 100.;

    return 0;
}
//1.04