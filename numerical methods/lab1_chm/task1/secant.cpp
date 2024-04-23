#include <iostream>
#include <iomanip>
#include <time.h>
#include <cmath>

using namespace std;

double f(double x) { return pow(sin(x), 2) * exp(x) - 3 * x + 1; }

unsigned int Solution(double accuracy = 1e-6) {
    double x0 = 0.3, x1 = 0.4, f0 = f(x0);
    int counter = 0;

    unsigned int t1 = clock();
    for ( ; ; ) {
        double f1 = f(x1), x2 = x1 - f1 * (x1 - x0) / (f1 - f0);
        cout << ++counter << " iteration: " << setprecision(15) << x2 << "\n";
        if (abs(x2 - x1) <= accuracy) {
            x1 = x2;
            break;
        }
        x0 = x1;
        x1 = x2;
        f0 = f1;
    }
    unsigned int t2 = clock();
    cout << "Solution: " << setprecision(15) << x1 << "\n";
    cout << "Number of iterations: " << counter << "\n";
    return t2 - t1;
}

int main() {
    unsigned int sum = 0;
    for (int i = 0; i < 10000; ++i) {
        sum += Solution();
    }
    cout << sum / 10000.;
    return 0;
}
//1.31